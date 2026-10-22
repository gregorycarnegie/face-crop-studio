//! Click two eye points per face, to turn Open Images boxes into training labels.
//!
//!   zig run -O ReleaseSafe tools/dataset/label_eyes.zig -- <data_dir> [--split train] [--port 8765] [--count 2000]
//!
//! Open Images has hand-drawn face boxes but no landmarks, and `fcs-core::face_cropper` aligns
//! crops on the eyes, so a replacement detector needs eye points. This serves the boxes one at a
//! time, zoomed, and records two clicks each. The page itself is label_eyes.html, embedded.
//!
//! `--split` picks which Open Images boxes to queue, defaulting to `validation,test`. Those two
//! splits are the held-out test set, so once they are labelled, training labels should come from
//! `--split train` instead -- otherwise the only clean yardstick gets spent. Images absent from
//! <data_dir>/images are fetched from the public mirror as they come up and cached there, which
//! is what makes the train split usable without downloading all 278,655 of its images first.
//! Each saved record carries the split it came from, so train and test labels stay separable.
//!
//! Labels append to <data_dir>/eye_labels.jsonl as you go, so a refresh, a crash or a week off
//! loses nothing: on restart, anything already answered is dropped from the queue. The last line
//! for a face wins, which is what makes "go back" work.
//!
//! Coordinates are stored normalised (0-1 of image width/height) so they stay valid whatever
//! size the image is served at. Clicks are recorded as viewer-left and viewer-right, matching
//! what the labeller sees, and nothing here converts them to a model's ordering.
//!
//! For whoever writes that conversion: measured over 10,880 large detections, YuNet's
//! `landmarks[0]` sits left of `landmarks[1]` on screen in 99.8% of them -- the usual
//! RetinaFace/YuNet order, where index 0 is the subject's own right eye. So on an upright face
//! `viewer_left` is `landmarks[0]`.
//!
//! That mapping holds only while the head's roll stays within +/-90 degrees. Rotate a face far
//! enough and the subject's right eye crosses to the viewer's right, so the two exchange places;
//! screen position stops implying which eye it is. Records with `"upside_down": true` are those
//! faces -- spotted because the clicks arrived right-to-left on screen, then confirmed by eye --
//! and their points are stored in screen order like all the others. A converter therefore takes
//! screen order from the coordinates and anatomy from that flag. 34 of the first 1,599 labelled
//! faces needed it, so it is not a rare edge case worth ignoring.
//!
//! A second flag, `"steep_roll": true`, marks pairs whose eyes sit within 8% of the box width of
//! each other horizontally. There the x-order carries almost no information about which side is
//! which, however the head is turned, so read left/right identity as unknown rather than
//! guessing. The points themselves are real and still worth training a landmark head on.
//!
//! ponytail: single user, no auth, binds to localhost only. If two people ever label at once,
//! give each their own port and jsonl and merge afterwards.
const std = @import("std");
const Io = std.Io;
const u = @import("util.zig");
const json = std.json;

const PAGE = @embedFile("label_eyes.html");
const MIN_WIDTH_AT_640 = 32.0 / 640.0; // faces too small to crop are not worth labelling
const MIRROR = "https://open-images-dataset.s3.amazonaws.com/{s}/{s}.jpg";
const PREFETCH = 8; // images pulled ahead of the one on screen, so clicking never waits on a download

const SplitFile = struct { name: []const u8, file: []const u8 };
const SPLIT_FILES = [_]SplitFile{
    .{ .name = "validation", .file = "faces-validation-annotations-bbox.csv" },
    .{ .name = "test", .file = "faces-test-annotations-bbox.csv" },
    .{ .name = "train", .file = "faces-oidv6-train-annotations-bbox.csv" },
};

const Face = struct { id: []const u8, image: []const u8, split: []const u8, box: [4]f64 };

const State = struct {
    io: Io,
    data: []const u8,
    out_path: []const u8,
    queue: []const Face,
    total: usize,
    splits: std.StringHashMapUnmanaged([]const u8),
    order: std.StringHashMapUnmanaged(usize),
    queue_json: []const u8,
    save_mutex: Io.Mutex = .init,
    inflight_mutex: Io.Mutex = .init,
    inflight: std.StringHashMapUnmanaged(void) = .empty,
};

fn isImageId(s: []const u8) bool {
    if (s.len != 16) return false;
    for (s) |c| if (!std.ascii.isDigit(c) and !(c >= 'a' and c <= 'f')) return false;
    return true;
}

fn buildQueue(gpa: std.mem.Allocator, io: Io, data: []const u8, count: usize, splits: []const []const u8) ![]Face {
    var faces: std.ArrayList(Face) = .empty;
    for (splits) |split| {
        const file = for (SPLIT_FILES) |s| {
            if (std.mem.eql(u8, s.name, split)) break s.file;
        } else unreachable;
        const path = try std.fs.path.join(gpa, &.{ data, file });
        const csv = u.Csv.load(gpa, io, path) catch |e| switch (e) {
            error.FileNotFound => {
                std.debug.print("no {s} boxes at {s}, skipping that split\n", .{ split, file });
                continue;
            },
            else => return e,
        };
        const c_id, const c_x0, const c_x1, const c_y0, const c_y1 = .{ csv.col("ImageID"), csv.col("XMin"), csv.col("XMax"), csv.col("YMin"), csv.col("YMax") };
        const c_group, const c_dep = .{ csv.col("IsGroupOf"), csv.col("IsDepiction") };
        for (csv.rows) |r| {
            if (std.mem.eql(u8, r[c_group], "1") or std.mem.eql(u8, r[c_dep], "1")) continue;
            const xmin = try std.fmt.parseFloat(f64, r[c_x0]);
            const xmax = try std.fmt.parseFloat(f64, r[c_x1]);
            if (xmax - xmin < MIN_WIDTH_AT_640) continue;
            const ymin = try std.fmt.parseFloat(f64, r[c_y0]);
            const ymax = try std.fmt.parseFloat(f64, r[c_y1]);
            var id: Io.Writer.Allocating = .init(gpa);
            try id.writer.print("{s}:", .{r[c_id]});
            try u.fixed(&id.writer, xmin, 6);
            try id.writer.writeAll(",");
            try u.fixed(&id.writer, ymin, 6);
            try faces.append(gpa, .{ .id = id.written(), .image = r[c_id], .split = split, .box = .{ xmin, ymin, xmax, ymax } });
        }
    }
    // Seeded, so the queue is the same every run -- and deliberately not filtered by which
    // images happen to be on disk, which would make the order depend on the download history.
    var rng = u.PyRandom.init(0);
    rng.shuffle(Face, faces.items);
    return faces.items[0..@min(count, faces.items.len)];
}

fn writeFace(w: *Io.Writer, f: Face) !void {
    try w.print("{{\"id\": \"{s}\", \"image\": \"{s}\", \"split\": \"{s}\", \"box\": [", .{ f.id, f.image, f.split });
    for (f.box, 0..) |v, i| {
        if (i > 0) try w.writeAll(", ");
        try u.pyFloat(w, v);
    }
    try w.writeAll("]}");
}

/// Python's `json.dumps` for a parsed value (default separators, ASCII-only output).
fn dump(w: *Io.Writer, v: json.Value) !void {
    switch (v) {
        .null => try w.writeAll("null"),
        .bool => |b| try w.writeAll(if (b) "true" else "false"),
        .integer => |n| try w.print("{d}", .{n}),
        .float => |f| try u.pyFloat(w, f),
        .number_string => |s| try w.writeAll(s),
        .string => |s| try dumpString(w, s),
        .array => |a| {
            try w.writeAll("[");
            for (a.items, 0..) |item, i| {
                if (i > 0) try w.writeAll(", ");
                try dump(w, item);
            }
            try w.writeAll("]");
        },
        .object => |o| {
            try w.writeAll("{");
            for (o.keys(), o.values(), 0..) |k, val, i| {
                if (i > 0) try w.writeAll(", ");
                try dumpString(w, k);
                try w.writeAll(": ");
                try dump(w, val);
            }
            try w.writeAll("}");
        },
    }
}

fn dumpString(w: *Io.Writer, s: []const u8) !void {
    try w.writeAll("\"");
    var view = std.unicode.Utf8View.init(s) catch return w.writeAll("\"");
    var it = view.iterator();
    while (it.nextCodepoint()) |cp| switch (cp) {
        '"' => try w.writeAll("\\\""),
        '\\' => try w.writeAll("\\\\"),
        '\n' => try w.writeAll("\\n"),
        '\r' => try w.writeAll("\\r"),
        '\t' => try w.writeAll("\\t"),
        0x20...0x21, 0x23...0x5b, 0x5d...0x7e => try w.writeAll(&.{@intCast(cp)}),
        else => if (cp >= 0x10000) {
            const c = cp - 0x10000;
            try w.print("\\u{x:0>4}\\u{x:0>4}", .{ 0xD800 + (c >> 10), 0xDC00 + (c & 0x3FF) });
        } else try w.print("\\u{x:0>4}", .{cp}),
    };
    try w.writeAll("\"");
}

/// One image's bytes, downloading and caching it when it is not on disk yet.
fn fetchImage(state: *State, gpa: std.mem.Allocator, stem: []const u8) ?[]u8 {
    const path = std.fmt.allocPrint(gpa, "{s}/images/{s}.jpg", .{ state.data, stem }) catch return null;
    defer gpa.free(path);
    if (u.readFile(gpa, state.io, path)) |bytes| return bytes else |_| {}
    const split = state.splits.get(stem) orelse return null;
    const url = std.fmt.allocPrint(gpa, MIRROR, .{ split, stem }) catch return null;
    defer gpa.free(url);
    var client: std.http.Client = .{ .allocator = gpa, .io = state.io };
    defer client.deinit();
    var body: Io.Writer.Allocating = .init(gpa);
    const res = client.fetch(.{ .location = .{ .url = url }, .response_writer = &body.writer }) catch return null;
    if (res.status != .ok) return null;
    if (std.fs.path.dirname(path)) |parent| Io.Dir.cwd().createDirPath(state.io, parent) catch {};
    // Write then rename: a download cut halfway must not leave a broken image cached, which
    // would then be served from disk forever.
    const part = std.fmt.allocPrint(gpa, "{s}.part", .{path}) catch return null;
    defer gpa.free(part);
    u.writeFile(state.io, part, body.written()) catch return null;
    Io.Dir.rename(Io.Dir.cwd(), part, Io.Dir.cwd(), path, state.io) catch return null;
    return body.toOwnedSlice() catch null;
}

fn prefetchWorker(state: *State, stem: []const u8) void {
    const gpa = std.heap.smp_allocator;
    if (fetchImage(state, gpa, stem)) |b| gpa.free(b);
    state.inflight_mutex.lockUncancelable(state.io);
    _ = state.inflight.remove(stem);
    state.inflight_mutex.unlock(state.io);
}

fn prefetchAfter(state: *State, stem: []const u8) void {
    const gpa = std.heap.smp_allocator;
    const position = state.order.get(stem) orelse return;
    const end = @min(state.queue.len, position + 1 + PREFETCH);
    for (state.queue[@min(position + 1, end)..end]) |face| {
        const path = std.fmt.allocPrint(gpa, "{s}/images/{s}.jpg", .{ state.data, face.image }) catch return;
        defer gpa.free(path);
        if (Io.Dir.cwd().access(state.io, path, .{})) |_| continue else |_| {}
        state.inflight_mutex.lockUncancelable(state.io);
        const fresh = if (state.inflight.getOrPut(gpa, face.image)) |e| !e.found_existing else |_| false;
        state.inflight_mutex.unlock(state.io);
        if (!fresh) continue;
        const t = std.Thread.spawn(.{}, prefetchWorker, .{ state, face.image }) catch continue;
        t.detach();
    }
}

fn appendLabel(state: *State, gpa: std.mem.Allocator, line: []const u8) !void {
    state.save_mutex.lockUncancelable(state.io);
    defer state.save_mutex.unlock(state.io);
    const file = try Io.Dir.cwd().createFile(state.io, state.out_path, .{ .truncate = false });
    defer file.close(state.io);
    const end = try file.length(state.io);
    const bytes = try std.fmt.allocPrint(gpa, "{s}\n", .{line});
    defer gpa.free(bytes);
    try file.writePositionalAll(state.io, bytes, end);
}

fn serve(state: *State, stream: std.Io.net.Stream) void {
    const gpa = std.heap.smp_allocator;
    defer stream.close(state.io);
    var in_buf: [16 * 1024]u8 = undefined;
    var out_buf: [16 * 1024]u8 = undefined;
    var reader = stream.reader(state.io, &in_buf);
    var writer = stream.writer(state.io, &out_buf);
    var http_server = std.http.Server.init(&reader.interface, &writer.interface);
    while (true) {
        var request = http_server.receiveHead() catch return;
        handle(state, gpa, &request) catch return;
        if (!request.head.keep_alive) return;
    }
}

fn handle(state: *State, gpa: std.mem.Allocator, request: *std.http.Server.Request) !void {
    const target = request.head.target;
    const path = target[0 .. std.mem.indexOfScalar(u8, target, '?') orelse target.len];
    switch (request.head.method) {
        .GET => {
            if (std.mem.eql(u8, path, "/") or std.mem.eql(u8, path, "/index.html"))
                return request.respond(PAGE, .{ .extra_headers = &.{.{ .name = "content-type", .value = "text/html; charset=utf-8" }} });
            if (std.mem.eql(u8, path, "/queue.json"))
                return request.respond(state.queue_json, .{ .extra_headers = &.{.{ .name = "content-type", .value = "application/json" }} });
            if (std.mem.startsWith(u8, path, "/img/")) {
                const stem = std.mem.trimEnd(u8, path["/img/".len..], ".jpg");
                if (!isImageId(stem)) return respondText(request, .not_found, "no");
                prefetchAfter(state, stem);
                const body = fetchImage(state, gpa, stem) orelse return respondText(request, .not_found, "image not available");
                defer gpa.free(body);
                return request.respond(body, .{ .extra_headers = &.{.{ .name = "content-type", .value = "image/jpeg" }} });
            }
            return respondText(request, .not_found, "no");
        },
        .POST => {
            if (!std.mem.eql(u8, path, "/save")) return respondText(request, .not_found, "no");
            var arena_state = std.heap.ArenaAllocator.init(gpa);
            defer arena_state.deinit();
            const arena = arena_state.allocator();
            var body_buf: [4096]u8 = undefined;
            const body_reader = request.readerExpectNone(&body_buf);
            const body = try body_reader.allocRemaining(arena, .limited(1 << 20));
            const parsed = json.parseFromSliceLeaky(json.Value, arena, if (body.len == 0) "{}" else body, .{}) catch return respondText(request, .bad_request, "bad json");
            if (parsed != .object) return respondText(request, .bad_request, "bad json");
            var rec = parsed.object;
            // Stamped here rather than taken from the page: the split decides whether a label is
            // training data or test data, so it comes from the queue the server built.
            const image = if (rec.get("image")) |v| (if (v == .string) v.string else "") else "";
            try rec.put(arena, "split", .{ .string = state.splits.get(image) orelse "unknown" });
            var line: Io.Writer.Allocating = .init(arena);
            try dump(&line.writer, .{ .object = rec });
            try appendLabel(state, gpa, line.written());
            return respondText(request, .ok, "ok");
        },
        else => return respondText(request, .not_found, "no"),
    }
}

fn respondText(request: *std.http.Server.Request, status: std.http.Status, text: []const u8) !void {
    try request.respond(text, .{ .status = status, .extra_headers = &.{.{ .name = "content-type", .value = "text/plain" }} });
}

pub fn main(init: std.process.Init) !void {
    const gpa = init.arena.allocator();
    const io = init.io;
    const args = try init.minimal.args.toSlice(gpa);
    var data_arg: ?[]const u8 = null;
    var split_arg: []const u8 = "validation,test";
    var port: u16 = 8765;
    var count: usize = 2000;
    var i: usize = 1;
    while (i < args.len) : (i += 1) {
        if (std.mem.eql(u8, args[i], "--split") and i + 1 < args.len) {
            i += 1;
            split_arg = args[i];
        } else if (std.mem.eql(u8, args[i], "--port") and i + 1 < args.len) {
            i += 1;
            port = try std.fmt.parseInt(u16, args[i], 10);
        } else if (std.mem.eql(u8, args[i], "--count") and i + 1 < args.len) {
            i += 1;
            count = try std.fmt.parseInt(usize, args[i], 10);
        } else data_arg = args[i];
    }
    const data = try Io.Dir.cwd().realPathFileAlloc(io, data_arg orelse return error.Usage, gpa);

    var splits: std.ArrayList([]const u8) = .empty;
    var sit = std.mem.splitScalar(u8, split_arg, ',');
    while (sit.next()) |p| {
        const s = std.mem.trim(u8, p, " \t");
        if (s.len == 0) continue;
        const known = for (SPLIT_FILES) |f| {
            if (std.mem.eql(u8, f.name, s)) break true;
        } else false;
        if (!known) {
            std.debug.print("unknown split {s}; pick from validation, test, train\n", .{s});
            std.process.exit(2);
        }
        try splits.append(gpa, s);
    }

    const out_path = try std.fs.path.join(gpa, &.{ data, "eye_labels.jsonl" });
    const queue = try buildQueue(gpa, io, data, count, splits.items);
    var done = u.lastWins(gpa, io, out_path, "id") catch |e| switch (e) {
        error.FileNotFound => std.StringArrayHashMapUnmanaged(json.Value).empty,
        else => return e,
    };
    _ = &done;
    var remaining: std.ArrayList(Face) = .empty;
    for (queue) |f| if (!done.contains(f.id)) try remaining.append(gpa, f);

    var state: State = .{
        .io = io,
        .data = data,
        .out_path = out_path,
        .queue = remaining.items,
        .total = queue.len,
        .splits = .empty,
        .order = .empty,
        .queue_json = "",
    };
    for (queue) |f| try state.splits.put(gpa, f.image, f.split);
    for (remaining.items, 0..) |f, k| try state.order.put(gpa, f.image, k);
    var qj: Io.Writer.Allocating = .init(gpa);
    try qj.writer.writeAll("{\"queue\": [");
    for (remaining.items, 0..) |f, k| {
        if (k > 0) try qj.writer.writeAll(", ");
        try writeFace(&qj.writer, f);
    }
    try qj.writer.print("], \"total\": {d}}}", .{queue.len});
    state.queue_json = qj.written();

    var to_fetch: usize = 0;
    for (remaining.items) |f| {
        const p = try std.fmt.allocPrint(gpa, "{s}/images/{s}.jpg", .{ data, f.image });
        Io.Dir.cwd().access(io, p, .{}) catch {
            to_fetch += 1;
        };
    }
    var labelled: usize = 0;
    for (done.values()) |r| labelled += @intFromBool(!u.truthy(u.optField(r, "skipped")));

    var so: u.Stdout = undefined;
    so.init(io);
    const w = so.w();
    const joined = try std.mem.join(gpa, "+", splits.items);
    try w.print("{d} faces in the queue from {s}, {d} already answered ({d} with eye points), {d} to go\n", .{ queue.len, joined, done.count(), labelled, remaining.items.len });
    if (to_fetch > 0) try w.print("{d} of those images are not local yet and will be fetched as they come up, {d} ahead of the screen\n", .{ to_fetch, PREFETCH });
    try w.print("labels append to {s}\nopen http://127.0.0.1:{d}/  (ctrl-c to stop; progress is saved)\n", .{ out_path, port });
    try w.flush();

    const address = try Io.net.IpAddress.parseIp4("127.0.0.1", port);
    var server = try address.listen(io, .{ .reuse_address = true });
    defer server.deinit(io);
    while (true) {
        const stream = server.accept(io) catch continue;
        const t = std.Thread.spawn(.{}, serve, .{ &state, stream }) catch {
            stream.close(io);
            continue;
        };
        t.detach();
    }
}
