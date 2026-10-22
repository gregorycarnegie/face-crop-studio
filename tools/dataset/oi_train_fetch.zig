//! Pick N Open Images train images with croppable faces, record their credits, download them.
//!
//!   zig run -O ReleaseSafe tools/dataset/oi_train_fetch.zig -- <openimages_dir> [--count 80000] [--workers 32] [--select-only]
//!
//! Why 80,000: that is about 164,000 croppable faces, the scale of WIDER FACE (159k), which is
//! what SCRFD and YuNet were trained on. The label-count curve (CURVE_RESULTS.md) found images,
//! not eye labels, to be the constraint; this is the run that tests it at industry scale.
//!
//! Selection rules, each for a reason:
//!
//! * **Train split only.** Validation and test are the held-out test set and carry clicked eyes.
//! * **At least one croppable face** -- real (not a depiction), not a group-of box, and at least
//!   5% of image width (32 px at the detector's 640 input), the same bar as DATA_CARD.md.
//! * **Rotation recorded as 0.0.** Nothing in this pipeline reads Open Images' `Rotation` column,
//!   and ~1% of train images are marked 90/180/270 while 14% record nothing. Rather than guess
//!   which frame the boxes were drawn in, those are left out; the pool is ~3x what is needed.
//! * **The 2,482 images behind `labelv2_train.txt` always go in**, because they carry the clicked
//!   eye pairs that supervise the landmark head.
//!
//! Writes `train_manifest.csv` beside the data: one row per image with its licence, author,
//! author URL and landing page. CC BY 2.0 requires attribution, and DATA_CARD.md section 5 asks
//! for the manifest to travel with the weights. Downloads go into `images/` next to the existing
//! corpus, skip anything already there, and so can be interrupted and rerun. The shuffle
//! reproduces Python's `random.Random(20260919).shuffle`, so the selection matches earlier runs.
const std = @import("std");
const Io = std.Io;
const u = @import("util.zig");

const MIRROR = "https://open-images-dataset.s3.amazonaws.com/train/{s}.jpg";
const SEED = 20260919;
const MIN_WIDTH = 0.05;
const FIELDS = [_][]const u8{ "ImageID", "License", "Author", "AuthorProfileURL", "OriginalLandingURL" };

const StrSet = std.StringHashMapUnmanaged(void);

fn croppableImages(gpa: std.mem.Allocator, io: Io, path: []const u8) !StrSet {
    var keep: StrSet = .empty;
    const csv = try u.Csv.load(gpa, io, path);
    const c_id, const c_group, const c_dep, const c_x0, const c_x1 = .{ csv.col("ImageID"), csv.col("IsGroupOf"), csv.col("IsDepiction"), csv.col("XMin"), csv.col("XMax") };
    for (csv.rows) |r| {
        if (!std.mem.eql(u8, r[c_group], "0") or !std.mem.eql(u8, r[c_dep], "0")) continue;
        if (try std.fmt.parseFloat(f64, r[c_x1]) - try std.fmt.parseFloat(f64, r[c_x0]) >= MIN_WIDTH) try keep.put(gpa, r[c_id], {});
    }
    return keep;
}

fn labelledImages(gpa: std.mem.Allocator, io: Io, path: []const u8) !StrSet {
    var names: StrSet = .empty;
    const text = u.readFile(gpa, io, path) catch return names;
    var it = std.mem.splitScalar(u8, text, '\n');
    while (it.next()) |line| {
        if (!std.mem.startsWith(u8, line, "#")) continue;
        var parts = std.mem.tokenizeAny(u8, line, " \t\r");
        _ = parts.next();
        const file = parts.next() orelse continue;
        try names.put(gpa, file[0 .. std.mem.findScalarLast(u8, file, '.') orelse file.len], {});
    }
    return names;
}

fn csvField(w: *Io.Writer, s: []const u8) !void {
    if (std.mem.indexOfAny(u8, s, ",\"\r\n") == null) return w.writeAll(s);
    try w.writeAll("\"");
    for (s) |c| {
        if (c == '"') try w.writeAll("\"");
        try w.writeAll(&.{c});
    }
    try w.writeAll("\"");
}

fn writeManifest(gpa: std.mem.Allocator, io: Io, path: []const u8, rows: []const []const []const u8) !void {
    var out: Io.Writer.Allocating = .init(gpa);
    const w = &out.writer;
    for (FIELDS, 0..) |f, i| {
        if (i > 0) try w.writeAll(",");
        try w.writeAll(f);
    }
    try w.writeAll("\r\n");
    for (rows) |r| {
        for (r, 0..) |f, i| {
            if (i > 0) try w.writeAll(",");
            try csvField(w, f);
        }
        try w.writeAll("\r\n");
    }
    try u.writeFile(io, path, out.written());
}

const Job = struct {
    io: Io,
    chosen: []const []const u8,
    dest: []const u8,
    next: std.atomic.Value(usize) = .init(0),
    done: std.atomic.Value(usize) = .init(0),
    mutex: Io.Mutex = .init,
    failed: std.ArrayList(Failure) = .empty,
    out: *u.Stdout,
};
const Failure = struct { image: []const u8, err: []const u8 };

fn fetchOne(job: *Job, client: *std.http.Client, image: []const u8) ?[]const u8 {
    const gpa = std.heap.smp_allocator;
    const path = std.fmt.allocPrint(gpa, "{s}/{s}.jpg", .{ job.dest, image }) catch return "OutOfMemory";
    defer gpa.free(path);
    if (Io.Dir.cwd().statFile(job.io, path, .{})) |st| {
        if (st.size > 0) return null;
    } else |_| {}
    const url = std.fmt.allocPrint(gpa, MIRROR, .{image}) catch return "OutOfMemory";
    defer gpa.free(url);
    var body: Io.Writer.Allocating = .init(gpa);
    defer body.deinit();
    const res = client.fetch(.{ .location = .{ .url = url }, .response_writer = &body.writer }) catch |e| return @errorName(e);
    if (res.status != .ok) return "HTTP error";
    const tmp = std.fmt.allocPrint(gpa, "{s}.part", .{path}) catch return "OutOfMemory";
    defer gpa.free(tmp);
    u.writeFile(job.io, tmp, body.written()) catch |e| return @errorName(e);
    // never leave a truncated .jpg that a rerun would skip
    Io.Dir.rename(Io.Dir.cwd(), tmp, Io.Dir.cwd(), path, job.io) catch |e| return @errorName(e);
    return null;
}

fn worker(job: *Job) void {
    var client: std.http.Client = .{ .allocator = std.heap.smp_allocator, .io = job.io };
    defer client.deinit();
    while (true) {
        const i = job.next.fetchAdd(1, .monotonic);
        if (i >= job.chosen.len) return;
        if (fetchOne(job, &client, job.chosen[i])) |err| {
            job.mutex.lockUncancelable(job.io);
            job.failed.append(std.heap.smp_allocator, .{ .image = job.chosen[i], .err = err }) catch {};
            job.mutex.unlock(job.io);
        }
        const done = job.done.fetchAdd(1, .monotonic) + 1;
        if (done % 2000 == 0 or done == job.chosen.len) {
            job.mutex.lockUncancelable(job.io);
            job.out.w().print("{d}/{d} done, {d} failed\n", .{ done, job.chosen.len, job.failed.items.len }) catch {};
            job.out.w().flush() catch {};
            job.mutex.unlock(job.io);
        }
    }
}

pub fn main(init: std.process.Init) !void {
    const gpa = init.arena.allocator();
    const io = init.io;
    const args = try init.minimal.args.toSlice(gpa);
    var data: ?[]const u8 = null;
    var count: usize = 80_000;
    var workers: usize = 32;
    var select_only = false;
    var i: usize = 1;
    while (i < args.len) : (i += 1) {
        if (std.mem.eql(u8, args[i], "--count") and i + 1 < args.len) {
            i += 1;
            count = try std.fmt.parseInt(usize, args[i], 10);
        } else if (std.mem.eql(u8, args[i], "--workers") and i + 1 < args.len) {
            i += 1;
            workers = try std.fmt.parseInt(usize, args[i], 10);
        } else if (std.mem.eql(u8, args[i], "--select-only")) {
            select_only = true;
        } else data = args[i];
    }
    const dir = data orelse return error.Usage;
    var so: u.Stdout = undefined;
    so.init(io);
    const w = so.w();

    const candidates = try croppableImages(gpa, io, try std.fs.path.join(gpa, &.{ dir, "faces-oidv6-train-annotations-bbox.csv" }));
    const required = try labelledImages(gpa, io, try std.fs.path.join(gpa, &.{ dir, "labelv2_train.txt" }));
    try w.print("train images with a croppable face: {d}; required (eye-labelled): {d}\n", .{ candidates.count(), required.count() });
    try w.flush();

    var rows: std.StringArrayHashMapUnmanaged([]const []const u8) = .empty;
    const ic = try u.Csv.load(gpa, io, try std.fs.path.join(gpa, &.{ dir, "train-images-boxable-with-rotation.csv" }));
    const c_rot = ic.col("Rotation");
    var cols: [FIELDS.len]usize = undefined;
    for (FIELDS, 0..) |f, k| cols[k] = ic.col(f);
    for (ic.rows) |r| {
        const image = r[cols[0]];
        if (required.contains(image) or (candidates.contains(image) and std.mem.eql(u8, r[c_rot], "0.0"))) {
            const picked = try gpa.alloc([]const u8, FIELDS.len);
            for (cols, 0..) |c, k| picked[k] = r[c];
            try rows.put(gpa, image, picked);
        }
    }

    var req_it = required.keyIterator();
    var missing: usize = 0;
    while (req_it.next()) |k| missing += @intFromBool(!rows.contains(k.*));
    if (missing > 0) {
        std.debug.print("{d} eye-labelled images are absent from the image table\n", .{missing});
        std.process.exit(1);
    }
    var pool: std.ArrayList([]const u8) = .empty;
    for (rows.keys()) |k| if (!required.contains(k)) try pool.append(gpa, k);
    std.mem.sort([]const u8, pool.items, {}, u.lessStr);
    var rng = u.PyRandom.init(SEED);
    rng.shuffle([]const u8, pool.items);
    var chosen: std.ArrayList([]const u8) = .empty;
    var req_keys: std.ArrayList([]const u8) = .empty;
    req_it = required.keyIterator();
    while (req_it.next()) |k| try req_keys.append(gpa, k.*);
    std.mem.sort([]const u8, req_keys.items, {}, u.lessStr);
    try chosen.appendSlice(gpa, req_keys.items);
    try chosen.appendSlice(gpa, pool.items[0..@min(pool.items.len, count -| req_keys.items.len)]);
    try w.print("eligible pool {d}; chosen {d}\n", .{ pool.items.len + required.count(), chosen.items.len });

    const manifest = try std.fs.path.join(gpa, &.{ dir, "train_manifest.csv" });
    var manifest_rows: std.ArrayList([]const []const u8) = .empty;
    for (chosen.items) |image| try manifest_rows.append(gpa, rows.get(image).?);
    try writeManifest(gpa, io, manifest, manifest_rows.items);
    try w.print("wrote {s}\n", .{manifest});
    try w.flush();
    if (select_only) return;

    const dest = try std.fs.path.join(gpa, &.{ dir, "images" });
    try Io.Dir.cwd().createDirPath(io, dest);
    var job: Job = .{ .io = io, .chosen = chosen.items, .dest = dest, .out = &so };
    const threads = try gpa.alloc(std.Thread, @min(workers, chosen.items.len));
    for (threads) |*t| t.* = try std.Thread.spawn(.{}, worker, .{&job});
    for (threads) |t| t.join();

    // Photos removed from the mirror since 2018 are expected (DATA_CARD.md caveat 2); they are
    // listed rather than fatal, and the manifest is rewritten without them.
    if (job.failed.items.len > 0) {
        var log: Io.Writer.Allocating = .init(gpa);
        var gone: StrSet = .empty;
        for (job.failed.items) |f| {
            try log.writer.print("{s}\t{s}\n", .{ f.image, f.err });
            try gone.put(gpa, f.image, {});
        }
        try u.writeFile(io, try std.fs.path.join(gpa, &.{ dir, "train_fetch_failed.txt" }), log.written());
        var kept: std.ArrayList([]const []const u8) = .empty;
        for (manifest_rows.items) |r| if (!gone.contains(r[0])) try kept.append(gpa, r);
        try writeManifest(gpa, io, manifest, kept.items);
    }
    try w.print("finished: {d} images on disk, {d} unavailable\n", .{ chosen.items.len - job.failed.items.len, job.failed.items.len });
    try w.flush();
}
