//! Two detectors on a corpus with no labels: count the disagreements, then show them.
//!
//!   zig run -O ReleaseFast tools/dataset/compare_detectors.zig -- <a.json> <b.json> <out.html>
//!       [--label-a YuNet] [--label-b SCRFD] [--count 60] [--min-side 24]
//!
//! Both files are `fcs-cli --json` shaped (YuNet writes it; `scrfd_detect.zig` writes it for a
//! checkpoint). Boxes are matched at IoU >= 0.5, and the counts that come back are: found by both,
//! found only by A, found only by B.
//!
//! Without ground truth, "more detections" is not "better" -- the extra boxes could be faces or
//! furniture. So the numbers are only half of it, and the page is the other half: a crop of every
//! disagreement, biggest first, captioned with which model found it. The question a human answers
//! is the only one that matters here, one crop at a time: is that a face?
//!
//! Comparison is only fair at matched false-positive rates, which this cannot measure and does not
//! try to. Pick the thresholds elsewhere -- on Open Images, YuNet at 0.8 and the 80k SCRFD at 0.4
//! both sit near 0.11-0.14 FP/image (SCRFD_80K.md) -- and pass detections already filtered.
//!
//! Tiles are cut from the local JPEGs with CSS (the same square, clamped and stretched to 160px,
//! the Python version resized with OpenCV), so no image library is needed; open the file locally.
const std = @import("std");
const builtin = @import("builtin");
const u = @import("util.zig");

const Box = [4]f64;
const Hit = struct { name: []const u8, box: Box };
const Images = std.StringArrayHashMapUnmanaged([]const Box);
const TILE = 160.0;

fn iou(a: Box, b: Box) f64 {
    const ix = @max(0.0, @min(a[2], b[2]) - @max(a[0], b[0]));
    const iy = @max(0.0, @min(a[3], b[3]) - @max(a[1], b[1]));
    const inter = ix * iy;
    const union_ = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter;
    return if (union_ > 0) inter / union_ else 0.0;
}

/// Boxes per image name as x1, y1, x2, y2; a later record for the same name replaces an earlier one.
fn load(gpa: std.mem.Allocator, records: []const std.json.Value, paths: *std.StringArrayHashMapUnmanaged([]const u8)) !Images {
    var out: Images = .empty;
    for (records) |record| {
        const image = u.str(u.field(record, "image"));
        const name = u.baseName(image);
        const gop = try paths.getOrPut(gpa, name);
        if (!gop.found_existing) gop.value_ptr.* = image;
        const dets = u.field(record, "detections").array.items;
        const boxes = try gpa.alloc(Box, dets.len);
        for (dets, boxes) |d, *b| {
            const v = u.field(d, "bbox").array.items;
            b.* = .{ u.num(v[0]), u.num(v[1]), u.num(v[0]) + u.num(v[2]), u.num(v[1]) + u.num(v[3]) };
        }
        try out.put(gpa, name, boxes);
    }
    return out;
}

const Tally = struct { both: usize = 0, only_a: usize = 0, only_b: usize = 0, images_with_any: usize = 0 };

/// Greedy matching, A's boxes in order, each taking the first unclaimed B box at IoU >= 0.5.
fn match(gpa: std.mem.Allocator, name: []const u8, a: []const Box, b: []const Box, min_side: f64, tally: *Tally, only_a: *std.ArrayList(Hit), only_b: *std.ArrayList(Hit)) !void {
    var aa: std.ArrayList(Box) = .empty;
    var bb: std.ArrayList(Box) = .empty;
    for (a) |x| if (x[2] - x[0] >= min_side) try aa.append(gpa, x);
    for (b) |x| if (x[2] - x[0] >= min_side) try bb.append(gpa, x);
    if (aa.items.len > 0 or bb.items.len > 0) tally.images_with_any += 1;
    const taken = try gpa.alloc(bool, bb.items.len);
    @memset(taken, false);
    for (aa.items) |box| {
        for (bb.items, 0..) |other, i| {
            if (!taken[i] and iou(box, other) >= 0.5) {
                taken[i] = true;
                tally.both += 1;
                break;
            }
        } else {
            tally.only_a += 1;
            try only_a.append(gpa, .{ .name = name, .box = box });
        }
    }
    for (bb.items, taken) |box, t| if (!t) {
        tally.only_b += 1;
        try only_b.append(gpa, .{ .name = name, .box = box });
    };
}

fn widerFirst(_: void, x: Hit, y: Hit) bool {
    return x.box[2] - x.box[0] > y.box[2] - y.box[0];
}

fn crops(gpa: std.mem.Allocator, io: std.Io, w: *std.Io.Writer, hits: []Hit, paths: std.StringArrayHashMapUnmanaged([]const u8), count: usize) !void {
    std.sort.block(Hit, hits, {}, widerFirst); // stable, like Python's sorted
    var first = true;
    for (hits[0..@min(count, hits.len)]) |h| {
        const path = try u.toLocal(gpa, paths.get(h.name).?);
        // shortcut: JPEG only (the corpora are all JPEG); add a PNG header read if that changes.
        const size = u.jpegSizeOfFile(gpa, io, path) orelse continue;
        const iw: f64 = @floatFromInt(size[0]);
        const ih: f64 = @floatFromInt(size[1]);
        const b = h.box;
        const side = @max(b[2] - b[0], b[3] - b[1]) * 1.6;
        const cx = (b[0] + b[2]) / 2;
        const cy = (b[1] + b[3]) / 2;
        // Python's int() truncates toward zero, then the crop is clamped to the image.
        const x1 = @max(0, @trunc(cx - side / 2));
        const y1 = @max(0, @trunc(cy - side / 2));
        const x2 = @min(iw, @trunc(cx + side / 2));
        const y2 = @min(ih, @trunc(cy + side / 2));
        if (x2 <= x1 or y2 <= y1) continue;
        const kx = TILE / (x2 - x1);
        const ky = TILE / (y2 - y1);
        if (!first) try w.writeAll("\n");
        first = false;
        try w.print("<figure><div class=\"t\" style=\"background-image:url({s});background-size:{d:.2}px {d:.2}px;background-position:{d:.2}px {d:.2}px\"></div>", .{ try u.asUri(gpa, path), iw * kx, ih * ky, -x1 * kx, -y1 * ky });
        try w.print("<figcaption>{s}<br>{d}px</figcaption></figure>", .{ h.name[0..@min(22, h.name.len)], @as(i64, @intFromFloat(@trunc(b[2] - b[0]))) });
    }
}

pub fn main(init: std.process.Init) !void {
    const gpa = init.arena.allocator();
    const io = init.io;
    const args = try init.minimal.args.toSlice(gpa);
    var positional: std.ArrayList([]const u8) = .empty;
    var label_a: []const u8 = "A";
    var label_b: []const u8 = "B";
    var count: usize = 60;
    var min_side: f64 = 24.0;
    var i: usize = 1;
    while (i < args.len) : (i += 1) {
        const arg = args[i];
        if (!std.mem.startsWith(u8, arg, "--")) {
            try positional.append(gpa, arg);
            continue;
        }
        i += 1;
        if (i == args.len) return error.MissingOptionValue;
        const value = args[i];
        if (std.mem.eql(u8, arg, "--label-a")) label_a = value else if (std.mem.eql(u8, arg, "--label-b")) label_b = value else if (std.mem.eql(u8, arg, "--count")) count = try std.fmt.parseInt(usize, value, 10) else if (std.mem.eql(u8, arg, "--min-side")) min_side = try std.fmt.parseFloat(f64, value) else return error.UnknownOption;
    }
    if (positional.items.len != 3) {
        std.debug.print("usage: compare_detectors <a.json> <b.json> <out.html> [--label-a A] [--label-b B] [--count 60] [--min-side 24]\n", .{});
        return error.Usage;
    }
    const out_path = positional.items[2];

    var paths: std.StringArrayHashMapUnmanaged([]const u8) = .empty;
    const a_boxes = try load(gpa, (try u.readJson(gpa, io, positional.items[0])).array.items, &paths);
    const b_boxes = try load(gpa, (try u.readJson(gpa, io, positional.items[1])).array.items, &paths);
    const names = try gpa.dupe([]const u8, paths.keys());
    std.mem.sort([]const u8, names, {}, u.lessStr);

    var tally: Tally = .{};
    var only_a: std.ArrayList(Hit) = .empty;
    var only_b: std.ArrayList(Hit) = .empty;
    for (names) |name| try match(gpa, name, a_boxes.get(name) orelse &.{}, b_boxes.get(name) orelse &.{}, min_side, &tally, &only_a, &only_b);

    var so: u.Stdout = undefined;
    so.init(io);
    const o = so.w();
    try o.print("images with a detection: {d}\n", .{tally.images_with_any});
    for ([_][]const u8{ "both models", try std.fmt.allocPrint(gpa, "only {s}", .{label_a}), try std.fmt.allocPrint(gpa, "only {s}", .{label_b}) }, [_]usize{ tally.both, tally.only_a, tally.only_b }) |label, n| {
        try u.padLeft(o, label, 22);
        try o.print(": {d}\n", .{n});
    }
    try o.flush();

    var out: std.Io.Writer.Allocating = .init(gpa);
    const w = &out.writer;
    try w.writeAll("<!doctype html><meta charset=utf-8><title>Detector disagreements</title>" ++
        "<style>body{font:13px system-ui;margin:24px;background:#111;color:#eee}" ++
        "section{display:flex;flex-wrap:wrap;gap:14px;margin-bottom:34px}figure{margin:0}" ++
        ".t{width:160px;height:160px;border-radius:4px;background-repeat:no-repeat}" ++
        "figcaption{text-align:center;color:#9ad;font-size:11px;margin-top:4px}" ++
        "h1{font-size:18px}h2{font-size:15px;color:#ffd79a}p{color:#aaa;max-width:64em}</style>" ++
        "<h1>What one detector found and the other missed</h1>");
    try w.print("<p>Matched at IoU&nbsp;0.5 over {d} images with any detection: <b>{d}</b> faces found by both, <b>{d}</b> only by {s}, <b>{d}</b> only by {s}. Boxes under ", .{ tally.images_with_any, tally.both, tally.only_a, label_a, tally.only_b, label_b });
    try u.fixed(w, min_side, 0);
    try w.writeAll("px wide are left out. Largest first. The only question is whether each crop is a face.</p>");
    try w.print("<h2>Found only by {s} ({d})</h2><section>", .{ label_a, tally.only_a });
    try crops(gpa, io, w, only_a.items, paths, count);
    try w.print("</section><h2>Found only by {s} ({d})</h2><section>", .{ label_b, tally.only_b });
    try crops(gpa, io, w, only_b.items, paths, count);
    try w.writeAll("</section>");
    try u.writeFile(io, out_path, out.written());
    try o.print("wrote {s}\n", .{out_path});
    try o.flush();
}

test "greedy matching claims each B box once and filters narrow boxes" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const gpa = arena.allocator();
    var tally: Tally = .{};
    var only_a: std.ArrayList(Hit) = .empty;
    var only_b: std.ArrayList(Hit) = .empty;
    // Two A boxes overlap the same B box: only the first matches. The 10px box is ignored.
    try match(gpa, "x.jpg", &.{ .{ 0, 0, 100, 100 }, .{ 5, 5, 105, 105 }, .{ 0, 0, 10, 10 } }, &.{ .{ 2, 2, 100, 100 }, .{ 500, 500, 600, 600 } }, 24, &tally, &only_a, &only_b);
    try std.testing.expectEqual(Tally{ .both = 1, .only_a = 1, .only_b = 1, .images_with_any = 1 }, tally);
    try std.testing.expectEqual(@as(f64, 5), only_a.items[0].box[0]);
    try std.testing.expectEqual(@as(f64, 500), only_b.items[0].box[0]);
    try match(gpa, "y.jpg", &.{.{ 0, 0, 10, 10 }}, &.{}, 24, &tally, &only_a, &only_b);
    try std.testing.expectEqual(@as(usize, 1), tally.images_with_any);
}

test "paths from Windows JSON resolve locally" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const got = try u.toLocal(arena.allocator(), "\\\\?\\C:\\Users\\a b\\x.jpg");
    const want = if (builtin.os.tag == .windows) "C:/Users/a b/x.jpg" else "/mnt/c/Users/a b/x.jpg";
    try std.testing.expectEqualStrings(want, got);
    try std.testing.expectEqualStrings("x.jpg", u.baseName("C:\\Users/a\\x.jpg"));
    try std.testing.expectEqual(@as(f64, 0), iou(.{ 0, 0, 1, 1 }, .{ 2, 2, 3, 3 }));
}
