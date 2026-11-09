//! Run a trained SCRFD checkpoint over a folder, writing fcs-cli's detection JSON format.
//!
//!   zig run -O ReleaseFast -lc tools/dataset/scrfd_detect.zig tools/dataset/vendor/stb.c -- <checkpoint.pth> <image_dir> <out.json> [--thresh 0.4] [--list names.txt] [--ort path]
//!
//! `--list` limits the run to the named files of `image_dir`. Each record also carries the decoded
//! image's `width` and `height` (EXIF orientation applied), which fcs-cli's own records lack.
//!
//! So a checkpoint can be compared against YuNet on a corpus neither model was trained on, using
//! the same tooling: `fcs-cli --json` writes this shape, and `compare_detectors.zig` reads it.
//!
//! `Detector` is shared with `eval_eye_error.zig`. It builds the checkpoint's graph in memory with
//! `export_scrfd.zig` and runs it on ONNX Runtime (ORT_DYLIB_PATH or `--ort`), then decodes the
//! raw per-stride tensors the way SCRFD's own wrapper does: letterbox into the top-left of a
//! 640x640 canvas with the aspect kept, distances times the stride from each anchor centre,
//! predictions divided by the letterbox scale, then the wrapper's NMS.
const std = @import("std");
const u = @import("util.zig");
const img = @import("image.zig");
const ort = @import("ort.zig");
const ck = @import("checkpoint.zig");
const net = @import("scrfd.zig");
const exporter = @import("export_scrfd.zig");

/// A detection as SCRFD's wrapper returns it: box corners, score, five keypoints.
pub const Row = [15]f32;

/// The SCRFD wrapper's own NMS, kept identical so scores are comparable with upstream. Returns
/// the kept rows' indices, best score first.
pub fn nms(gpa: std.mem.Allocator, rows: []const Row) ![]usize {
    var order: std.ArrayList(usize) = .empty;
    defer order.deinit(gpa);
    for (0..rows.len) |i| try order.append(gpa, rows.len - 1 - i);
    std.sort.block(usize, order.items, rows, struct {
        fn higher(r: []const Row, a: usize, b: usize) bool {
            return r[a][4] > r[b][4];
        }
    }.higher);
    var keep: std.ArrayList(usize) = .empty;
    var next: std.ArrayList(usize) = .empty;
    defer next.deinit(gpa);
    while (order.items.len > 0) {
        const i = order.items[0];
        try keep.append(gpa, i);
        const a = rows[i];
        const area_i = (a[2] - a[0] + 1) * (a[3] - a[1] + 1);
        next.clearRetainingCapacity();
        for (order.items[1..]) |j| {
            const b = rows[j];
            const w = @max(0, @min(a[2], b[2]) - @max(a[0], b[0]) + 1);
            const h = @max(0, @min(a[3], b[3]) - @max(a[1], b[1]) + 1);
            const area_j = (b[2] - b[0] + 1) * (b[3] - b[1] + 1);
            if ((w * h) / (area_i + area_j - w * h) <= @as(f32, 0.4)) try next.append(gpa, j);
        }
        std.mem.swap(std.ArrayList(usize), &order, &next);
    }
    return keep.toOwnedSlice(gpa);
}

pub const Detector = struct {
    oracle: ort.Oracle,
    size: usize,

    pub fn open(gpa: std.mem.Allocator, io: std.Io, runtime: []const u8, checkpoint: []const u8, size: usize) !Detector {
        const model = try net.Scrfd.fromCheckpoint(gpa, try ck.load(gpa, io, checkpoint));
        return .{ .oracle = try ort.Oracle.open(gpa, runtime, try exporter.buildOnnx(gpa, model, size)), .size = size };
    }

    pub fn close(self: *Detector) void {
        self.oracle.close();
    }

    /// Detections scoring at least `thresh`, after NMS, best first, in image coordinates.
    pub fn detect(self: *Detector, gpa: std.mem.Allocator, image: img.Image, thresh: f64) ![]Row {
        const size = self.size;
        // Letterbox into the top-left of a square canvas, as SCRFD.detect does.
        const ratio = @as(f64, @floatFromInt(image.height)) / @as(f64, @floatFromInt(image.width));
        const fsize: f64 = @floatFromInt(size);
        const new_w: usize = if (ratio > 1) @intFromFloat(@trunc(fsize / ratio)) else size;
        const new_h: usize = if (ratio > 1) size else @intFromFloat(@trunc(fsize * ratio));
        const scale: f32 = @floatCast(@as(f64, @floatFromInt(new_h)) / @as(f64, @floatFromInt(image.height)));
        const resized = try img.resizeLinear(gpa, image, new_w, new_h);
        defer resized.deinit(gpa);
        const plane = size * size;
        const input = try gpa.alloc(f32, 3 * plane);
        defer gpa.free(input);
        @memset(input, (0 - 127.5) / 128.0);
        for (0..new_h) |y| for (0..new_w) |x| for (0..3) |c| {
            input[c * plane + y * size + x] = (@as(f32, @floatFromInt(resized.pixels[(y * new_w + x) * 3 + c])) - 127.5) / 128.0;
        };

        var names: [9][:0]const u8 = undefined;
        for (&names, 0..) |*n, i| n.* = try std.fmt.allocPrintSentinel(gpa, "out_{d}", .{i}, 0);
        const outputs = try self.oracle.runAll(gpa, input, &.{ 1, 3, @intCast(size), @intCast(size) }, "input.1", &names);
        const floor: f32 = @floatCast(thresh);
        var rows: std.ArrayList(Row) = .empty;
        defer rows.deinit(gpa);
        for (net.strides, 0..) |stride, level| {
            const side = size / stride;
            const scores, const boxes, const points = .{ outputs[level].data, outputs[3 + level].data, outputs[6 + level].data };
            if (scores.len != side * side * net.anchors or boxes.len != scores.len * 4 or points.len != scores.len * 10) return error.UnexpectedOutputs;
            const s: f32 = @floatFromInt(stride);
            for (scores, 0..) |score, j| {
                if (!(score >= floor)) continue;
                const cell = j / net.anchors;
                const cx: f32 = @floatFromInt((cell % side) * stride);
                const cy: f32 = @floatFromInt((cell / side) * stride);
                const b = boxes[j * 4 ..][0..4];
                var row: Row = undefined;
                row[0] = (cx - b[0] * s) / scale;
                row[1] = (cy - b[1] * s) / scale;
                row[2] = (cx + b[2] * s) / scale;
                row[3] = (cy + b[3] * s) / scale;
                row[4] = score;
                for (0..5) |k| {
                    row[5 + 2 * k] = (cx + points[j * 10 + 2 * k] * s) / scale;
                    row[6 + 2 * k] = (cy + points[j * 10 + 2 * k + 1] * s) / scale;
                }
                try rows.append(gpa, row);
            }
        }
        const keep = try nms(gpa, rows.items);
        defer gpa.free(keep);
        const out = try gpa.alloc(Row, keep.len);
        for (out, keep) |*o, k| o.* = rows.items[k];
        return out;
    }
};

// shortcut: stb reads JPEG, PNG and BMP; WebP and TIFF count as unreadable. Add a decoder if a corpus needs them.
const SUFFIXES = [_][]const u8{ ".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff" };

pub fn main(init: std.process.Init) !void {
    const gpa = init.arena.allocator();
    const io = init.io;
    const args = try init.minimal.args.toSlice(gpa);
    var positional: std.ArrayList([]const u8) = .empty;
    var thresh: f64 = 0.4;
    var runtime = ort.runtimePath(init.environ_map);
    var list: ?[]const u8 = null;
    var i: usize = 1;
    while (i < args.len) : (i += 1) {
        const arg = args[i];
        if (std.mem.startsWith(u8, arg, "--") and i + 1 < args.len) {
            i += 1;
            if (std.mem.eql(u8, arg, "--thresh")) thresh = try std.fmt.parseFloat(f64, args[i]) else if (std.mem.eql(u8, arg, "--ort")) runtime = args[i] else if (std.mem.eql(u8, arg, "--list")) list = args[i] else return error.UnknownOption;
        } else try positional.append(gpa, arg);
    }
    if (positional.items.len != 3) {
        std.debug.print("usage: scrfd_detect <checkpoint.pth> <image_dir> <out.json> [--thresh 0.4] [--list names.txt] [--ort path]\n", .{});
        return error.Usage;
    }
    const dir_path = positional.items[1];
    var detector = try Detector.open(gpa, io, runtime, positional.items[0], 640);
    defer detector.close();

    var paths: std.ArrayList([]const u8) = .empty;
    if (list) |names| {
        // Only these files of image_dir, one name per line: a shared folder needn't be split up.
        var lines = std.mem.splitScalar(u8, try u.readFile(gpa, io, names), '\n');
        while (lines.next()) |raw| {
            const name = std.mem.trim(u8, raw, " \t\r");
            if (name.len > 0) try paths.append(gpa, try std.fs.path.join(gpa, &.{ dir_path, name }));
        }
    } else {
        var dir = try std.Io.Dir.cwd().openDir(io, dir_path, .{ .iterate = true });
        defer dir.close(io);
        var it = dir.iterate();
        while (try it.next(io)) |e| {
            if (e.kind != .file) continue;
            const ext = std.fs.path.extension(e.name);
            const lower = try std.ascii.allocLowerString(gpa, ext);
            for (SUFFIXES) |s| if (std.mem.eql(u8, s, lower)) {
                try paths.append(gpa, try std.fs.path.join(gpa, &.{ dir_path, e.name }));
                break;
            };
        }
    }
    std.mem.sort([]const u8, paths.items, {}, u.lessStr);

    var out: u.Stdout = undefined;
    out.init(io);
    var json: std.Io.Writer.Allocating = .init(gpa);
    const w = &json.writer;
    try w.writeAll("[");
    var records: usize = 0;
    var faces: usize = 0;
    var unreadable: usize = 0;
    for (paths.items, 1..) |path, index| {
        var arena: std.heap.ArenaAllocator = .init(std.heap.smp_allocator);
        defer arena.deinit();
        const tmp = arena.allocator();
        const frame = img.load(tmp, io, path, .apply_exif) catch {
            unreadable += 1;
            continue;
        };
        const dets = try detector.detect(tmp, frame, thresh);
        if (records > 0) try w.writeAll(", ");
        try w.writeAll("{\"image\": ");
        try u.dumpString(w, path);
        try w.print(", \"width\": {d}, \"height\": {d}, \"detections\": [", .{ frame.width, frame.height });
        for (dets, 0..) |d, k| {
            if (k > 0) try w.writeAll(", ");
            const x1: f64 = d[0];
            const y1: f64 = d[1];
            try w.writeAll("{\"score\": ");
            try u.pyFloat(w, d[4]);
            try w.writeAll(", \"bbox\": [");
            for ([_]f64{ x1, y1, @as(f64, d[2]) - x1, @as(f64, d[3]) - y1 }, 0..) |v, n| {
                if (n > 0) try w.writeAll(", ");
                try u.pyFloat(w, v);
            }
            try w.writeAll("], \"landmarks\": [");
            for (0..5) |p| {
                if (p > 0) try w.writeAll(", ");
                try w.writeAll("[");
                try u.pyFloat(w, d[5 + 2 * p]);
                try w.writeAll(", ");
                try u.pyFloat(w, d[6 + 2 * p]);
                try w.writeAll("]");
            }
            try w.writeAll("]}");
        }
        try w.writeAll("]}");
        records += 1;
        faces += dets.len;
        if (index % 200 == 0 or index == paths.items.len) {
            try out.w().print("{d}/{d} images, {d} faces\n", .{ index, paths.items.len, faces });
            try out.w().flush();
        }
    }
    try w.writeAll("]");
    try u.writeFile(io, positional.items[2], json.written());
    try out.w().print("wrote {s}: {d} images, {d} faces, {d} unreadable\n", .{ positional.items[2], records, faces, unreadable });
    try out.w().flush();
}

test "NMS keeps the best box and drops heavy overlaps only" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    var rows: [3]Row = @splat(@splat(0));
    rows[0][0..5].* = .{ 0, 0, 10, 10, 0.5 };
    rows[1][0..5].* = .{ 1, 1, 11, 11, 0.9 }; // overlaps 0 heavily, higher score
    rows[2][0..5].* = .{ 50, 50, 60, 60, 0.7 };
    try std.testing.expectEqualSlices(usize, &.{ 1, 2 }, try nms(arena.allocator(), &rows));
}
