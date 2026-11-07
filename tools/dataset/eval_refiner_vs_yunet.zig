//! Does the refiner give a better eye line than YuNet's own landmarks, on YuNet's own boxes?
//!
//!   zig run -O ReleaseFast -lc tools/dataset/eval_refiner_vs_yunet.zig tools/dataset/vendor/stb.c -- <data_dir> <eye_refiner.pt> [detections.json] [--ort path]
//!
//! `train_eye_refiner.py` scores itself against ground-truth boxes, which measures the refiner in
//! isolation. This measures what would actually ship: YuNet detects a face, the refiner re-reads
//! the eyes from YuNet's box, and the result is compared against YuNet's own landmarks on exactly
//! the same faces. A paired comparison, so neither side gets an easier set of faces than the
//! other.
//!
//! Faces YuNet does not find are excluded from both columns. They are a recall problem, and the
//! refiner cannot fix a face that was never detected.
//!
//! The checkpoint runs through the graph `export_eye_refiner.zig` builds, on ONNX Runtime
//! (ORT_DYLIB_PATH or `--ort`), so no torch or GPU is needed.
const std = @import("std");
const u = @import("util.zig");
const img = @import("image.zig");
const ort = @import("ort.zig");
const ck = @import("checkpoint.zig");
const net = @import("refiner.zig");
const exporter = @import("export_eye_refiner.zig");

fn iou(a: [4]f64, b: [4]f64) f64 {
    const ix = @max(0.0, @min(a[2], b[2]) - @max(a[0], b[0]));
    const iy = @max(0.0, @min(a[3], b[3]) - @max(a[1], b[1]));
    const inter = ix * iy;
    const union_ = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter;
    return if (union_ > 0) inter / union_ else 0.0;
}

fn lineAngle(first: [2]f64, second: [2]f64) f64 {
    return std.math.atan2(second[1] - first[1], second[0] - first[0]) * (180.0 / std.math.pi);
}

/// `line_angle` on torch's float32 points: the subtraction happens in float32.
fn lineAngle32(p: [2][2]f32) f64 {
    return std.math.atan2(@as(f64, p[1][1] - p[0][1]), @as(f64, p[1][0] - p[0][0])) * (180.0 / std.math.pi);
}

fn difference(a: f64, b: f64) f64 {
    const gap = @mod(@abs(a - b), 360);
    return @min(gap, 360 - gap);
}

fn usable(a: std.json.Value) bool {
    return u.num(u.field(a, "num_keypoints")) == 2 and !u.truthy(u.field(u.field(a, "fcs"), "steep_roll"));
}

/// A checkpoint statistics dict, if present and non-empty (Python's `or` chain).
fn stats(state: ck.Value, key: []const u8) ?*ck.Dict {
    const v = state.dict.get(key) orelse return null;
    return if (v == .dict and v.dict.items.items.len > 0) v.dict else null;
}

fn angleMedian(d: *ck.Dict) !f64 {
    const v = d.get("angle_median") orelse return error.MissingAngleMedian;
    return switch (v) {
        .float => |f| f,
        .integer => |n| @floatFromInt(n),
        else => error.MissingAngleMedian,
    };
}

fn share(values: []const f64, limit: f64) f64 {
    var n: usize = 0;
    for (values) |v| n += @intFromBool(v <= limit);
    return @as(f64, @floatFromInt(n)) / @as(f64, @floatFromInt(values.len));
}

/// `f"{v:>{width}.{n}f}"` followed by `suffix`, which Python writes outside the padding.
fn col(gpa: std.mem.Allocator, w: *std.Io.Writer, v: f64, width: usize, comptime n: u8, suffix: []const u8) !void {
    try u.padLeft(w, try u.fixedAlloc(gpa, v, n), width);
    try w.writeAll(suffix);
}

/// `f"{v:>{width}.1%}"`: here the percent sign is inside the padding.
fn pct(gpa: std.mem.Allocator, w: *std.Io.Writer, v: f64, width: usize) !void {
    try u.padLeft(w, try std.fmt.allocPrint(gpa, "{s}%", .{try u.fixedAlloc(gpa, v * 100, 1)}), width);
}

pub fn main(init: std.process.Init) !void {
    const gpa = init.arena.allocator();
    const io = init.io;
    const args = try init.minimal.args.toSlice(gpa);
    var positional: std.ArrayList([]const u8) = .empty;
    var runtime = ort.runtimePath(init.environ_map);
    var i: usize = 1;
    while (i < args.len) : (i += 1) {
        if (std.mem.eql(u8, args[i], "--ort") and i + 1 < args.len) {
            i += 1;
            runtime = args[i];
        } else try positional.append(gpa, args[i]);
    }
    if (positional.items.len < 2 or positional.items.len > 3) {
        std.debug.print("usage: eval_refiner_vs_yunet <data_dir> <eye_refiner.pt> [detections.json] [--ort path]\n", .{});
        return error.Usage;
    }
    const data = try std.Io.Dir.cwd().realPathFileAlloc(io, positional.items[0], gpa);
    const dump = if (positional.items.len == 3) positional.items[2] else try std.fs.path.join(gpa, &.{ data, "det_0.8.json" });

    const state = try ck.load(gpa, io, positional.items[1]);
    const model = try net.Refiner.fromCheckpoint(gpa, state);
    var oracle = try ort.Oracle.open(gpa, runtime, try exporter.buildOnnx(gpa, model));
    defer oracle.close();
    var out: u.Stdout = undefined;
    out.init(io);
    const w = out.w();
    // Checkpoints from before the validation split carried a single "stats"; later ones carry
    // "val_stats" for the epoch selection and "test_stats" for the one look at the test set.
    try w.print("refiner from {s}", .{std.fs.path.basename(positional.items[1])});
    if (stats(state, "val_stats") orelse stats(state, "stats")) |selected| {
        try w.print(", selected at {s} deg", .{try u.fixedAlloc(gpa, try angleMedian(selected), 2)});
    }
    if (stats(state, "test_stats")) |held_out| {
        try w.print(", {s} deg on the test set, ground-truth boxes", .{try u.fixedAlloc(gpa, try angleMedian(held_out), 2)});
    }
    try w.writeAll("\n");
    try w.flush();

    const document = try u.readJson(gpa, io, try std.fs.path.join(gpa, &.{ data, "face_keypoints_test.json" }));
    var by_image: std.AutoHashMapUnmanaged(i64, std.ArrayList(std.json.Value)) = .empty;
    for (u.field(document, "annotations").array.items) |a| {
        const gop = try by_image.getOrPut(gpa, u.field(a, "image_id").integer);
        if (!gop.found_existing) gop.value_ptr.* = .empty;
        try gop.value_ptr.append(gpa, a);
    }
    var detected: std.StringHashMapUnmanaged([]const std.json.Value) = .empty;
    for ((try u.readJson(gpa, io, dump)).array.items) |record| {
        try detected.put(gpa, u.stemOf(u.str(u.field(record, "image"))), u.field(record, "detections").array.items);
    }

    var refiner_angles: std.ArrayList(f64) = .empty;
    var yunet_angles: std.ArrayList(f64) = .empty;
    var refiner_distance: std.ArrayList(f64) = .empty;
    var yunet_distance: std.ArrayList(f64) = .empty;
    var better: usize = 0;
    for (u.field(document, "images").array.items) |image| {
        const anns = if (by_image.get(u.field(image, "id").integer)) |list| list.items else &.{};
        var faces: std.ArrayList(std.json.Value) = .empty;
        for (anns) |a| if (usable(a)) try faces.append(gpa, a);
        if (faces.items.len == 0) continue;
        const file_name = u.str(u.field(image, "file_name"));
        const dets = detected.get(u.stemOf(file_name)) orelse continue;
        if (dets.len == 0) continue;
        var arena: std.heap.ArenaAllocator = .init(std.heap.smp_allocator);
        defer arena.deinit();
        const tmp = arena.allocator();
        const frame = img.load(tmp, io, try std.fs.path.join(tmp, &.{ data, "images", file_name }), .apply_exif) catch continue;
        const boxes = try tmp.alloc([4]f64, dets.len);
        for (boxes, dets) |*b, d| {
            const v = u.field(d, "bbox").array.items;
            b.* = .{ u.num(v[0]), u.num(v[1]), u.num(v[0]) + u.num(v[2]), u.num(v[1]) + u.num(v[3]) };
        }

        for (faces.items) |annotation| {
            const bbox = u.field(annotation, "bbox").array.items;
            const x, const y, const bw, const bh = .{ u.num(bbox[0]), u.num(bbox[1]), u.num(bbox[2]), u.num(bbox[3]) };
            var best: ?usize = null;
            var best_iou: f64 = 0.5;
            for (boxes, 0..) |candidate, index| {
                const overlap = iou(.{ x, y, x + bw, y + bh }, candidate);
                if (overlap >= best_iou) {
                    best = index;
                    best_iou = overlap;
                }
            }
            const b = best orelse continue;

            const kp = u.field(annotation, "keypoints").array.items;
            const truth = [2][2]f64{ .{ u.num(kp[0]), u.num(kp[1]) }, .{ u.num(kp[3]), u.num(kp[4]) } };
            const truth_angle = lineAngle(truth[0], truth[1]);

            // The refiner re-reads the eyes from YuNet's box, exactly as it would in the app.
            const d = u.field(dets[b], "bbox").array.items;
            const dx, const dy, const dw, const dh = .{ u.num(d[0]), u.num(d[1]), u.num(d[2]), u.num(d[3]) };
            const centre = [2]f32{ @floatCast(dx + dw / 2), @floatCast(dy + dh / 2) };
            const predicted = try net.readEyes(tmp, &oracle, frame, centre, @max(dw, dh) * 1.25);

            const own = u.field(dets[b], "landmarks").array.items;
            const own_eyes = [2][2]f64{ .{ u.num(own[0].array.items[0]), u.num(own[0].array.items[1]) }, .{ u.num(own[1].array.items[0]), u.num(own[1].array.items[1]) } };
            const r = difference(truth_angle, lineAngle32(predicted));
            const yn = difference(truth_angle, lineAngle(own_eyes[0], own_eyes[1]));
            try refiner_angles.append(gpa, r);
            try yunet_angles.append(gpa, yn);
            if (r < yn) better += 1;
            for (truth, predicted, own_eyes) |expected, got_refiner, got_yunet| {
                const pr = [2]f64{ got_refiner[0], got_refiner[1] };
                try refiner_distance.append(gpa, if (bw != 0) std.math.hypot(expected[0] - pr[0], expected[1] - pr[1]) / bw else 0);
                try yunet_distance.append(gpa, if (bw != 0) std.math.hypot(expected[0] - got_yunet[0], expected[1] - got_yunet[1]) / bw else 0);
            }
        }
    }

    const total = refiner_angles.items.len;
    if (total == 0) {
        try w.writeAll("nothing matched, so there is nothing to compare\n");
        return w.flush();
    }
    try w.print("\npaired on {d} faces that YuNet detected at IoU>=0.5\n\n", .{total});
    try u.padRight(w, "", 22);
    try w.writeAll(" ");
    try u.padLeft(w, "refiner", 10);
    try w.writeAll(" ");
    try u.padLeft(w, "YuNet", 10);
    try w.writeAll("\n");
    try u.padRight(w, "angle error, median", 22);
    try w.writeAll(" ");
    try col(gpa, w, try u.median(gpa, refiner_angles.items), 9, 2, "d");
    try w.writeAll(" ");
    try col(gpa, w, try u.median(gpa, yunet_angles.items), 9, 2, "d");
    try w.writeAll("\n");
    for ([_]f64{ 2, 5, 10 }) |limit| {
        try u.padRight(w, try std.fmt.allocPrint(gpa, "  within {d} deg", .{@as(u32, @intFromFloat(limit))}), 22);
        try w.writeAll(" ");
        try pct(gpa, w, share(refiner_angles.items, limit), 9);
        try w.writeAll(" ");
        try pct(gpa, w, share(yunet_angles.items, limit), 9);
        try w.writeAll("\n");
    }
    try u.padRight(w, "eye distance, median", 22);
    try w.writeAll(" ");
    try col(gpa, w, try u.median(gpa, refiner_distance.items), 9, 3, "w");
    try w.writeAll(" ");
    try col(gpa, w, try u.median(gpa, yunet_distance.items), 9, 3, "w");
    try w.writeAll("\n");
    try w.print("\nrefiner closer on {d} of {d} faces ({s}%)\n", .{ better, total, try u.fixedAlloc(gpa, @as(f64, @floatFromInt(better)) / @as(f64, @floatFromInt(total)) * 100, 1) });
    try w.flush();
}

test "angle differences wrap around a full turn" {
    try std.testing.expectEqual(@as(f64, 2), difference(179, -179));
    try std.testing.expectEqual(@as(f64, 10), difference(5, -5));
    try std.testing.expectApproxEqAbs(@as(f64, 45), lineAngle(.{ 0, 0 }, .{ 1, 1 }), 1e-12);
}
