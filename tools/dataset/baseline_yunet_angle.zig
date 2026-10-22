//! Score YuNet's own eye points on the test set, as the baseline a new model has to beat.
//!
//!   zig run tools/dataset/baseline_yunet_angle.zig -- <data_dir> [detections.json]
//!
//! Reads face_keypoints_test.json and a detection dump from `fcs-cli --json` (default
//! det_0.8.json), matches them the same way `eval_eye_error.py` does, and reports the same
//! metrics. Without this, the numbers from a trained model are uninterpretable: nobody knows
//! whether 6 degrees of eye-line error is good or bad until the shipping detector is measured on
//! identical faces with identical matching.
//!
//! Landmark order matches ours: index 0 is the subject's right eye, which sits on the viewer's
//! left for an upright face (measured over 10,880 detections -- see the note in
//! `fcs-core/src/face_cropper.rs`). Only faces whose eyes were clicked and whose side is known are
//! scored, so `steep_roll` faces do not turn our labelling ambiguity into model error.
const std = @import("std");
const u = @import("util.zig");
const json = std.json;

const Box = [4]f64;

fn iou(a: Box, b: Box) f64 {
    const ix = @max(0.0, @min(a[2], b[2]) - @max(a[0], b[0]));
    const iy = @max(0.0, @min(a[3], b[3]) - @max(a[1], b[1]));
    const inter = ix * iy;
    const union_ = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter;
    return if (union_ > 0) inter / union_ else 0.0;
}

/// Basename without extension, for a path recorded on Windows and possibly read on Linux.
///
/// `fcs-cli --json` writes `image` as a Windows path, backslashes and all, so split on both.
fn stemOf(path: []const u8) []const u8 {
    const base = path[if (std.mem.findScalarLast(u8, path, '/')) |i| i + 1 else 0..];
    const base2 = base[if (std.mem.findScalarLast(u8, base, '\\')) |i| i + 1 else 0..];
    return base2[0 .. std.mem.findScalarLast(u8, base2, '.') orelse base2.len];
}

pub fn main(init: std.process.Init) !void {
    const gpa = init.arena.allocator();
    const io = init.io;
    const args = try init.minimal.args.toSlice(gpa);
    if (args.len < 2) return error.Usage;
    const data = args[1];
    const dump = if (args.len > 2) args[2] else try std.fs.path.join(gpa, &.{ data, "det_0.8.json" });

    const truth = try u.readJson(gpa, io, try std.fs.path.join(gpa, &.{ data, "face_keypoints_test.json" }));
    var by_image: std.AutoHashMapUnmanaged(i64, std.ArrayList(json.Value)) = .empty;
    for (u.field(truth, "annotations").array.items) |a| {
        const e = try by_image.getOrPut(gpa, u.field(a, "image_id").integer);
        if (!e.found_existing) e.value_ptr.* = .empty;
        try e.value_ptr.append(gpa, a);
    }
    var detected: std.StringHashMapUnmanaged([]json.Value) = .empty;
    for ((try u.readJson(gpa, io, dump)).array.items) |rec| {
        try detected.put(gpa, stemOf(u.str(u.field(rec, "image"))), u.field(rec, "detections").array.items);
    }

    var scored: usize = 0;
    var matched: usize = 0;
    var angles: std.ArrayList(f64) = .empty;
    var relative: std.ArrayList(f64) = .empty;
    for (u.field(truth, "images").array.items) |image| {
        const dets = detected.get(std.fs.path.stem(u.str(u.field(image, "file_name")))) orelse &.{};
        var boxes: std.ArrayList(Box) = .empty;
        for (dets) |d| {
            const b = u.field(d, "bbox").array.items;
            try boxes.append(gpa, .{ u.num(b[0]), u.num(b[1]), u.num(b[0]) + u.num(b[2]), u.num(b[1]) + u.num(b[3]) });
        }
        for ((by_image.get(u.field(image, "id").integer) orelse continue).items) |annotation| {
            if (u.field(annotation, "num_keypoints").integer != 2 or u.truthy(u.field(u.field(annotation, "fcs"), "steep_roll"))) continue;
            scored += 1;
            const bb = u.field(annotation, "bbox").array.items;
            const x, const y, const w, const h = .{ u.num(bb[0]), u.num(bb[1]), u.num(bb[2]), u.num(bb[3]) };
            const box: Box = .{ x, y, x + w, y + h };
            var best: ?usize = null;
            var best_iou: f64 = 0.5;
            for (boxes.items, 0..) |candidate, index| {
                const overlap = iou(box, candidate);
                if (overlap >= best_iou) {
                    best = index;
                    best_iou = overlap;
                }
            }
            const b = best orelse continue;
            matched += 1;
            const k = u.field(annotation, "keypoints").array.items;
            const truth_eyes = [2][2]f64{ .{ u.num(k[0]), u.num(k[1]) }, .{ u.num(k[3]), u.num(k[4]) } };
            const lm = u.field(dets[b], "landmarks").array.items;
            var predicted: [2][2]f64 = undefined;
            for (0..2) |i| predicted[i] = .{ u.num(lm[i].array.items[0]), u.num(lm[i].array.items[1]) };
            const truth_angle = std.math.atan2(truth_eyes[1][1] - truth_eyes[0][1], truth_eyes[1][0] - truth_eyes[0][0]);
            const pred_angle = std.math.atan2(predicted[1][1] - predicted[0][1], predicted[1][0] - predicted[0][0]);
            const difference = @mod(@abs(std.math.radiansToDegrees(truth_angle - pred_angle)), 360.0);
            try angles.append(gpa, @min(difference, 360 - difference));
            for (truth_eyes, predicted) |e, g| try relative.append(gpa, if (w != 0) std.math.hypot(e[0] - g[0], e[1] - g[1]) / w else 0.0);
        }
    }

    var so: u.Stdout = undefined;
    so.init(io);
    const out = so.w();
    try out.print("baseline: {s}\n  scorable faces: {d}\n", .{ std.fs.path.basename(dump), scored });
    if (scored > 0) try out.print("  detected at IoU>=0.5: {d} ({d:.1}%)\n", .{ matched, @as(f64, @floatFromInt(matched)) / @as(f64, @floatFromInt(scored)) * 100 });
    if (angles.items.len > 0) {
        try out.print("  eye-line angle error median: {d:.2} deg\n", .{try u.median(gpa, angles.items)});
        for ([_]f64{ 2, 5, 10 }) |limit| {
            var n: usize = 0;
            for (angles.items) |a| n += @intFromBool(a <= limit);
            try out.print("    within {d} deg: {d:.1}%\n", .{ @as(u32, @intFromFloat(limit)), @as(f64, @floatFromInt(n)) / @as(f64, @floatFromInt(angles.items.len)) * 100 });
        }
        try out.print("  eye distance median: {d:.3} of box width\n", .{try u.median(gpa, relative.items)});
    } else try out.writeAll("  nothing matched, so there is no baseline to report\n");
    try out.flush();
}
