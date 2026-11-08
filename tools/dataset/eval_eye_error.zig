//! Measure a trained SCRFD model's eye-point error on the held-out test set.
//!
//!   zig run -O ReleaseFast -lc tools/dataset/eval_eye_error.zig tools/dataset/vendor/stb.c -- <checkpoint.pth> <data_dir> [--thresh 0.5] [--limit N] [--all-boxes labelv2.txt] [--ort path]
//!
//! Sweeps the score threshold and reports, at each one, detection recall, false positives per
//! image and eye-line angle error against face_keypoints_test.json, so models trained on
//! different numbers of eye labels can be compared at a sane operating point.
//!
//! The sweep exists because a single low threshold is misleading. At 0.02 -- the value SCRFD's own
//! evaluation uses -- the 400-label model reached 96.7% recall while emitting **39.4 boxes per
//! image that match no annotation at all**. Recall measured there says almost nothing, and it
//! flatters the keypoints too: matching picks the best-IoU box out of forty candidates per face,
//! so the eye points are read off whichever proposal happens to fit best.
//!
//! The checkpoint runs through `scrfd_detect.zig`'s `Detector`: the graph `export_scrfd.zig`
//! builds, on ONNX Runtime, decoded as SCRFD's own wrapper does.
//!
//! Error is normalised by ground-truth box width rather than interocular distance: the usual
//! 5-point NME divides by eye separation, which collapses towards zero on the steeply rolled
//! faces this corpus contains and inflates their error arbitrarily. Box width is stable across
//! roll and is the scale everything else here is reported in. Raw pixels are printed too, since
//! a normalised figure hides whether an error matters for cropping.
//!
//! Only faces whose eyes were clicked *and* whose side is known are scored: `steep_roll` faces
//! carry points but no reliable left/right identity, so scoring them would measure our labelling
//! ambiguity rather than the model.
const std = @import("std");
const json = std.json;
const u = @import("util.zig");
const img = @import("image.zig");
const ort = @import("ort.zig");
const sd = @import("scrfd_detect.zig");

const Box = [4]f64;
const THRESHOLDS = [_]f64{ 0.02, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7 };

fn iou(a: Box, b: Box) f64 {
    const ix = @max(0.0, @min(a[2], b[2]) - @max(a[0], b[0]));
    const iy = @max(0.0, @min(a[3], b[3]) - @max(a[1], b[1]));
    const inter = ix * iy;
    const union_ = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter;
    return if (union_ > 0) inter / union_ else 0.0;
}

const Tally = struct {
    scored: usize = 0,
    matched: usize = 0,
    spurious: usize = 0,
    faces: usize = 0,
    found: usize = 0,
    rel: std.ArrayList(f64) = .empty,
    deg: std.ArrayList(f64) = .empty,
};

const Boxes = struct { every: std.ArrayList(Box) = .empty, real: std.ArrayList(Box) = .empty };

fn xywh(v: json.Value) Box {
    const b = v.array.items;
    return .{ u.num(b[0]), u.num(b[1]), u.num(b[0]) + u.num(b[2]), u.num(b[1]) + u.num(b[3]) };
}

fn byFileName(_: void, a: json.Value, b: json.Value) bool {
    return std.mem.lessThan(u8, u.str(u.field(a, "file_name")), u.str(u.field(b, "file_name")));
}

/// `f"{v:>{width}.{n}f}"`, with Python's `nan`.
fn col(gpa: std.mem.Allocator, w: *std.Io.Writer, v: f64, width: usize, comptime n: u8) !void {
    try u.padLeft(w, if (std.math.isNan(v)) "nan" else try u.fixedAlloc(gpa, v, n), width);
}

fn pct(gpa: std.mem.Allocator, w: *std.Io.Writer, v: f64, width: usize) !void {
    try u.padLeft(w, try std.fmt.allocPrint(gpa, "{s}%", .{try u.fixedAlloc(gpa, v * 100, 1)}), width);
}

pub fn main(init: std.process.Init) !void {
    const gpa = init.arena.allocator();
    const io = init.io;
    const args = try init.minimal.args.toSlice(gpa);
    var positional: std.ArrayList([]const u8) = .empty;
    var thresh: f64 = 0.5;
    var limit: usize = 0;
    var all_boxes_path: ?[]const u8 = null;
    var runtime = ort.runtimePath(init.environ_map);
    var i: usize = 1;
    while (i < args.len) : (i += 1) {
        const arg = args[i];
        if (std.mem.startsWith(u8, arg, "--") and i + 1 < args.len) {
            i += 1;
            const v = args[i];
            if (std.mem.eql(u8, arg, "--thresh")) thresh = try std.fmt.parseFloat(f64, v) else if (std.mem.eql(u8, arg, "--limit")) limit = try std.fmt.parseInt(usize, v, 10) else if (std.mem.eql(u8, arg, "--all-boxes")) all_boxes_path = v else if (std.mem.eql(u8, arg, "--ort")) runtime = v else return error.UnknownOption;
        } else try positional.append(gpa, arg);
    }
    if (positional.items.len != 2) {
        std.debug.print("usage: eval_eye_error <checkpoint.pth> <data_dir> [--thresh 0.5] [--limit N] [--all-boxes labelv2.txt] [--ort path]\n", .{});
        return error.Usage;
    }
    const data = try std.Io.Dir.cwd().realPathFileAlloc(io, positional.items[1], gpa);
    const doc = try u.readJson(gpa, io, try std.fs.path.join(gpa, &.{ data, "face_keypoints_test.json" }));
    var by_image: std.AutoHashMapUnmanaged(i64, std.ArrayList(json.Value)) = .empty;
    for (u.field(doc, "annotations").array.items) |a| {
        const gop = try by_image.getOrPut(gpa, u.field(a, "image_id").integer);
        if (!gop.found_existing) gop.value_ptr.* = .empty;
        try gop.value_ptr.append(gpa, a);
    }

    // file name -> (every face and ignore region, real faces only). Ignore regions are group-of
    // boxes and depictions: a detection there is not wrong, and not a face to count as found.
    var all_boxes: std.StringHashMapUnmanaged(Boxes) = .empty;
    if (all_boxes_path) |path| {
        var current: ?*Boxes = null;
        var it = std.mem.splitScalar(u8, try u.readFile(gpa, io, path), '\n');
        while (it.next()) |raw| {
            const line = std.mem.trimEnd(u8, raw, "\r");
            var words = std.mem.tokenizeAny(u8, line, " \t");
            if (std.mem.startsWith(u8, line, "#")) {
                _ = words.next();
                const gop = try all_boxes.getOrPut(gpa, words.next() orelse return error.BadLabels);
                gop.value_ptr.* = .{};
                current = gop.value_ptr;
            } else if (std.mem.trim(u8, line, " \t").len > 0) {
                var values: std.ArrayList(f64) = .empty;
                while (words.next()) |v| try values.append(gpa, try std.fmt.parseFloat(f64, v));
                if (values.items.len < 4) return error.BadLabels;
                const entry = current orelse return error.BadLabels;
                try entry.every.append(gpa, values.items[0..4].*);
                if (values.items.len > 5) try entry.real.append(gpa, values.items[0..4].*);
            }
        }
    }

    var detector = try sd.Detector.open(gpa, io, runtime, positional.items[0], 640);
    defer detector.close();

    var thresholds: std.ArrayList(f64) = .empty;
    for (THRESHOLDS) |t| if (t >= thresh) try thresholds.append(gpa, t);
    if (thresholds.items.len == 0) return error.ThresholdAboveSweep;
    const stats = try gpa.alloc(Tally, thresholds.items.len);
    @memset(stats, .{});
    var images_seen: usize = 0;
    const order = try gpa.dupe(json.Value, u.field(doc, "images").array.items);
    std.mem.sort(json.Value, order, {}, byFileName);
    const shown = if (limit > 0) order[0..@min(limit, order.len)] else order;

    for (shown) |image| {
        const annotations = if (by_image.get(u.field(image, "id").integer)) |list| list.items else &.{};
        var faces: std.ArrayList(json.Value) = .empty;
        for (annotations) |a| {
            if (u.num(u.field(a, "num_keypoints")) == 2 and !u.truthy(u.field(u.field(a, "fcs"), "steep_roll"))) try faces.append(gpa, a);
        }
        if (faces.items.len == 0) continue;
        var arena: std.heap.ArenaAllocator = .init(std.heap.smp_allocator);
        defer arena.deinit();
        const tmp = arena.allocator();
        const file_name = u.str(u.field(image, "file_name"));
        const frame = img.load(tmp, io, try std.fs.path.join(tmp, &.{ data, "images", file_name }), .apply_exif) catch continue;
        images_seen += 1;
        // One forward pass per image at the lowest threshold; every higher threshold is a
        // filter on the scores that come back, so the sweep costs one pass, not eight.
        const dets = try detector.detect(tmp, frame, thresholds.items[0]);

        // Spurious is judged against *every* box in the file, not just the scorable ones: a
        // detection landing on a face whose eyes were skipped, or whose side is unknown, is
        // right rather than spurious. It remains an upper bound, because Open Images boxes are
        // not exhaustive (DATA_CARD.md section 1).
        var every: []const Box = blk: {
            const list = try tmp.alloc(Box, annotations.len);
            for (list, annotations) |*b, a| b.* = xywh(u.field(a, "bbox"));
            break :blk list;
        };
        var real: []const Box = &.{};
        if (all_boxes.get(file_name)) |entry| {
            every = entry.every.items;
            real = entry.real.items;
        }

        for (thresholds.items, stats) |threshold, *tally| {
            var keep: std.ArrayList(usize) = .empty;
            for (dets, 0..) |d, k| if (@as(f64, d[4]) >= threshold) try keep.append(tmp, k);
            for (keep.items) |k| {
                const box = Box{ dets[k][0], dets[k][1], dets[k][2], dets[k][3] };
                const hit = for (every) |b| {
                    if (iou(box, b) >= 0.5) break true;
                } else false;
                if (!hit) tally.spurious += 1;
            }
            for (real) |truth| {
                tally.faces += 1;
                const hit = for (keep.items) |k| {
                    if (iou(truth, .{ dets[k][0], dets[k][1], dets[k][2], dets[k][3] }) >= 0.5) break true;
                } else false;
                if (hit) tally.found += 1;
            }
            for (faces.items) |annotation| {
                tally.scored += 1;
                const bbox = u.field(annotation, "bbox").array.items;
                const truth = xywh(u.field(annotation, "bbox"));
                const w = u.num(bbox[2]);
                var best: ?usize = null;
                var best_iou: f64 = 0.5;
                for (keep.items) |k| {
                    const overlap = iou(truth, .{ dets[k][0], dets[k][1], dets[k][2], dets[k][3] });
                    if (overlap >= best_iou) {
                        best = k;
                        best_iou = overlap;
                    }
                }
                const b = best orelse continue;
                tally.matched += 1;
                const kp = u.field(annotation, "keypoints").array.items;
                var truth_eyes: [2][2]f64 = undefined;
                var pred_eyes: [2][2]f64 = undefined;
                for (0..2) |slot| { // right_eye, left_eye, in the subject's own frame
                    truth_eyes[slot] = .{ u.num(kp[slot * 3]), u.num(kp[slot * 3 + 1]) };
                    pred_eyes[slot] = .{ dets[b][5 + 2 * slot], dets[b][6 + 2 * slot] };
                    const distance = std.math.hypot(truth_eyes[slot][0] - pred_eyes[slot][0], truth_eyes[slot][1] - pred_eyes[slot][1]);
                    try tally.rel.append(gpa, if (w != 0) distance / w else 0);
                }
                // The tilt of the eye line, which is what face_cropper's alignment consumes.
                const truth_angle = std.math.atan2(truth_eyes[1][1] - truth_eyes[0][1], truth_eyes[1][0] - truth_eyes[0][0]);
                const pred_angle = std.math.atan2(pred_eyes[1][1] - pred_eyes[0][1], pred_eyes[1][0] - pred_eyes[0][0]);
                const difference = @mod(@abs((truth_angle - pred_angle) * (180.0 / std.math.pi)), 360);
                try tally.deg.append(gpa, @min(difference, 360 - difference));
            }
        }
    }

    var out: u.Stdout = undefined;
    out.init(io);
    const w = out.w();
    try w.print("checkpoint: {s}\n", .{std.fs.path.basename(positional.items[0])});
    try w.print("scorable faces: {d} (eyes clicked, side known) over {d} images\n", .{ stats[0].scored, images_seen });
    for ([_][]const u8{ "thresh", "recall", "all-face", "FP/image", "angle med", "<=5 deg", "eye err" }, [_]usize{ 7, 8, 9, 9, 10, 8, 9 }, 0..) |h, width, k| {
        if (k > 0) try w.writeAll(" ");
        try u.padLeft(w, h, width);
    }
    try w.writeAll("\n");
    for (thresholds.items, stats) |threshold, tally| {
        const recall = if (tally.scored > 0) @as(f64, @floatFromInt(tally.matched)) / @as(f64, @floatFromInt(tally.scored)) else 0;
        const per_image = if (images_seen > 0) @as(f64, @floatFromInt(tally.spurious)) / @as(f64, @floatFromInt(images_seen)) else 0;
        const angle = if (tally.deg.items.len > 0) try u.median(gpa, tally.deg.items) else std.math.nan(f64);
        var within: usize = 0;
        for (tally.deg.items) |a| within += @intFromBool(a <= 5);
        const share = if (tally.deg.items.len > 0) @as(f64, @floatFromInt(within)) / @as(f64, @floatFromInt(tally.deg.items.len)) else 0;
        const relative = if (tally.rel.items.len > 0) try u.median(gpa, tally.rel.items) else std.math.nan(f64);
        try col(gpa, w, threshold, 7, 2);
        try w.writeAll(" ");
        try pct(gpa, w, recall, 7);
        try w.writeAll(" ");
        if (tally.faces > 0) try pct(gpa, w, @as(f64, @floatFromInt(tally.found)) / @as(f64, @floatFromInt(tally.faces)), 8) else try u.padLeft(w, "-", 8);
        try w.writeAll(" ");
        try col(gpa, w, per_image, 9, 2);
        try w.writeAll(" ");
        try col(gpa, w, angle, 9, 2);
        try w.writeAll("d ");
        try pct(gpa, w, share, 7);
        try w.writeAll(" ");
        try col(gpa, w, relative, 8, 3);
        try w.writeAll("w\n");
    }
    try w.writeAll(
        \\  all-face is recall over every real Open Images face (needs --all-boxes).
        \\  recall is of scorable faces at IoU>=0.5; FP/image counts detections matching no
        \\  box at all; angle is eye-line error in degrees; eye err is median distance as a
        \\  fraction of box width. Compare models at one threshold, not at their best each.
        \\
    );
    try w.flush();
}

test "IoU of disjoint, nested and identical boxes" {
    try std.testing.expectEqual(@as(f64, 0), iou(.{ 0, 0, 1, 1 }, .{ 2, 2, 3, 3 }));
    try std.testing.expectEqual(@as(f64, 0.25), iou(.{ 0, 0, 2, 2 }, .{ 0, 0, 1, 1 }));
    try std.testing.expectEqual(@as(f64, 1), iou(.{ 0, 0, 2, 2 }, .{ 0, 0, 2, 2 }));
}
