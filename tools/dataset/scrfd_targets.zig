//! SCRFD's training targets as MMDetection computes them for one augmented 640x640 image:
//! the anchors, ATSS assignment (`ATSSAssigner`, mode 0) and the per-anchor targets
//! `SCRFDHead._get_target_single` builds from it. Float32 throughout, as torch runs it.
//!
//! Anchors: strides 8/16/32, two per location (base sizes 16/64/256 at scales 1 and 2, ratio
//! 1), centred on (x * stride, y * stride), indexed (level, y, x, anchor) -- the order the head's
//! outputs flatten in.
const std = @import("std");
const Allocator = std.mem.Allocator;

pub const size = 640;
pub const strides = [3]usize{ 8, 16, 32 };
pub const per_location = 2;
pub const level_counts = blk: {
    var counts: [3]usize = undefined;
    for (strides, &counts) |s, *c| c.* = (size / s) * (size / s) * per_location;
    break :blk counts;
};
pub const anchor_count = level_counts[0] + level_counts[1] + level_counts[2];
const base_sizes = [3]f32{ 16, 64, 256 };
const scales = [per_location]f32{ 1, 2 };

pub const Box = [4]f32; // x1, y1, x2, y2

pub fn anchors() [anchor_count]Box {
    @setEvalBranchQuota(200000);
    var out: [anchor_count]Box = undefined;
    var i: usize = 0;
    for (strides, base_sizes) |s, base| {
        const side = size / s;
        for (0..side) |y| for (0..side) |x| for (scales) |scale| {
            const cx: f32 = @floatFromInt(x * s);
            const cy: f32 = @floatFromInt(y * s);
            const half = base * scale / 2;
            out[i] = .{ cx - half, cy - half, cx + half, cy + half };
            i += 1;
        };
    }
    return out;
}

/// The stride of anchor `i`.
pub fn strideOf(i: usize) usize {
    if (i < level_counts[0]) return strides[0];
    if (i < level_counts[0] + level_counts[1]) return strides[1];
    return strides[2];
}

pub fn centre(b: Box) [2]f32 {
    return .{ (b[0] + b[2]) / 2, (b[1] + b[3]) / 2 };
}

/// `bbox_overlaps` (eps 1e-6): IoU, or IoF (overlap over the first box's area).
fn overlap(a: Box, b: Box, iof: bool) f32 {
    const area_a = (a[2] - a[0]) * (a[3] - a[1]);
    const area_b = (b[2] - b[0]) * (b[3] - b[1]);
    const w = @max(@min(a[2], b[2]) - @max(a[0], b[0]), 0);
    const h = @max(@min(a[3], b[3]) - @max(a[1], b[1]), 0);
    const inter = w * h;
    const union_ = if (iof) area_a else area_a + area_b - inter;
    return inter / @max(union_, 1e-6);
}

/// ATSS: per level and ground truth, the `topk` anchors whose centres are nearest the truth's
/// centre are candidates; those whose IoU reaches the candidates' mean + std and whose centre
/// lies inside the truth are positives; an anchor positive for several truths takes the one it
/// overlaps most. Returns, per anchor, -1 (ignored: inside an ignore region by IoF), 0
/// (background) or k + 1 (truth k). With no truths, everything is background -- ignore regions
/// included, as MMDetection returns before looking at them.
pub fn assign(gpa: Allocator, all: []const Box, gts: []const Box, ignores: []const Box, ignore_iof: f32, topk: usize) ![]i32 {
    const assigned = try gpa.alloc(i32, all.len);
    @memset(assigned, 0);
    if (gts.len == 0 or all.len == 0) return assigned;
    const inf: f32 = 100000000;

    const ignored = try gpa.alloc(bool, all.len);
    defer gpa.free(ignored);
    @memset(ignored, false);
    if (ignore_iof > 0) for (all, 0..) |a, i| {
        var best: f32 = -std.math.inf(f32);
        for (ignores) |g| best = @max(best, overlap(a, g, true));
        if (ignores.len > 0 and best > ignore_iof) {
            ignored[i] = true;
            assigned[i] = -1;
        }
    };

    // Best IoU per anchor among the truths it is a positive candidate for.
    const best_iou = try gpa.alloc(f32, all.len);
    defer gpa.free(best_iou);
    @memset(best_iou, -inf);
    const candidates = try gpa.alloc(usize, topk * strides.len);
    defer gpa.free(candidates);
    const distances = try gpa.alloc(f32, level_counts[0]);
    defer gpa.free(distances);
    const order = try gpa.alloc(usize, level_counts[0]);
    defer gpa.free(order);

    for (gts, 0..) |g, gi| {
        const gc = centre(g);
        var count: usize = 0;
        var start: usize = 0;
        for (level_counts) |n| {
            const d = distances[0..n];
            for (d, 0..) |*v, k| {
                if (ignored[start + k]) {
                    v.* = inf;
                } else {
                    const ac = centre(all[start + k]);
                    v.* = @sqrt((ac[0] - gc[0]) * (ac[0] - gc[0]) + (ac[1] - gc[1]) * (ac[1] - gc[1]));
                }
            }
            const k_level = @min(topk, n);
            // The k nearest, ties to the lower index (a stable partial selection).
            const idx = order[0..n];
            for (idx, 0..) |*o, k| o.* = k;
            std.sort.block(usize, idx, d, struct {
                fn nearer(dist: []const f32, a: usize, b: usize) bool {
                    return dist[a] < dist[b];
                }
            }.nearer);
            for (idx[0..k_level]) |k| {
                candidates[count] = start + k;
                count += 1;
            }
            start += n;
        }
        const cand = candidates[0..count];
        var mean: f32 = 0;
        for (cand) |c| mean += overlap(all[c], g, false);
        mean /= @floatFromInt(count);
        var sq: f32 = 0;
        for (cand) |c| {
            const v = overlap(all[c], g, false) - mean;
            sq += v * v;
        }
        const threshold = mean + @sqrt(sq / @as(f32, @floatFromInt(count - 1)));
        const gt_area = @sqrt(@max((g[2] - g[0]) * (g[3] - g[1]), 1e-4));
        for (cand) |c| {
            const iou = overlap(all[c], g, false);
            if (!(iou >= threshold)) continue;
            const ac = centre(all[c]);
            const inside = @min(@min(ac[0] - g[0], ac[1] - g[1]), @min(g[2] - ac[0], g[3] - ac[1])) / gt_area;
            if (!(inside > 0.001)) continue;
            if (iou > best_iou[c]) { // first truth wins ties, as torch's max does
                best_iou[c] = iou;
                assigned[c] = @intCast(gi + 1);
            }
        }
    }
    return assigned;
}

/// A face after augmentation: its box and five keypoints (x, y, visibility weight).
pub const Face = struct { box: Box, kps: [5][3]f32 };

/// One positive anchor's targets, in pixels as `_get_target_single` stores them.
pub const Positive = struct {
    anchor: u32,
    box: Box,
    kps: [10]f32,
    kps_weight: f32, // mean of the five visibility weights
};

pub const Targets = struct {
    /// Per anchor: 1 to count in the classification loss, 0 for ignored anchors.
    label_weight: []u8,
    positives: []Positive,
};

/// The assignment turned into training targets for one image.
pub fn targets(gpa: Allocator, all: []const Box, faces: []const Face, ignores: []const Box, ignore_iof: f32) !Targets {
    const gts = try gpa.alloc(Box, faces.len);
    defer gpa.free(gts);
    for (gts, faces) |*g, f| g.* = f.box;
    const assigned = try assign(gpa, all, gts, ignores, ignore_iof, 9);
    defer gpa.free(assigned);
    const label_weight = try gpa.alloc(u8, all.len);
    var positives: std.ArrayList(Positive) = .empty;
    for (assigned, label_weight, 0..) |a, *w, i| {
        w.* = @intFromBool(a >= 0);
        if (a <= 0) continue;
        const f = faces[@intCast(a - 1)];
        var p: Positive = .{ .anchor = @intCast(i), .box = f.box, .kps = undefined, .kps_weight = 0 };
        for (0..5) |k| {
            p.kps[k * 2] = f.kps[k][0];
            p.kps[k * 2 + 1] = f.kps[k][1];
            p.kps_weight += f.kps[k][2];
        }
        p.kps_weight /= 5;
        try positives.append(gpa, p);
    }
    return .{ .label_weight = label_weight, .positives = try positives.toOwnedSlice(gpa) };
}

test "anchors follow MMDetection's grid order and sizes" {
    const a = comptime anchors();
    try std.testing.expectEqual(@as(usize, 16800), a.len);
    try std.testing.expectEqualSlices(f32, &.{ -8, -8, 8, 8 }, &a[0]);
    try std.testing.expectEqualSlices(f32, &.{ -16, -16, 16, 16 }, &a[1]);
    try std.testing.expectEqualSlices(f32, &.{ 0, -8, 16, 8 }, &a[2]);
    try std.testing.expectEqualSlices(f32, &.{ -32, -32, 32, 32 }, &a[12800]);
    try std.testing.expectEqual(@as(usize, 32), strideOf(16799));
}

test "a face gets positives at the level its size suits, inside its box; ignore regions mask anchors" {
    const gpa = std.testing.allocator;
    const a = comptime anchors();
    const face: Face = .{ .box = .{ 300, 300, 364, 364 }, .kps = @splat(.{ 0, 0, 1 }) };
    const t = try targets(gpa, &a, &.{face}, &.{.{ 0, 0, 100, 100 }}, 0.5);
    defer gpa.free(t.label_weight);
    defer gpa.free(t.positives);
    try std.testing.expect(t.positives.len > 0);
    for (t.positives) |p| {
        const c = centre(a[p.anchor]);
        try std.testing.expect(c[0] > 300 and c[0] < 364 and c[1] > 300 and c[1] < 364);
        try std.testing.expectEqual(@as(f32, 1), p.kps_weight);
    }
    try std.testing.expectEqual(@as(u8, 1), t.label_weight[0]); // (0,0): IoF 0.25 with the region, below 0.5
    try std.testing.expectEqual(@as(u8, 0), t.label_weight[(5 * 80 + 5) * 2]); // (40,40): wholly inside it
    try std.testing.expectEqual(@as(u8, 1), t.label_weight[12800 - 1]);
}
