//! SCRFD-500M (the `scrfd_500m_bnkps` architecture) read from an mmdet checkpoint, as a program of
//! convolution, add and upsample steps, plus an independent CPU forward pass with BatchNorm left
//! unfolded, used to verify exports. Channel counts, groups and block counts come from the
//! checkpoint's tensor shapes; the wiring is mmdet's MobileNetV1 + PAFPN + SCRFDHead.
const std = @import("std");
const ck = @import("checkpoint.zig");
const Allocator = std.mem.Allocator;

pub const eps: f32 = 1e-5; // mmcv's BatchNorm default
pub const anchors = 2;
pub const strides = [_]usize{ 8, 16, 32 };
/// Channels per anchor of the class, box and keypoint maps.
pub const per_anchor = [_]usize{ 1, 4, 10 };

pub const Bn = struct { gamma: []const f32, beta: []const f32, mean: []const f32, variance: []const f32 };

pub const Conv = struct {
    /// The checkpoint name without `.weight`, e.g. `backbone.stem.0.0`.
    name: []const u8,
    out: usize,
    in_per_group: usize,
    kernel: usize,
    stride: usize,
    groups: usize,
    weight: []const f32,
    bias: ?[]const f32,
    bn: ?Bn,
    relu: bool,

    pub fn pad(self: Conv) usize {
        return self.kernel / 2;
    }
};

/// Step `i` writes value `i + 1`; value 0 is the input image.
pub const Step = union(enum) {
    conv: struct { input: usize, conv: Conv },
    add: [2]usize,
    upsample: usize,
};

pub const Scrfd = struct {
    steps: []const Step,
    /// Channels and downsampling factor of every value.
    channels: []const usize,
    scale: []const usize,
    /// Per stride: the values holding the raw class, box and keypoint maps.
    heads: [3][3]usize,

    pub fn fromCheckpoint(gpa: Allocator, checkpoint: ck.Value) !Scrfd {
        if (checkpoint != .dict) return error.InvalidCheckpoint;
        const state = checkpoint.dict.get("state_dict") orelse return error.MissingStateDict;
        if (state != .dict) return error.InvalidCheckpoint;
        var b: Builder = .{ .gpa = gpa, .state = state.dict };
        try b.channels.append(gpa, 3);
        try b.scale.append(gpa, 1);

        var x = try b.conv(0, "backbone.stem.0.0", "backbone.stem.0.1", 2, true);
        x = try b.block(x, "backbone.stem.1", 1);
        var features: [3]usize = undefined;
        for (1..5) |layer| {
            var count: usize = 0;
            while (b.has(try b.fmt("backbone.layer{d}.{d}.0.weight", .{ layer, count }))) : (count += 1) {
                x = try b.block(x, try b.fmt("backbone.layer{d}.{d}", .{ layer, count }), if (count == 0) 2 else 1);
            }
            if (count == 0) return error.MissingBackboneLayer;
            if (layer >= 2) features[layer - 2] = x; // PAFPN start_level=1
        }

        const l0 = try b.conv(features[0], "neck.lateral_convs.0.conv", null, 1, false);
        const l1 = try b.conv(features[1], "neck.lateral_convs.1.conv", null, 1, false);
        const l2 = try b.conv(features[2], "neck.lateral_convs.2.conv", null, 1, false);
        const top = try b.add(l1, try b.upsample(l2));
        const bottom = try b.add(l0, try b.upsample(top));
        const f0 = try b.conv(bottom, "neck.fpn_convs.0.conv", null, 1, false);
        const f1 = try b.conv(top, "neck.fpn_convs.1.conv", null, 1, false);
        const f2 = try b.conv(l2, "neck.fpn_convs.2.conv", null, 1, false);
        const p1_in = try b.add(f1, try b.conv(f0, "neck.downsample_convs.0.conv", null, 2, false));
        const p2_in = try b.add(f2, try b.conv(p1_in, "neck.downsample_convs.1.conv", null, 2, false));
        const levels = [3]usize{ f0, try b.conv(p1_in, "neck.pafpn_convs.0.conv", null, 1, false), try b.conv(p2_in, "neck.pafpn_convs.1.conv", null, 1, false) };

        var heads: [3][3]usize = undefined;
        for (strides, levels, &heads) |s, level, *head| {
            var y = level;
            var k: usize = 0;
            while (b.has(try b.fmt("bbox_head.cls_stride_convs.({d}, {d}).{d}.depthwise_conv.conv.weight", .{ s, s, k }))) : (k += 1) {
                const p = try b.fmt("bbox_head.cls_stride_convs.({d}, {d}).{d}", .{ s, s, k });
                y = try b.conv(y, try b.fmt("{s}.depthwise_conv.conv", .{p}), try b.fmt("{s}.depthwise_conv.bn", .{p}), 1, true);
                y = try b.conv(y, try b.fmt("{s}.pointwise_conv.conv", .{p}), try b.fmt("{s}.pointwise_conv.bn", .{p}), 1, true);
            }
            for ([_][]const u8{ "cls", "reg", "kps" }, per_anchor, head) |kind, per, *v| {
                v.* = try b.conv(y, try b.fmt("bbox_head.stride_{s}.({d}, {d})", .{ kind, s, s }), null, 1, false);
                if (b.channels.items[v.*] != anchors * per) return error.UnexpectedHeadChannels;
            }
        }
        return .{ .steps = try b.steps.toOwnedSlice(gpa), .channels = try b.channels.toOwnedSlice(gpa), .scale = try b.scale.toOwnedSlice(gpa), .heads = heads };
    }

    /// The nine outputs in the export's deployment layout -- each (1, A*C, H, W) map as
    /// (H*W*A, C), spatial-major with anchors interleaved, class maps sigmoided -- ordered
    /// class, box, keypoints for strides 8, 16, 32.
    pub fn forward(self: Scrfd, gpa: Allocator, input: []const f32, size: usize) ![9][]f32 {
        if (size % 32 != 0 or input.len != 3 * size * size) return error.InvalidInputShape;
        const values = try gpa.alloc([]const f32, self.channels.len);
        defer {
            for (values[1..]) |v| gpa.free(v);
            gpa.free(values);
        }
        values[0] = input;
        for (self.steps, 1..) |step, id| {
            const side = size / self.scale[id];
            values[id] = switch (step) {
                .conv => |c| try convolve(gpa, c.conv, values[c.input], size / self.scale[c.input]),
                .add => |pair| blk: {
                    const out = try gpa.alloc(f32, values[pair[0]].len);
                    for (out, values[pair[0]], values[pair[1]]) |*o, a, b| o.* = a + b;
                    break :blk out;
                },
                .upsample => |from| blk: {
                    const channels = self.channels[id];
                    const out = try gpa.alloc(f32, channels * side * side);
                    const half = side / 2;
                    for (0..channels) |c| for (0..side) |y| for (0..side) |x| {
                        out[(c * side + y) * side + x] = values[from][(c * half + y / 2) * half + x / 2];
                    };
                    break :blk out;
                },
            };
        }
        var outputs: [9][]f32 = undefined;
        for (0..3) |kind| for (0..3) |level| {
            const v = self.heads[level][kind];
            outputs[kind * 3 + level] = try deploy(gpa, values[v], size / self.scale[v], per_anchor[kind], kind == 0);
        };
        return outputs;
    }
};

const Builder = struct {
    gpa: Allocator,
    state: *ck.Dict,
    steps: std.ArrayList(Step) = .empty,
    channels: std.ArrayList(usize) = .empty,
    scale: std.ArrayList(usize) = .empty,

    fn fmt(self: *Builder, comptime f: []const u8, args: anytype) ![]const u8 {
        return std.fmt.allocPrint(self.gpa, f, args);
    }

    fn has(self: *Builder, key: []const u8) bool {
        return self.state.get(key) != null;
    }

    fn tensor(self: *Builder, key: []const u8) !struct { []const f32, []const usize } {
        const v = self.state.get(key) orelse {
            std.debug.print("missing parameter: {s}\n", .{key});
            return error.MissingParameter;
        };
        if (v != .tensor or v.tensor.storage.dtype != .float32) return error.InvalidParameter;
        const data = try v.tensor.floats(self.gpa);
        for (data) |x| if (!std.math.isFinite(x)) return error.NonFiniteParameter;
        return .{ data, v.tensor.shape };
    }

    fn vector(self: *Builder, key: []const u8, n: usize) ![]const f32 {
        const data, const shape = try self.tensor(key);
        if (shape.len != 1 or shape[0] != n) return error.InvalidParameterShape;
        return data;
    }

    fn push(self: *Builder, step: Step, channels: usize, scale: usize) !usize {
        try self.steps.append(self.gpa, step);
        try self.channels.append(self.gpa, channels);
        try self.scale.append(self.gpa, scale);
        return self.steps.items.len;
    }

    fn conv(self: *Builder, input: usize, name: []const u8, bn_name: ?[]const u8, stride: usize, relu: bool) !usize {
        const weight, const shape = try self.tensor(try self.fmt("{s}.weight", .{name}));
        if (shape.len != 4 or shape[2] != shape[3] or shape[2] % 2 != 1 or shape[1] == 0) return error.InvalidConvWeight;
        const in = self.channels.items[input];
        if (in % shape[1] != 0) return error.InvalidConvWeight;
        const groups = in / shape[1];
        if (shape[0] % groups != 0) return error.InvalidConvWeight;
        const out = shape[0];
        const bias_key = try self.fmt("{s}.bias", .{name});
        const bn: ?Bn = if (bn_name) |p| .{
            .gamma = try self.vector(try self.fmt("{s}.weight", .{p}), out),
            .beta = try self.vector(try self.fmt("{s}.bias", .{p}), out),
            .mean = try self.vector(try self.fmt("{s}.running_mean", .{p}), out),
            .variance = try self.vector(try self.fmt("{s}.running_var", .{p}), out),
        } else null;
        if (bn) |n| for (n.variance) |v| if (v < 0) return error.InvalidRunningVariance;
        const c: Conv = .{
            .name = name,
            .out = out,
            .in_per_group = shape[1],
            .kernel = shape[2],
            .stride = stride,
            .groups = groups,
            .weight = weight,
            .bias = if (self.has(bias_key)) try self.vector(bias_key, out) else null,
            .bn = bn,
            .relu = relu,
        };
        return self.push(.{ .conv = .{ .input = input, .conv = c } }, out, self.scale.items[input] * stride);
    }

    /// MobileNetV1's depthwise-separable block: 3x3 depthwise (`.0`, BN `.1`), 1x1 pointwise (`.3`, BN `.4`).
    fn block(self: *Builder, input: usize, prefix: []const u8, stride: usize) !usize {
        const dw = try self.conv(input, try self.fmt("{s}.0", .{prefix}), try self.fmt("{s}.1", .{prefix}), stride, true);
        return self.conv(dw, try self.fmt("{s}.3", .{prefix}), try self.fmt("{s}.4", .{prefix}), 1, true);
    }

    fn add(self: *Builder, a: usize, b: usize) !usize {
        if (self.channels.items[a] != self.channels.items[b] or self.scale.items[a] != self.scale.items[b]) return error.MismatchedAdd;
        return self.push(.{ .add = .{ a, b } }, self.channels.items[a], self.scale.items[a]);
    }

    fn upsample(self: *Builder, input: usize) !usize {
        if (self.scale.items[input] % 2 != 0) return error.InvalidUpsample;
        return self.push(.{ .upsample = input }, self.channels.items[input], self.scale.items[input] / 2);
    }
};

/// Grouped convolution, then BatchNorm as trained (not folded), then ReLU.
fn convolve(gpa: Allocator, c: Conv, x: []const f32, side: usize) ![]f32 {
    const p = c.pad();
    const out_side = (side + 2 * p - c.kernel) / c.stride + 1;
    const out = try gpa.alloc(f32, c.out * out_side * out_side);
    const per_group = c.out / c.groups;
    const k = c.kernel;
    for (0..c.out) |o| {
        const g = o / per_group;
        const w = c.weight[o * c.in_per_group * k * k ..][0 .. c.in_per_group * k * k];
        const shift = if (c.bn) |n| [2]f32{ n.gamma[o] / @sqrt(n.variance[o] + eps), n.mean[o] } else [2]f32{ 1, 0 };
        for (0..out_side) |oy| for (0..out_side) |ox| {
            var sum: f32 = 0;
            for (0..c.in_per_group) |ci| {
                const plane = x[(g * c.in_per_group + ci) * side * side ..][0 .. side * side];
                for (0..k) |ky| {
                    const iy = @as(isize, @intCast(oy * c.stride + ky)) - @as(isize, @intCast(p));
                    if (iy < 0 or iy >= side) continue;
                    for (0..k) |kx| {
                        const ix = @as(isize, @intCast(ox * c.stride + kx)) - @as(isize, @intCast(p));
                        if (ix < 0 or ix >= side) continue;
                        sum += plane[@as(usize, @intCast(iy)) * side + @as(usize, @intCast(ix))] * w[(ci * k + ky) * k + kx];
                    }
                }
            }
            if (c.bias) |bias| sum += bias[o];
            var v = sum;
            if (c.bn) |n| v = (sum - shift[1]) * shift[0] + n.beta[o];
            out[(o * out_side + oy) * out_side + ox] = if (c.relu) @max(0, v) else v;
        };
    }
    return out;
}

/// (A*C, H, W) to (H*W*A, C), optionally sigmoided.
pub fn deploy(gpa: Allocator, map: []const f32, side: usize, per: usize, sigmoid: bool) ![]f32 {
    const plane = side * side;
    const out = try gpa.alloc(f32, map.len);
    for (0..plane) |p| for (0..anchors) |a| for (0..per) |c| {
        const v = map[(a * per + c) * plane + p];
        out[(p * anchors + a) * per + c] = if (sigmoid) 1 / (1 + @exp(-v)) else v;
    };
    return out;
}

test "grouped convolution: depthwise channels stay separate, padding and stride hold" {
    const gpa = std.testing.allocator;
    // Two channels, depthwise 3x3 of ones, stride 1: each output is the sum of its 3x3 window.
    const ones: [18]f32 = @splat(1);
    const c: Conv = .{ .name = "", .out = 2, .in_per_group = 1, .kernel = 3, .stride = 1, .groups = 2, .weight = &ones, .bias = &.{ 0, 100 }, .bn = null, .relu = false };
    const x = [_]f32{ 1, 2, 3, 4, 10, 20, 30, 40 }; // two 2x2 planes
    const out = try convolve(gpa, c, &x, 2);
    defer gpa.free(out);
    try std.testing.expectEqualSlices(f32, &.{ 10, 10, 10, 10, 200, 200, 200, 200 }, out);
    const strided = try convolve(gpa, .{ .name = "", .out = 2, .in_per_group = 1, .kernel = 3, .stride = 2, .groups = 2, .weight = &ones, .bias = null, .bn = .{ .gamma = &.{ 2, 1 }, .beta = &.{ 0, -1000 }, .mean = &.{ 0, 0 }, .variance = &.{ 1 - eps, 1 - eps } }, .relu = true }, &x, 2);
    defer gpa.free(strided);
    try std.testing.expectEqualSlices(f32, &.{ 20, 0 }, strided); // BN scales the first, ReLU clips the second
}

test "deployment layout interleaves anchors under each location" {
    const gpa = std.testing.allocator;
    // Two anchors, one channel, a 1x2 map... as a 2x2 plane: values encode (anchor, pixel).
    const map = [_]f32{ 0, 1, 2, 3, 10, 11, 12, 13 };
    const out = try deploy(gpa, &map, 2, 1, false);
    defer gpa.free(out);
    try std.testing.expectEqualSlices(f32, &.{ 0, 10, 1, 11, 2, 12, 3, 13 }, out);
}
