//! Independent CPU reference for the unfused refiner checkpoint, used to verify exports.
const std = @import("std");
const ck = @import("checkpoint.zig");
const img = @import("image.zig");
const Allocator = std.mem.Allocator;
pub const size = 112;
pub const channels = [_]usize{ 3, 32, 32, 64, 64, 128, 128, 128, 128, 256, 256 };

pub const Conv = struct {
    input: usize,
    output: usize,
    weight: []const f32,
    scale: []const f32,
    shift: []const f32,
};

pub const Linear = struct { input: usize, output: usize, weight: []const f32, bias: []const f32 };

pub const Refiner = struct {
    convs: [10]Conv,
    head: [2]Linear,

    pub fn fromCheckpoint(gpa: Allocator, value: ck.Value) !Refiner {
        if (value != .dict) return error.InvalidRefinerCheckpoint;
        const input_size = value.dict.get("size") orelse return error.MissingCropSize;
        if (input_size != .integer or input_size.integer != size) return error.IncompatibleCropSize;
        const model = value.dict.get("model") orelse return error.MissingModelState;
        if (model != .dict or model.dict.items.items.len != 64) return error.InvalidParameterCount;
        var result: Refiner = undefined;
        for (&result.convs, 0..) |*conv, i| {
            const input = channels[i];
            const output = channels[i + 1];
            const weight = try parameter(gpa, model.dict, try std.fmt.allocPrint(gpa, "features.{d}.weight", .{i * 3}), &.{ output, input, 3, 3 });
            const bn = try std.fmt.allocPrint(gpa, "features.{d}", .{i * 3 + 1});
            const gamma = try parameter(gpa, model.dict, try std.fmt.allocPrint(gpa, "{s}.weight", .{bn}), &.{output});
            const beta = try parameter(gpa, model.dict, try std.fmt.allocPrint(gpa, "{s}.bias", .{bn}), &.{output});
            const mean = try parameter(gpa, model.dict, try std.fmt.allocPrint(gpa, "{s}.running_mean", .{bn}), &.{output});
            const variance = try parameter(gpa, model.dict, try std.fmt.allocPrint(gpa, "{s}.running_var", .{bn}), &.{output});
            const counter = model.dict.get(try std.fmt.allocPrint(gpa, "{s}.num_batches_tracked", .{bn})) orelse return error.MissingParameter;
            if (counter != .tensor or counter.tensor.storage.dtype != .int64 or counter.tensor.shape.len != 0) return error.InvalidParameterShape;
            const scale = try gpa.alloc(f32, output);
            const shift = try gpa.alloc(f32, output);
            for (scale, shift, gamma, beta, mean, variance) |*s, *b, g, bias, m, v| {
                if (v < 0) return error.InvalidRunningVariance;
                s.* = g / @sqrt(v + 1e-5);
                b.* = bias - m * s.*;
                if (!std.math.isFinite(s.*) or !std.math.isFinite(b.*)) return error.NonFiniteParameter;
            }
            conv.* = .{ .input = input, .output = output, .weight = weight, .scale = scale, .shift = shift };
        }
        for (&result.head, [_]usize{ 2, 4 }, [_]usize{ 256, 128 }, [_]usize{ 128, 4 }) |*linear, index, input, output| {
            linear.* = .{
                .input = input,
                .output = output,
                .weight = try parameter(gpa, model.dict, try std.fmt.allocPrint(gpa, "head.{d}.weight", .{index}), &.{ output, input }),
                .bias = try parameter(gpa, model.dict, try std.fmt.allocPrint(gpa, "head.{d}.bias", .{index}), &.{output}),
            };
        }
        return result;
    }

    pub fn predict(self: Refiner, gpa: Allocator, input: []const f32) ![4]f32 {
        if (input.len != 3 * size * size) return error.InvalidInputShape;
        var activation: []const f32 = input;
        var owned: ?[]const f32 = null;
        defer if (owned) |data| gpa.free(data);
        var side: usize = size;
        for (self.convs, 0..) |conv, i| {
            const stride: usize = if (i % 2 == 0) 2 else 1;
            const next = try convolution(gpa, conv, activation, side, stride);
            if (owned) |data| gpa.free(data);
            owned = next;
            activation = next;
            side = (side - 1) / stride + 1;
        }
        var pooled: [256]f32 = undefined;
        const pixels = side * side;
        for (&pooled, 0..) |*v, c| {
            var sum: f32 = 0;
            for (activation[c * pixels ..][0..pixels]) |x| sum += x;
            v.* = sum / @as(f32, @floatFromInt(pixels));
        }
        var hidden: [128]f32 = undefined;
        dense(self.head[0], &pooled, &hidden, true);
        var output: [4]f32 = undefined;
        dense(self.head[1], &hidden, &output, false);
        for (output) |v| if (!std.math.isFinite(v)) return error.NonFinitePrediction;
        return output;
    }
};

/// The refiner's read of one detection, as the evaluation tools crop it: the box centre (float32,
/// as NumPy held it), the longer side grown 1.25x, scaled to the crop without rotation. Returns
/// the subject's right then left eye in image pixels, in float32 like torch's output.
/// `oracle` runs the exported graph (`export_eye_refiner.buildOnnx`); link `-lc vendor/stb.c`.
pub fn readEyes(gpa: Allocator, oracle: anytype, frame: img.Image, centre: [2]f32, side: f64) ![2][2]f32 {
    var m = img.rotationMatrix(centre[0], centre[1], 0, size / side);
    m[2] += size / 2.0 - @as(f64, centre[0]);
    m[5] += size / 2.0 - @as(f64, centre[1]);
    const crop = try img.warpAffine(gpa, frame, m, size, size);
    defer crop.deinit(gpa);
    const plane = size * size;
    const input = try gpa.alloc(f32, 3 * plane);
    defer gpa.free(input);
    for (0..3) |c| for (0..plane) |p| {
        input[c * plane + p] = (@as(f32, @floatFromInt(crop.pixels[p * 3 + c])) - 127.5) / 128.0;
    };
    const out = try oracle.run(gpa, input, &.{ 1, 3, size, size }, "crop", "eyes");
    if (out.data.len != 4) return error.InvalidOutputShape;
    const half: f32 = size / 2.0;
    const back: f32 = @floatCast(side / size);
    var eyes: [2][2]f32 = undefined;
    for (0..2) |k| for (0..2) |c| {
        eyes[k][c] = (out.data[k * 2 + c] * size - half) * back + centre[c];
    };
    return eyes;
}

fn parameter(gpa: Allocator, model: *ck.Dict, name: []const u8, shape: []const usize) ![]const f32 {
    const value = model.get(name) orelse {
        std.debug.print("missing parameter: {s}\n", .{name});
        return error.MissingParameter;
    };
    if (value != .tensor or value.tensor.storage.dtype != .float32 or !std.mem.eql(usize, shape, value.tensor.shape)) return error.InvalidParameterShape;
    const data = try value.tensor.floats(gpa);
    for (data) |v| if (!std.math.isFinite(v)) return error.NonFiniteParameter;
    return data;
}

fn convolution(gpa: Allocator, layer: Conv, input: []const f32, side: usize, stride: usize) ![]f32 {
    const out_side = (side - 1) / stride + 1;
    const output = try gpa.alloc(f32, layer.output * out_side * out_side);
    for (0..layer.output) |o| {
        for (0..out_side) |y| {
            for (0..out_side) |x| {
                var sum: f32 = 0;
                for (0..layer.input) |c| {
                    for (0..3) |ky| {
                        const iy = @as(isize, @intCast(y * stride + ky)) - 1;
                        if (iy < 0 or iy >= side) continue;
                        for (0..3) |kx| {
                            const ix = @as(isize, @intCast(x * stride + kx)) - 1;
                            if (ix < 0 or ix >= side) continue;
                            sum += input[c * side * side + @as(usize, @intCast(iy)) * side + @as(usize, @intCast(ix))] * layer.weight[(o * layer.input + c) * 9 + ky * 3 + kx];
                        }
                    }
                }
                output[(o * out_side + y) * out_side + x] = @max(0, sum * layer.scale[o] + layer.shift[o]);
            }
        }
    }
    return output;
}

fn dense(layer: Linear, input: []const f32, output: []f32, relu: bool) void {
    for (output, 0..) |*v, o| {
        var sum = layer.bias[o];
        for (input, layer.weight[o * layer.input ..][0..layer.input]) |x, w| sum += x * w;
        v.* = if (relu) @max(0, sum) else sum;
    }
}

test "unfused convolution handles padding, stride, BatchNorm and ReLU" {
    const layer: Conv = .{ .input = 1, .output = 1, .weight = &.{ 1, 1, 1, 1, 1, 1, 1, 1, 1 }, .scale = &.{2}, .shift = &.{-1} };
    const out = try convolution(std.testing.allocator, layer, &.{ 1, 2, 3, 4 }, 2, 1);
    defer std.testing.allocator.free(out);
    try std.testing.expectEqualSlices(f32, &.{ 19, 19, 19, 19 }, out);
    const strided = try convolution(std.testing.allocator, layer, &.{ 1, 2, 3, 4 }, 2, 2);
    defer std.testing.allocator.free(strided);
    try std.testing.expectEqualSlices(f32, &.{19}, strided);
    const negative = try convolution(std.testing.allocator, layer, &.{ -1, -2, -3, -4 }, 2, 1);
    defer std.testing.allocator.free(negative);
    try std.testing.expectEqualSlices(f32, &.{ 0, 0, 0, 0 }, negative);
}

test "only the hidden dense layer clips negative values" {
    var hidden: [2]f32 = undefined;
    dense(.{ .input = 2, .output = 2, .weight = &.{ 1, 0, 0, -1 }, .bias = &.{ 0.5, 0 } }, &.{ 1, 2 }, &hidden, true);
    var output: [1]f32 = undefined;
    dense(.{ .input = 2, .output = 1, .weight = &.{ -1, 3 }, .bias = &.{0.25} }, &hidden, &output, false);
    try std.testing.expectEqualSlices(f32, &.{-1.25}, &output);
}

test "the complete synthetic checkpoint loads, including broadcast tensor views" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const gpa = arena.allocator();
    const checkpoint = try ck.load(gpa, std.testing.io, "tools/dataset/testdata/refiner.pt");
    const model = try Refiner.fromCheckpoint(gpa, checkpoint);
    try std.testing.expectEqualSlices(f32, &.{ 0.25, 0.5, -0.75, 1.25 }, model.head[1].bias);
    try std.testing.expectEqual(@as(usize, 32 * 3 * 9), model.convs[0].weight.len);
    try std.testing.expectApproxEqAbs(@as(f32, 1 / @sqrt(@as(f32, 1 + 1e-5))), model.convs[0].scale[0], 1e-6);
    const parameters = checkpoint.dict.get("model").?.dict;
    parameters.items.items[0].value.tensor.shape = &.{1};
    try std.testing.expectError(error.InvalidParameterShape, Refiner.fromCheckpoint(gpa, checkpoint));
}
