//! Export a legacy refiner checkpoint to ONNX, verified before replacing the destination.
//!   zig run -O ReleaseFast -lc tools/dataset/export_eye_refiner.zig -- <checkpoint.pt> <out.onnx> [--tolerance 1e-5] [--ort path]
//! ORT_DYLIB_PATH selects ONNX Runtime (API 24+); -lc is only required on Unix.
const std = @import("std");
const builtin = @import("builtin");
const ck = @import("checkpoint.zig");
const net = @import("refiner.zig");
const pb = @import("onnx.zig");
const ort = @import("ort.zig");
const metadata = @import("stamp_model_metadata.zig");
const Allocator = std.mem.Allocator;

fn intsAttribute(gpa: Allocator, name: []const u8, values: []const u64) ![]const u8 {
    var out: std.ArrayList(u8) = .empty;
    try pb.putBytes(&out, gpa, 1, name);
    for (values) |v| try pb.putInt(&out, gpa, 8, v);
    try pb.putInt(&out, gpa, 20, 7);
    return out.toOwnedSlice(gpa);
}

fn intAttribute(gpa: Allocator, name: []const u8, value: u64) ![]const u8 {
    var out: std.ArrayList(u8) = .empty;
    try pb.putBytes(&out, gpa, 1, name);
    try pb.putInt(&out, gpa, 3, value);
    try pb.putInt(&out, gpa, 20, 2);
    return out.toOwnedSlice(gpa);
}

fn node(gpa: Allocator, graph: *std.ArrayList(u8), op: []const u8, inputs: []const []const u8, output: []const u8, attrs: []const []const u8) !void {
    var out: std.ArrayList(u8) = .empty;
    defer out.deinit(gpa);
    for (inputs) |input| try pb.putBytes(&out, gpa, 1, input);
    try pb.putBytes(&out, gpa, 2, output);
    try pb.putBytes(&out, gpa, 4, op);
    for (attrs) |attr| try pb.putBytes(&out, gpa, 5, attr);
    try pb.putBytes(graph, gpa, 1, out.items);
}

fn tensor(gpa: Allocator, graph: *std.ArrayList(u8), name: []const u8, shape: []const usize, values: []const f32) !void {
    var out: std.ArrayList(u8) = .empty;
    defer out.deinit(gpa);
    var count: usize = 1;
    for (shape) |dim| {
        count = std.math.mul(usize, count, dim) catch return error.InvalidParameterShape;
        try pb.putInt(&out, gpa, 1, dim);
    }
    if (count != values.len) return error.InvalidParameterShape;
    try pb.putInt(&out, gpa, 2, 1);
    try pb.putBytes(&out, gpa, 8, name);
    const raw = try gpa.alloc(u8, try std.math.mul(usize, values.len, 4));
    defer gpa.free(raw);
    for (values, 0..) |v, i| {
        if (!std.math.isFinite(v)) return error.NonFiniteParameter;
        std.mem.writeInt(u32, raw[i * 4 ..][0..4], @bitCast(v), .little);
    }
    try pb.putBytes(&out, gpa, 9, raw);
    try pb.putBytes(graph, gpa, 5, out.items);
}

/// The first dimension is symbolic; the remaining dimensions are fixed.
fn valueInfo(gpa: Allocator, name: []const u8, dims: []const u64) ![]const u8 {
    var symbolic: std.ArrayList(u8) = .empty;
    try pb.putBytes(&symbolic, gpa, 2, "batch");
    var shape: std.ArrayList(u8) = .empty;
    try pb.putBytes(&shape, gpa, 1, symbolic.items);
    for (dims) |dim| {
        var axis: std.ArrayList(u8) = .empty;
        defer axis.deinit(gpa);
        try pb.putInt(&axis, gpa, 1, dim);
        try pb.putBytes(&shape, gpa, 1, axis.items);
    }
    var tensor_type: std.ArrayList(u8) = .empty;
    try pb.putInt(&tensor_type, gpa, 1, 1);
    try pb.putBytes(&tensor_type, gpa, 2, shape.items);
    var type_proto: std.ArrayList(u8) = .empty;
    try pb.putBytes(&type_proto, gpa, 1, tensor_type.items);
    var out: std.ArrayList(u8) = .empty;
    try pb.putBytes(&out, gpa, 1, name);
    try pb.putBytes(&out, gpa, 2, type_proto.items);
    return out.toOwnedSlice(gpa);
}

pub fn buildOnnx(gpa: Allocator, model: net.Refiner) ![]const u8 {
    var graph: std.ArrayList(u8) = .empty;
    try pb.putBytes(&graph, gpa, 2, "Face Crop Studio eye refiner");
    var input: []const u8 = "crop";
    for (model.convs, 0..) |conv, i| {
        // Keep the exact names the Rust refiner loads; they no longer depend on torch's tracer.
        const weight_name = try std.fmt.allocPrint(gpa, "onnx::Conv_{d}", .{101 + i * 3});
        const bias_name = try std.fmt.allocPrint(gpa, "onnx::Conv_{d}", .{102 + i * 3});
        const folded = try gpa.alloc(f32, conv.weight.len);
        for (folded, conv.weight, 0..) |*w, original, k| w.* = original * conv.scale[k / (conv.input * 9)];
        try tensor(gpa, &graph, weight_name, &.{ conv.output, conv.input, 3, 3 }, folded);
        try tensor(gpa, &graph, bias_name, &.{conv.output}, conv.shift);
        const convolved = try std.fmt.allocPrint(gpa, "conv_{d}", .{i});
        const activated = try std.fmt.allocPrint(gpa, "relu_{d}", .{i});
        const stride: u64 = if (i % 2 == 0) 2 else 1;
        try node(gpa, &graph, "Conv", &.{ input, weight_name, bias_name }, convolved, &.{
            try intsAttribute(gpa, "kernel_shape", &.{ 3, 3 }),
            try intsAttribute(gpa, "strides", &.{ stride, stride }),
            try intsAttribute(gpa, "pads", &.{ 1, 1, 1, 1 }),
        });
        try node(gpa, &graph, "Relu", &.{convolved}, activated, &.{});
        input = activated;
    }
    try node(gpa, &graph, "GlobalAveragePool", &.{input}, "pooled", &.{});
    try node(gpa, &graph, "Flatten", &.{"pooled"}, "flattened", &.{});
    input = "flattened";
    for (model.head, [_]usize{ 2, 4 }, 0..) |linear, index, i| {
        const weight = try std.fmt.allocPrint(gpa, "head.{d}.weight", .{index});
        const bias = try std.fmt.allocPrint(gpa, "head.{d}.bias", .{index});
        try tensor(gpa, &graph, weight, &.{ linear.output, linear.input }, linear.weight);
        try tensor(gpa, &graph, bias, &.{linear.output}, linear.bias);
        try node(gpa, &graph, "Gemm", &.{ input, weight, bias }, if (i == 0) "hidden" else "eyes", &.{try intAttribute(gpa, "transB", 1)});
        if (i == 0) try node(gpa, &graph, "Relu", &.{"hidden"}, "hidden_relu", &.{});
        input = "hidden_relu";
    }
    try pb.putBytes(&graph, gpa, 11, try valueInfo(gpa, "crop", &.{ 3, net.size, net.size }));
    try pb.putBytes(&graph, gpa, 12, try valueInfo(gpa, "eyes", &.{4}));
    var opset: std.ArrayList(u8) = .empty;
    try pb.putInt(&opset, gpa, 2, 12);
    var out: std.ArrayList(u8) = .empty;
    try pb.putInt(&out, gpa, 1, 7);
    try pb.putBytes(&out, gpa, 2, "Zig ONNX exporter");
    try pb.putBytes(&out, gpa, 3, builtin.zig_version_string);
    try pb.putBytes(&out, gpa, 7, graph.items);
    try pb.putBytes(&out, gpa, 8, opset.items);
    return metadata.stampExport(gpa, out.items, .eye_refiner, "tools/dataset/train_eye_refiner.py, exported by export_eye_refiner.zig");
}

fn verify(gpa: Allocator, model: net.Refiner, data: []const u8, runtime: []const u8, tolerance: f64) !f64 {
    var oracle = try ort.Oracle.open(gpa, runtime, data);
    defer oracle.close();
    const count = 3 * net.size * net.size;
    const probe = try gpa.alloc(f32, 3 * count);
    var expected: [12]f32 = undefined;
    for (0..3) |batch| {
        const input = probe[batch * count ..][0..count];
        for (input, 0..) |*x, i| x.* = switch (batch) {
            0 => 0,
            1 => -127.5 / 128.0,
            else => @as(f32, @floatFromInt((i * 7919 + 104729) % 65521)) / 32760.0 - 1,
        };
        const values = try model.predict(gpa, input);
        @memcpy(expected[batch * 4 ..][0..4], &values);
    }
    const got = try oracle.run(gpa, probe, &.{ 3, 3, net.size, net.size }, "crop", "eyes");
    if (!std.mem.eql(i64, got.shape, &.{ 3, 4 }) or got.data.len != expected.len) return error.InvalidOutputShape;
    var worst: f64 = 0;
    for (expected, got.data) |a, b| {
        if (!std.math.isFinite(b)) return error.NonFinitePrediction;
        worst = @max(worst, @abs(@as(f64, a) - @as(f64, b)));
    }
    std.debug.print("worst checkpoint-vs-ONNX difference {e} (tolerance {e})\n", .{ worst, tolerance });
    if (worst > tolerance) return error.ExportDoesNotMatchCheckpoint;
    return worst;
}

pub fn main(init: std.process.Init) !void {
    const gpa = init.arena.allocator();
    const args = try init.minimal.args.toSlice(gpa);
    if (args.len < 3) {
        std.debug.print("usage: export_eye_refiner <checkpoint.pt> <out.onnx> [--tolerance 1e-5] [--ort path]\n", .{});
        return error.Usage;
    }
    var tolerance: f64 = 1e-5;
    var runtime = ort.runtimePath(init.environ_map);
    var i: usize = 3;
    while (i < args.len) : (i += 2) {
        if (i + 1 == args.len) return error.MissingOptionValue;
        if (std.mem.eql(u8, args[i], "--tolerance")) tolerance = try std.fmt.parseFloat(f64, args[i + 1]) else if (std.mem.eql(u8, args[i], "--ort")) runtime = args[i + 1] else return error.UnknownOption;
    }
    if (!std.math.isFinite(tolerance) or tolerance < 0) return error.InvalidTolerance;
    const cwd = std.Io.Dir.cwd();
    const source = try cwd.realPathFileAlloc(init.io, args[1], gpa);
    const destination = cwd.realPathFileAlloc(init.io, args[2], gpa) catch |err| switch (err) {
        error.FileNotFound => null,
        else => return err,
    };
    if (destination) |path| if (std.mem.eql(u8, source, path)) return error.OutputWouldReplaceCheckpoint;
    const checkpoint = try ck.load(gpa, init.io, source);
    const model = try net.Refiner.fromCheckpoint(gpa, checkpoint);
    std.debug.print("checkpoint: {s}\n", .{std.fs.path.basename(source)});
    for ([_][]const u8{ "val_stats", "test_stats", "stats", "epoch" }) |key| if (checkpoint.dict.get(key)) |value| {
        var text: std.Io.Writer.Allocating = .init(gpa);
        try ck.report(&text.writer, value);
        std.debug.print("{s}: {s}\n", .{ key, text.written() });
    };
    const data = try buildOnnx(gpa, model);
    _ = try verify(gpa, model, data, runtime, tolerance);
    const output_path = destination orelse args[2];
    const permissions = if (destination != null) (try cwd.statFile(init.io, output_path, .{})).permissions else std.Io.File.Permissions.default_file;
    var file = try cwd.createFileAtomic(init.io, output_path, .{ .replace = true, .permissions = permissions });
    defer file.deinit(init.io);
    var buffer: [4096]u8 = undefined;
    var writer = file.file.writer(init.io, &buffer);
    try writer.interface.writeAll(data);
    try writer.flush();
    try file.replace(init.io);
    std.debug.print("wrote {s}\nEXPORT VERIFIED\n", .{args[2]});
}

test "ONNX tensor payloads keep little-endian values and reject mismatched shapes" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const gpa = arena.allocator();
    var graph: std.ArrayList(u8) = .empty;
    try tensor(gpa, &graph, "weight", &.{2}, &.{ 1, -2 });
    const encoded = try (try pb.field(graph.items, 5)).?.bytes();
    try std.testing.expectEqualStrings("weight", try pb.string(encoded, 8));
    try std.testing.expectEqualSlices(u8, &.{ 0, 0, 128, 63, 0, 0, 0, 192 }, try (try pb.field(encoded, 9)).?.bytes());
    try std.testing.expectError(error.InvalidParameterShape, tensor(gpa, &graph, "bad", &.{3}, &.{ 1, 2 }));
    try std.testing.expectError(error.NonFiniteParameter, tensor(gpa, &graph, "bad", &.{1}, &.{std.math.nan(f32)}));
}

test "ONNX interfaces retain a symbolic batch dimension" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const value = try valueInfo(arena.allocator(), "crop", &.{ 3, 112, 112 });
    const type_proto = try (try pb.field(value, 2)).?.bytes();
    const tensor_type = try (try pb.field(type_proto, 1)).?.bytes();
    const shape = try (try pb.field(tensor_type, 2)).?.bytes();
    var reader: pb.Reader = .{ .data = shape };
    try std.testing.expectEqualStrings("batch", try pb.string(try (try reader.next()).?.bytes(), 2));
    for ([_]u64{ 3, 112, 112 }) |dim| try std.testing.expectEqual(dim, try (try pb.field(try (try reader.next()).?.bytes(), 1)).?.int());
    try std.testing.expect((try reader.next()) == null);
}
