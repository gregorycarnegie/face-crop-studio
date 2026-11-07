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
        try pb.putTensor(gpa, &graph, weight_name, &.{ conv.output, conv.input, 3, 3 }, folded);
        try pb.putTensor(gpa, &graph, bias_name, &.{conv.output}, conv.shift);
        const convolved = try std.fmt.allocPrint(gpa, "conv_{d}", .{i});
        const activated = try std.fmt.allocPrint(gpa, "relu_{d}", .{i});
        const stride: i64 = if (i % 2 == 0) 2 else 1;
        try pb.putNode(gpa, &graph, "Conv", &.{ input, weight_name, bias_name }, convolved, &.{
            try pb.attrInts(gpa, "kernel_shape", &.{ 3, 3 }),
            try pb.attrInts(gpa, "strides", &.{ stride, stride }),
            try pb.attrInts(gpa, "pads", &.{ 1, 1, 1, 1 }),
        });
        try pb.putNode(gpa, &graph, "Relu", &.{convolved}, activated, &.{});
        input = activated;
    }
    try pb.putNode(gpa, &graph, "GlobalAveragePool", &.{input}, "pooled", &.{});
    try pb.putNode(gpa, &graph, "Flatten", &.{"pooled"}, "flattened", &.{});
    input = "flattened";
    for (model.head, [_]usize{ 2, 4 }, 0..) |linear, index, i| {
        const weight = try std.fmt.allocPrint(gpa, "head.{d}.weight", .{index});
        const bias = try std.fmt.allocPrint(gpa, "head.{d}.bias", .{index});
        try pb.putTensor(gpa, &graph, weight, &.{ linear.output, linear.input }, linear.weight);
        try pb.putTensor(gpa, &graph, bias, &.{linear.output}, linear.bias);
        try pb.putNode(gpa, &graph, "Gemm", &.{ input, weight, bias }, if (i == 0) "hidden" else "eyes", &.{try pb.attrInt(gpa, "transB", 1)});
        if (i == 0) try pb.putNode(gpa, &graph, "Relu", &.{"hidden"}, "hidden_relu", &.{});
        input = "hidden_relu";
    }
    try pb.putBytes(&graph, gpa, 11, try pb.valueInfo(gpa, "crop", "batch", &.{ 3, net.size, net.size }));
    try pb.putBytes(&graph, gpa, 12, try pb.valueInfo(gpa, "eyes", "batch", &.{4}));
    const out = try pb.model(gpa, 7, 12, "Zig ONNX exporter", builtin.zig_version_string, graph.items);
    return metadata.stampExport(gpa, out, .eye_refiner, "tools/dataset/train_eye_refiner.py, exported by export_eye_refiner.zig");
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
    try pb.putTensor(gpa, &graph, "weight", &.{2}, &.{ 1, -2 });
    const encoded = try (try pb.field(graph.items, 5)).?.bytes();
    try std.testing.expectEqualStrings("weight", try pb.string(encoded, 8));
    try std.testing.expectEqualSlices(u8, &.{ 0, 0, 128, 63, 0, 0, 0, 192 }, try (try pb.field(encoded, 9)).?.bytes());
    try std.testing.expectError(error.InvalidParameterShape, pb.putTensor(gpa, &graph, "bad", &.{3}, &.{ 1, 2 }));
    try std.testing.expectError(error.NonFiniteParameter, pb.putTensor(gpa, &graph, "bad", &.{1}, &.{std.math.nan(f32)}));
}

test "ONNX interfaces retain a symbolic batch dimension" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const value = try pb.valueInfo(arena.allocator(), "crop", "batch", &.{ 3, 112, 112 });
    const type_proto = try (try pb.field(value, 2)).?.bytes();
    const tensor_type = try (try pb.field(type_proto, 1)).?.bytes();
    const shape = try (try pb.field(tensor_type, 2)).?.bytes();
    var reader: pb.Reader = .{ .data = shape };
    try std.testing.expectEqualStrings("batch", try pb.string(try (try reader.next()).?.bytes(), 2));
    for ([_]u64{ 3, 112, 112 }) |dim| try std.testing.expectEqual(dim, try (try pb.field(try (try reader.next()).?.bytes(), 1)).?.int());
    try std.testing.expect((try reader.next()) == null);
}
