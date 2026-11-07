//! Export a trained SCRFD checkpoint to ONNX, verified before it may replace the destination.
//!
//!   zig run -O ReleaseFast -lc tools/dataset/export_scrfd.zig -- <checkpoint.pth> <out.onnx> [--size 640] [--tolerance 1e-4] [--ort path]
//!
//! The graph is written here rather than traced: `scrfd.zig` reads the architecture out of the
//! checkpoint, and this emits it as image-in, raw per-stride tensors-out -- scores, box distances
//! and keypoints for strides 8/16/32 -- with no NMS, in the deployment layout SCRFD's own ONNX
//! wrapper decodes: each (1, A*C, H, W) map as (1, H*W*A, C), class maps sigmoided.
//!
//! Every BatchNorm is folded into the convolution feeding it, and every convolution keeps its
//! checkpoint name (prefixed `detector.`). `fcs-core`'s built-in CPU and WGSL engines look weights
//! up by those names, and `fcs-core/src/scrfd/topology.rs` is generated from this graph by
//! `scrfd_topology.zig`, so both stay stable across re-exports. The folding is float32 throughout,
//! as the numpy fold of the earlier Python exporter was.
//!
//! Verified before writing: ONNX Runtime (ORT_DYLIB_PATH or `--ort`) must reproduce
//! `scrfd.zig`'s independent forward pass, which runs the checkpoint with BatchNorm unfolded. An
//! export that does not match is worse than none, because it looks like a model.
const std = @import("std");
const builtin = @import("builtin");
const ck = @import("checkpoint.zig");
const net = @import("scrfd.zig");
const pb = @import("onnx.zig");
const ort = @import("ort.zig");
const metadata = @import("stamp_model_metadata.zig");
const Allocator = std.mem.Allocator;

const produced_by = "tools/dataset/export_scrfd.zig";

fn valueName(gpa: Allocator, id: usize) ![]const u8 {
    return if (id == 0) "input.1" else std.fmt.allocPrint(gpa, "v{d}", .{id});
}

/// The weight and bias of one convolution, BatchNorm folded in: w' = w * s, b' = (b - mean) * s + beta.
fn fold(gpa: Allocator, c: net.Conv) !struct { []const f32, []const f32 } {
    const n = c.bn orelse return .{ c.weight, c.bias orelse return error.MissingConvBias };
    const weight = try gpa.alloc(f32, c.weight.len);
    const bias = try gpa.alloc(f32, c.out);
    const per = c.weight.len / c.out;
    for (0..c.out) |o| {
        const s = n.gamma[o] / @sqrt(n.variance[o] + net.eps);
        for (weight[o * per ..][0..per], c.weight[o * per ..][0..per]) |*w, original| w.* = original * s;
        const b: f32 = if (c.bias) |own| own[o] else 0;
        bias[o] = (b - n.mean[o]) * s + n.beta[o];
        if (!std.math.isFinite(s) or !std.math.isFinite(bias[o])) return error.NonFiniteParameter;
    }
    return .{ weight, bias };
}

pub fn buildOnnx(gpa: Allocator, model: net.Scrfd, size: usize) ![]const u8 {
    if (size == 0 or size % 32 != 0) return error.SizeNotMultipleOf32;
    var graph: std.ArrayList(u8) = .empty;
    try pb.putBytes(&graph, gpa, 2, "main_graph");
    try pb.putTensor(gpa, &graph, "resize_roi", &.{0}, &.{});
    try pb.putTensor(gpa, &graph, "resize_scales", &.{4}, &.{ 1, 1, 2, 2 });
    for (net.per_anchor) |per| try pb.putTensorI64(gpa, &graph, try std.fmt.allocPrint(gpa, "deploy_shape_{d}", .{per}), &.{ 1, -1, @intCast(per) });
    // A value's name once any ReLU has run: consumers read the activation, not the raw conv.
    const names = try gpa.alloc([]const u8, model.channels.len);
    names[0] = "input.1";
    var folded: usize = 0;
    for (model.steps, 1..) |step, id| {
        const out = try valueName(gpa, id);
        names[id] = out;
        switch (step) {
            .conv => |s| {
                const c = s.conv;
                const weight, const bias = try fold(gpa, c);
                if (c.bn != null) folded += 1;
                const w_name = try std.fmt.allocPrint(gpa, "detector.{s}.weight", .{c.name});
                const b_name = try std.fmt.allocPrint(gpa, "detector.{s}.bias", .{c.name});
                try pb.putTensor(gpa, &graph, w_name, &.{ c.out, c.in_per_group, c.kernel, c.kernel }, weight);
                try pb.putTensor(gpa, &graph, b_name, &.{c.out}, bias);
                const k: i64 = @intCast(c.kernel);
                const p: i64 = @intCast(c.pad());
                const st: i64 = @intCast(c.stride);
                try pb.putNode(gpa, &graph, "Conv", &.{ names[s.input], w_name, b_name }, out, &.{
                    try pb.attrInts(gpa, "dilations", &.{ 1, 1 }),
                    try pb.attrInt(gpa, "group", @intCast(c.groups)),
                    try pb.attrInts(gpa, "kernel_shape", &.{ k, k }),
                    try pb.attrInts(gpa, "pads", &.{ p, p, p, p }),
                    try pb.attrInts(gpa, "strides", &.{ st, st }),
                });
                if (c.relu) {
                    names[id] = try std.fmt.allocPrint(gpa, "{s}_relu", .{out});
                    try pb.putNode(gpa, &graph, "Relu", &.{out}, names[id], &.{});
                }
            },
            .add => |pair| try pb.putNode(gpa, &graph, "Add", &.{ names[pair[0]], names[pair[1]] }, out, &.{}),
            .upsample => |from| try pb.putNode(gpa, &graph, "Resize", &.{ names[from], "resize_roi", "resize_scales" }, out, &.{
                try pb.attrString(gpa, "coordinate_transformation_mode", "asymmetric"),
                try pb.attrString(gpa, "mode", "nearest"),
                try pb.attrString(gpa, "nearest_mode", "floor"),
            }),
        }
        // After a stride's last map, its three deployment outputs, as torch's export placed them.
        for (model.heads, 0..) |head, level| if (head[2] == id) {
            for (head, net.per_anchor, 0..) |v, per, kind| {
                const output = try std.fmt.allocPrint(gpa, "out_{d}", .{kind * 3 + level});
                const nhwc = try std.fmt.allocPrint(gpa, "{s}_nhwc", .{output});
                const flat = if (kind == 0) try std.fmt.allocPrint(gpa, "{s}_logits", .{output}) else output;
                try pb.putNode(gpa, &graph, "Transpose", &.{names[v]}, nhwc, &.{try pb.attrInts(gpa, "perm", &.{ 0, 2, 3, 1 })});
                try pb.putNode(gpa, &graph, "Reshape", &.{ nhwc, try std.fmt.allocPrint(gpa, "deploy_shape_{d}", .{per}) }, flat, &.{});
                if (kind == 0) try pb.putNode(gpa, &graph, "Sigmoid", &.{flat}, output, &.{});
            }
        };
    }
    try pb.putBytes(&graph, gpa, 11, try pb.valueInfo(gpa, "input.1", null, &.{ 1, 3, size, size }));
    for (0..3) |kind| for (net.strides, 0..) |s, level| {
        const n = (size / s) * (size / s) * net.anchors;
        try pb.putBytes(&graph, gpa, 12, try pb.valueInfo(gpa, try std.fmt.allocPrint(gpa, "out_{d}", .{kind * 3 + level}), null, &.{ 1, n, net.per_anchor[kind] }));
    };
    std.debug.print("folded {d} BatchNorm layers into their convolutions, keeping their names\n", .{folded});
    const model_bytes = try pb.model(gpa, 6, 11, "Zig ONNX exporter", builtin.zig_version_string, graph.items);
    return metadata.stampExport(gpa, model_bytes, .scrfd, produced_by);
}

/// A deterministic probe in [-1, 1), like the normalised images the detector sees.
fn probe(gpa: Allocator, size: usize) ![]f32 {
    const input = try gpa.alloc(f32, 3 * size * size);
    var rng: std.Random.DefaultPrng = .init(20260920);
    for (input) |*x| x.* = rng.random().float(f32) * 2 - 1;
    return input;
}

/// Worst absolute difference between ONNX Runtime and the unfolded checkpoint forward.
pub fn verify(gpa: Allocator, model: net.Scrfd, data: []const u8, runtime: []const u8, size: usize) !f64 {
    var oracle = try ort.Oracle.open(gpa, runtime, data);
    defer oracle.close();
    const input = try probe(gpa, size);
    const expected = try model.forward(gpa, input, size);
    var names: [9][:0]const u8 = undefined;
    for (&names, 0..) |*n, i| n.* = try std.fmt.allocPrintSentinel(gpa, "out_{d}", .{i}, 0);
    const got = try oracle.runAll(gpa, input, &.{ 1, 3, @intCast(size), @intCast(size) }, "input.1", &names);
    var worst: f64 = 0;
    for (expected, got, 0..) |want, have, i| {
        const per = net.per_anchor[i / 3];
        if (have.shape.len != 3 or have.shape[0] != 1 or have.shape[2] != per or have.data.len != want.len) return error.InvalidOutputShape;
        for (want, have.data) |a, b| {
            if (!std.math.isFinite(b)) return error.NonFinitePrediction;
            worst = @max(worst, @abs(@as(f64, a) - b));
        }
    }
    return worst;
}

pub fn main(init: std.process.Init) !void {
    const gpa = init.arena.allocator();
    const args = try init.minimal.args.toSlice(gpa);
    var positional: std.ArrayList([]const u8) = .empty;
    var size: usize = 640;
    var tolerance: f64 = 1e-4;
    var runtime = ort.runtimePath(init.environ_map);
    var i: usize = 1;
    while (i < args.len) : (i += 1) {
        const arg = args[i];
        if (std.mem.startsWith(u8, arg, "--")) {
            if (i + 1 == args.len) return error.MissingOptionValue;
            i += 1;
            if (std.mem.eql(u8, arg, "--size")) size = try std.fmt.parseInt(usize, args[i], 10) else if (std.mem.eql(u8, arg, "--tolerance")) tolerance = try std.fmt.parseFloat(f64, args[i]) else if (std.mem.eql(u8, arg, "--ort")) runtime = args[i] else return error.UnknownOption;
        } else try positional.append(gpa, arg);
    }
    if (positional.items.len != 2) {
        std.debug.print("usage: export_scrfd <checkpoint.pth> <out.onnx> [--size 640] [--tolerance 1e-4] [--ort path]\n", .{});
        return error.Usage;
    }
    if (!std.math.isFinite(tolerance) or tolerance < 0) return error.InvalidTolerance;
    const cwd = std.Io.Dir.cwd();
    const source = try cwd.realPathFileAlloc(init.io, positional.items[0], gpa);
    const destination = cwd.realPathFileAlloc(init.io, positional.items[1], gpa) catch |err| switch (err) {
        error.FileNotFound => null,
        else => return err,
    };
    if (destination) |path| if (std.mem.eql(u8, source, path)) return error.OutputWouldReplaceCheckpoint;

    const model = try net.Scrfd.fromCheckpoint(gpa, try ck.load(gpa, init.io, source));
    std.debug.print("checkpoint: {s}, {d} steps\n", .{ std.fs.path.basename(source), model.steps.len });
    const data = try buildOnnx(gpa, model, size);
    const worst = try verify(gpa, model, data, runtime, size);
    std.debug.print("worst checkpoint-vs-ONNX difference {e} (tolerance {e})\n", .{ worst, tolerance });
    if (worst > tolerance) return error.ExportDoesNotMatchCheckpoint;

    const output_path = destination orelse positional.items[1];
    const permissions = if (destination != null) (try cwd.statFile(init.io, output_path, .{})).permissions else std.Io.File.Permissions.default_file;
    var file = try cwd.createFileAtomic(init.io, output_path, .{ .replace = true, .permissions = permissions });
    defer file.deinit(init.io);
    var buffer: [4096]u8 = undefined;
    var writer = file.file.writer(init.io, &buffer);
    try writer.interface.writeAll(data);
    try writer.flush();
    try file.replace(init.io);
    std.debug.print("wrote {s}\nEXPORT VERIFIED\n", .{positional.items[1]});
}

test "folding matches the unfolded convolution and keeps a signed-zero-free bias" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const c: net.Conv = .{ .name = "x", .out = 2, .in_per_group = 1, .kernel = 1, .stride = 1, .groups = 1, .weight = &.{ 2, 3 }, .bias = null, .bn = .{ .gamma = &.{ 1, 2 }, .beta = &.{ 0.5, 0 }, .mean = &.{ 1, 0 }, .variance = &.{ 1 - net.eps, 4 - net.eps } }, .relu = false };
    const weight, const bias = try fold(arena.allocator(), c);
    for ([_]f32{ 2, 3 }, weight) |want, got| try std.testing.expectApproxEqAbs(want, got, 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, -0.5), bias[0], 1e-6);
    try std.testing.expectEqual(@as(f32, 0), bias[1]);
    try std.testing.expect(!std.math.signbit(bias[1])); // (0 - 0) * s + 0 is +0, as numpy's fold gave
}

test "the synthetic checkpoint exports the graph the Rust engine's committed plan was generated from" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const gpa = arena.allocator();
    const model = try net.Scrfd.fromCheckpoint(gpa, try ck.load(gpa, std.testing.io, "tools/dataset/testdata/scrfd.pt"));
    const generated = try @import("scrfd_topology.zig").generate(gpa, try buildOnnx(gpa, model, 640));
    const committed = try std.Io.Dir.cwd().readFileAlloc(std.testing.io, "fcs-core/src/scrfd/topology.rs", gpa, .unlimited);
    // The committed file went through rustfmt; compare the tokens, not the layout.
    try std.testing.expectEqualStrings(try tokens(gpa, committed), try tokens(gpa, generated));
}

fn tokens(gpa: Allocator, text: []const u8) ![]const u8 {
    var out: std.ArrayList(u8) = .empty;
    for (text) |c| if (!std.ascii.isWhitespace(c)) try out.append(gpa, c);
    var kept: usize = 0;
    for (out.items, 0..) |c, i| {
        if (c == ',' and i + 1 < out.items.len and (out.items[i + 1] == '}' or out.items[i + 1] == ']')) continue;
        out.items[kept] = c;
        kept += 1;
    }
    return out.items[0..kept];
}
