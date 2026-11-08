//! Compile the exported SCRFD graph into the Rust executor's step table.
//!   zig run tools/dataset/scrfd_topology.zig -- <model.onnx> <out.rs>
const std = @import("std");
const pb = @import("onnx.zig");
const u = @import("util.zig");
const Allocator = std.mem.Allocator;

const Node = struct {
    op: []const u8,
    inputs: []const []const u8,
    outputs: []const []const u8,
    attrs: std.StringHashMapUnmanaged([]const i64),

    fn parse(gpa: Allocator, data: []const u8) !Node {
        var inputs: std.ArrayList([]const u8) = .empty;
        var outputs: std.ArrayList([]const u8) = .empty;
        var attrs: std.StringHashMapUnmanaged([]const i64) = .empty;
        var r: pb.Reader = .{ .data = data };
        while (try r.next()) |f| switch (f.tag) {
            1 => try inputs.append(gpa, try f.bytes()),
            2 => try outputs.append(gpa, try f.bytes()),
            5 => {
                const a = try f.bytes();
                const list = try pb.ints(gpa, a, 8);
                const value = if (list.len > 0) list else try pb.ints(gpa, a, 3);
                try attrs.put(gpa, try pb.string(a, 1), value);
            },
            else => {},
        };
        return .{ .op = try pb.string(data, 4), .inputs = try inputs.toOwnedSlice(gpa), .outputs = try outputs.toOwnedSlice(gpa), .attrs = attrs };
    }

    fn attr(self: Node, name: []const u8, default: i64) i64 {
        const values = self.attrs.get(name) orelse return default;
        return if (values.len > 0) values[0] else default;
    }
};

fn skipped(op: []const u8) bool {
    for ([_][]const u8{ "Shape", "Constant", "Gather", "Unsqueeze", "Concat", "Slice", "Cast", "Relu", "Sigmoid", "Transpose", "Reshape", "Identity" }) |name| {
        if (std.mem.eql(u8, name, op)) return true;
    }
    return false;
}

fn slotOf(name: []const u8, slots: std.StringHashMapUnmanaged(usize), producers: std.StringHashMapUnmanaged(Node)) !usize {
    var current = name;
    // A malformed graph must fail rather than cycle through aliases forever.
    for (0..producers.count() + 1) |_| {
        if (slots.get(current)) |slot| return slot;
        const node = producers.get(current) orelse return error.MissingProducer;
        if (!skipped(node.op) or node.inputs.len == 0) return error.UnsupportedProducer;
        current = node.inputs[0];
    }
    return error.CyclicGraph;
}

fn rustString(w: *std.Io.Writer, text: []const u8) !void {
    try w.writeByte('"');
    for (text) |c| switch (c) {
        '"', '\\' => {
            try w.writeByte('\\');
            try w.writeByte(c);
        },
        0...31, 127 => try w.print("\\x{x:0>2}", .{c}),
        else => try w.writeByte(c),
    };
    try w.writeByte('"');
}

pub fn generate(gpa: Allocator, model: []const u8) ![]const u8 {
    const graph = try (try pb.field(model, 7) orelse return error.MissingGraph).bytes();
    var nodes: std.ArrayList(Node) = .empty;
    var outputs: std.ArrayList([]const u8) = .empty;
    var shapes: std.StringHashMapUnmanaged([]const i64) = .empty;
    var producers: std.StringHashMapUnmanaged(Node) = .empty;
    var relus: std.StringHashMapUnmanaged(void) = .empty;
    var r: pb.Reader = .{ .data = graph };
    while (try r.next()) |f| switch (f.tag) {
        1 => {
            const node = try Node.parse(gpa, try f.bytes());
            if (node.outputs.len == 0) return error.MissingNodeOutput;
            try nodes.append(gpa, node);
            for (node.outputs) |name| {
                if (producers.contains(name)) return error.DuplicateProducer;
                try producers.put(gpa, name, node);
            }
            if (std.mem.eql(u8, node.op, "Relu")) {
                if (node.inputs.len != 1) return error.InvalidRelu;
                try relus.put(gpa, node.inputs[0], {});
            }
        },
        5 => {
            const tensor = try f.bytes();
            try shapes.put(gpa, try pb.string(tensor, 8), try pb.ints(gpa, tensor, 1));
        },
        12 => try outputs.append(gpa, try pb.string(try f.bytes(), 1)),
        else => {},
    };
    if (outputs.items.len == 0) return error.MissingGraphOutputs;

    var slots: std.StringHashMapUnmanaged(usize) = .empty;
    try slots.put(gpa, "input.1", 0);
    var body: std.Io.Writer.Allocating = .init(gpa);
    const w = &body.writer;
    var steps: usize = 0;
    for (nodes.items) |node| {
        if (skipped(node.op)) continue;
        if (node.inputs.len == 0 or node.outputs.len != 1) return error.InvalidNode;
        const input = try slotOf(node.inputs[0], slots, producers);
        if (std.mem.eql(u8, node.op, "Conv")) {
            if (node.inputs.len < 2 or node.inputs.len > 3) return error.InvalidConv;
            const weight = node.inputs[1];
            const shape = shapes.get(weight) orelse return error.MissingWeight;
            if (shape.len != 4) return error.InvalidConvWeight;
            for (shape) |d| if (d <= 0) return error.InvalidConvWeight;
            try w.print("    Step::Conv {{ input: {d}, weight: ", .{input});
            try rustString(w, weight);
            try w.writeAll(", bias: ");
            try rustString(w, if (node.inputs.len > 2) node.inputs[2] else "");
            try w.print(", out_channels: {d}, in_per_group: {d}, kernel: {d}, stride: {d}, pad: {d}, groups: {d}, relu: {} }},\n", .{
                shape[0], shape[1], node.attr("kernel_shape", 1), node.attr("strides", 1), node.attr("pads", 0), node.attr("group", 1), relus.contains(node.outputs[0]),
            });
        } else if (std.mem.eql(u8, node.op, "Add")) {
            if (node.inputs.len != 2) return error.InvalidAdd;
            try w.print("    Step::Add {{ a: {d}, b: {d} }},\n", .{ input, try slotOf(node.inputs[1], slots, producers) });
        } else if (std.mem.eql(u8, node.op, "Resize")) {
            // shortcut: this model uses nearest 2x Resize; validate scales before supporting other graphs.
            try w.print("    Step::Upsample2x {{ input: {d} }},\n", .{input});
        } else {
            std.debug.print("unhandled op {s}; the generator needs a step kind for it\n", .{node.op});
            return error.UnsupportedOperator;
        }
        steps += 1;
        try slots.put(gpa, node.outputs[0], steps);
    }
    var out: std.Io.Writer.Allocating = .init(gpa);
    try out.writer.writeAll(
        "//! SCRFD-500M's topology, generated by `tools/dataset/scrfd_topology.zig`.\n" ++
            "//!\n" ++
            "//! Do not edit: re-run the generator against the exported model instead. Every step\n" ++
            "//! writes one slot and refers to earlier slots by index, slot 0 being the input.\n" ++
            "//! Weights are looked up by name, which is why the exporter folds BatchNorm itself\n" ++
            "//! rather than letting torch's constant folding rename everything.\n\n" ++
            "use super::plan::Step;\n\n" ++
            "/// The nine head outputs, as `(1, anchors*channels, H, W)` maps: class, box and\n" ++
            "/// keypoint convolutions for strides 8, 16 and 32.\n",
    );
    try out.writer.print("pub const OUTPUTS: [usize; {d}] = [", .{outputs.items.len});
    for (outputs.items, 0..) |name, i| {
        if (i > 0) try out.writer.writeAll(", ");
        try out.writer.print("{d}", .{try slotOf(name, slots, producers)});
    }
    try out.writer.print("];\n\n/// {d} steps in execution order.\npub const STEPS: [Step; {d}] = [\n{s}];\n", .{ steps, steps, body.written() });
    return out.toOwnedSlice();
}

pub fn main(init: std.process.Init) !void {
    const gpa = init.arena.allocator();
    const args = try init.minimal.args.toSlice(gpa);
    if (args.len != 3) {
        std.debug.print("usage: scrfd_topology <model.onnx> <out.rs>\n", .{});
        return error.Usage;
    }
    const text = try generate(gpa, try u.readFile(gpa, init.io, args[1]));
    try u.writeFile(init.io, args[2], text);
    std.debug.print("wrote {s}\n", .{args[2]});
}

test "aliases resolve to one-based executor slots and cycles fail" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const gpa = arena.allocator();
    var producers: std.StringHashMapUnmanaged(Node) = .empty;
    try producers.put(gpa, "relu", .{ .op = "Relu", .inputs = &.{"conv"}, .outputs = &.{"relu"}, .attrs = .empty });
    var slots: std.StringHashMapUnmanaged(usize) = .empty;
    try slots.put(gpa, "input.1", 0);
    try slots.put(gpa, "conv", 1);
    try std.testing.expectEqual(@as(usize, 1), try slotOf("relu", slots, producers));
    try producers.put(gpa, "loop", .{ .op = "Identity", .inputs = &.{"loop"}, .outputs = &.{"loop"}, .attrs = .empty });
    try std.testing.expectError(error.CyclicGraph, slotOf("loop", slots, producers));
    try std.testing.expectError(error.MissingProducer, slotOf("absent", slots, producers));
}

test "generation follows deployment aliases back to the raw convolution" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const gpa = arena.allocator();
    var conv: std.ArrayList(u8) = .empty;
    try pb.putBytes(&conv, gpa, 1, "input.1");
    try pb.putBytes(&conv, gpa, 1, "weight");
    try pb.putBytes(&conv, gpa, 2, "conv");
    try pb.putBytes(&conv, gpa, 4, "Conv");
    var relu: std.ArrayList(u8) = .empty;
    try pb.putBytes(&relu, gpa, 1, "conv");
    try pb.putBytes(&relu, gpa, 2, "output");
    try pb.putBytes(&relu, gpa, 4, "Relu");
    var weight: std.ArrayList(u8) = .empty;
    for ([_]u64{ 16, 3, 1, 1 }) |dim| try pb.putInt(&weight, gpa, 1, dim);
    try pb.putBytes(&weight, gpa, 8, "weight");
    var output: std.ArrayList(u8) = .empty;
    try pb.putBytes(&output, gpa, 1, "output");
    var graph: std.ArrayList(u8) = .empty;
    try pb.putBytes(&graph, gpa, 1, conv.items);
    try pb.putBytes(&graph, gpa, 1, relu.items);
    try pb.putBytes(&graph, gpa, 5, weight.items);
    try pb.putBytes(&graph, gpa, 12, output.items);
    var model: std.ArrayList(u8) = .empty;
    try pb.putBytes(&model, gpa, 7, graph.items);
    const generated = try generate(gpa, model.items);
    try std.testing.expect(std.mem.indexOf(u8, generated, "pub const OUTPUTS: [usize; 1] = [1];") != null);
    try std.testing.expect(std.mem.indexOf(u8, generated, "Step::Conv { input: 0, weight: \"weight\", bias: \"\", out_channels: 16, in_per_group: 3, kernel: 1, stride: 1, pad: 0, groups: 1, relu: true }") != null);
    try std.testing.expectError(error.MissingGraph, generate(gpa, ""));
}
