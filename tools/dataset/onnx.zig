//! The ONNX protobuf fields needed by the dataset tools; tensor payloads stay borrowed.
//! Tags follow https://github.com/onnx/onnx/blob/main/onnx/onnx.proto.
const std = @import("std");
const Allocator = std.mem.Allocator;

pub const Error = error{InvalidProtobuf};

pub fn varint(data: []const u8, pos: *usize) Error!u64 {
    var value: u64 = 0;
    for (0..10) |i| {
        if (pos.* == data.len) return error.InvalidProtobuf;
        const b = data[pos.*];
        pos.* += 1;
        if (i == 9 and b > 1) return error.InvalidProtobuf;
        value |= @as(u64, b & 127) << @as(u6, @intCast(i * 7));
        if (b < 128) return value;
    }
    return error.InvalidProtobuf;
}

pub const Field = struct {
    tag: u32,
    wire: u3,
    data: []const u8,
    raw: []const u8,
    value: u64 = 0,

    pub fn bytes(self: Field) Error![]const u8 {
        if (self.wire != 2) return error.InvalidProtobuf;
        return self.data;
    }

    pub fn int(self: Field) Error!u64 {
        if (self.wire != 0) return error.InvalidProtobuf;
        return self.value;
    }
};

pub const Reader = struct {
    data: []const u8,
    pos: usize = 0,

    pub fn next(self: *Reader) Error!?Field {
        if (self.pos == self.data.len) return null;
        const start = self.pos;
        const key = try varint(self.data, &self.pos);
        if (key >> 3 == 0 or key >> 3 > 0x1fffffff) return error.InvalidProtobuf;
        const wire: u3 = @truncate(key);
        var value: u64 = 0;
        const len: u64 = switch (wire) {
            0 => blk: {
                value = try varint(self.data, &self.pos);
                break :blk 0;
            },
            1 => 8,
            2 => try varint(self.data, &self.pos),
            5 => 4,
            else => return error.InvalidProtobuf,
        };
        if (len > self.data.len - self.pos) return error.InvalidProtobuf;
        const data = self.data[self.pos..][0..@intCast(len)];
        self.pos += @intCast(len);
        return .{ .tag = @intCast(key >> 3), .wire = wire, .data = data, .raw = self.data[start..self.pos], .value = value };
    }
};

pub fn field(data: []const u8, tag: u32) Error!?Field {
    var r: Reader = .{ .data = data };
    var result: ?Field = null;
    while (try r.next()) |f| if (f.tag == tag) {
        result = f;
    };
    return result;
}

pub fn string(data: []const u8, tag: u32) Error![]const u8 {
    return if (try field(data, tag)) |f| try f.bytes() else "";
}

/// Repeated int64 accepts both packed and unpacked encodings.
pub fn ints(gpa: Allocator, data: []const u8, tag: u32) ![]const i64 {
    var result: std.ArrayList(i64) = .empty;
    errdefer result.deinit(gpa);
    var r: Reader = .{ .data = data };
    while (try r.next()) |f| {
        if (f.tag != tag) continue;
        if (f.wire == 0) {
            try result.append(gpa, @bitCast(f.value));
        } else {
            const values = try f.bytes();
            var pos: usize = 0;
            while (pos < values.len) try result.append(gpa, @bitCast(try varint(values, &pos)));
        }
    }
    return result.toOwnedSlice(gpa);
}

fn putVarint(out: *std.ArrayList(u8), gpa: Allocator, value: u64) !void {
    var n = value;
    while (n >= 128) : (n >>= 7) try out.append(gpa, @as(u8, @truncate(n)) | 128);
    try out.append(gpa, @intCast(n));
}

pub fn putInt(out: *std.ArrayList(u8), gpa: Allocator, tag: u32, value: u64) !void {
    try putVarint(out, gpa, @as(u64, tag) << 3);
    try putVarint(out, gpa, value);
}

pub fn putBytes(out: *std.ArrayList(u8), gpa: Allocator, tag: u32, data: []const u8) !void {
    try putVarint(out, gpa, (@as(u64, tag) << 3) | 2);
    try putVarint(out, gpa, data.len);
    try out.appendSlice(gpa, data);
}

// ── Writing graphs (the exporters) ─────────────────────────────────────────────

/// AttributeProto INTS.
pub fn attrInts(gpa: Allocator, name: []const u8, values: []const i64) ![]const u8 {
    var out: std.ArrayList(u8) = .empty;
    try putBytes(&out, gpa, 1, name);
    for (values) |v| try putInt(&out, gpa, 8, @bitCast(v));
    try putInt(&out, gpa, 20, 7);
    return out.toOwnedSlice(gpa);
}

/// AttributeProto INT.
pub fn attrInt(gpa: Allocator, name: []const u8, value: i64) ![]const u8 {
    var out: std.ArrayList(u8) = .empty;
    try putBytes(&out, gpa, 1, name);
    try putInt(&out, gpa, 3, @bitCast(value));
    try putInt(&out, gpa, 20, 2);
    return out.toOwnedSlice(gpa);
}

/// AttributeProto STRING.
pub fn attrString(gpa: Allocator, name: []const u8, value: []const u8) ![]const u8 {
    var out: std.ArrayList(u8) = .empty;
    try putBytes(&out, gpa, 1, name);
    try putBytes(&out, gpa, 4, value);
    try putInt(&out, gpa, 20, 3);
    return out.toOwnedSlice(gpa);
}

/// Append a NodeProto with one output to a GraphProto.
pub fn putNode(gpa: Allocator, graph: *std.ArrayList(u8), op: []const u8, inputs: []const []const u8, output: []const u8, attrs: []const []const u8) !void {
    var out: std.ArrayList(u8) = .empty;
    defer out.deinit(gpa);
    for (inputs) |input| try putBytes(&out, gpa, 1, input);
    try putBytes(&out, gpa, 2, output);
    try putBytes(&out, gpa, 4, op);
    for (attrs) |attr| try putBytes(&out, gpa, 5, attr);
    try putBytes(graph, gpa, 1, out.items);
}

/// Append a float32 initializer (raw little-endian data); non-finite values are refused.
pub fn putTensor(gpa: Allocator, graph: *std.ArrayList(u8), name: []const u8, shape: []const usize, values: []const f32) !void {
    var out: std.ArrayList(u8) = .empty;
    defer out.deinit(gpa);
    var count: usize = 1;
    for (shape) |dim| {
        count = std.math.mul(usize, count, dim) catch return error.InvalidParameterShape;
        try putInt(&out, gpa, 1, dim);
    }
    if (count != values.len) return error.InvalidParameterShape;
    try putInt(&out, gpa, 2, 1);
    try putBytes(&out, gpa, 8, name);
    const raw = try gpa.alloc(u8, try std.math.mul(usize, values.len, 4));
    defer gpa.free(raw);
    for (values, 0..) |v, i| {
        if (!std.math.isFinite(v)) return error.NonFiniteParameter;
        std.mem.writeInt(u32, raw[i * 4 ..][0..4], @bitCast(v), .little);
    }
    try putBytes(&out, gpa, 9, raw);
    try putBytes(graph, gpa, 5, out.items);
}

/// Append an int64 initializer, e.g. a Reshape target.
pub fn putTensorI64(gpa: Allocator, graph: *std.ArrayList(u8), name: []const u8, values: []const i64) !void {
    var out: std.ArrayList(u8) = .empty;
    defer out.deinit(gpa);
    try putInt(&out, gpa, 1, values.len);
    try putInt(&out, gpa, 2, 7);
    try putBytes(&out, gpa, 8, name);
    const raw = try gpa.alloc(u8, values.len * 8);
    defer gpa.free(raw);
    for (values, 0..) |v, i| std.mem.writeInt(i64, raw[i * 8 ..][0..8], v, .little);
    try putBytes(&out, gpa, 9, raw);
    try putBytes(graph, gpa, 5, out.items);
}

/// A float32 ValueInfoProto; with `batch`, the first dimension is that symbolic name.
pub fn valueInfo(gpa: Allocator, name: []const u8, batch: ?[]const u8, dims: []const u64) ![]const u8 {
    var shape: std.ArrayList(u8) = .empty;
    if (batch) |symbol| {
        var axis: std.ArrayList(u8) = .empty;
        defer axis.deinit(gpa);
        try putBytes(&axis, gpa, 2, symbol);
        try putBytes(&shape, gpa, 1, axis.items);
    }
    for (dims) |dim| {
        var axis: std.ArrayList(u8) = .empty;
        defer axis.deinit(gpa);
        try putInt(&axis, gpa, 1, dim);
        try putBytes(&shape, gpa, 1, axis.items);
    }
    var tensor_type: std.ArrayList(u8) = .empty;
    try putInt(&tensor_type, gpa, 1, 1);
    try putBytes(&tensor_type, gpa, 2, shape.items);
    var type_proto: std.ArrayList(u8) = .empty;
    try putBytes(&type_proto, gpa, 1, tensor_type.items);
    var out: std.ArrayList(u8) = .empty;
    try putBytes(&out, gpa, 1, name);
    try putBytes(&out, gpa, 2, type_proto.items);
    return out.toOwnedSlice(gpa);
}

/// A ModelProto around a finished graph.
pub fn model(gpa: Allocator, ir_version: u64, opset: u64, producer: []const u8, producer_version: []const u8, graph: []const u8) ![]const u8 {
    var opset_import: std.ArrayList(u8) = .empty;
    try putInt(&opset_import, gpa, 2, opset);
    var out: std.ArrayList(u8) = .empty;
    try putInt(&out, gpa, 1, ir_version);
    try putBytes(&out, gpa, 2, producer);
    try putBytes(&out, gpa, 3, producer_version);
    try putBytes(&out, gpa, 7, graph);
    try putBytes(&out, gpa, 8, opset_import.items);
    return out.toOwnedSlice(gpa);
}

test "protobuf rejects truncated and overflowing fields and preserves unknown fields" {
    var r: Reader = .{ .data = &.{ 0x0a, 3, 1 } };
    try std.testing.expectError(error.InvalidProtobuf, r.next());
    r = .{ .data = &.{0} };
    try std.testing.expectError(error.InvalidProtobuf, r.next());
    var pos: usize = 0;
    try std.testing.expectError(error.InvalidProtobuf, varint(&.{ 255, 255, 255, 255, 255, 255, 255, 255, 255, 255 }, &pos));
    var out: std.ArrayList(u8) = .empty;
    defer out.deinit(std.testing.allocator);
    try putInt(&out, std.testing.allocator, 999, 1234567);
    try putBytes(&out, std.testing.allocator, 7, "graph");
    r = .{ .data = out.items };
    try std.testing.expectEqual(@as(u64, 1234567), try (try r.next()).?.int());
    try std.testing.expectEqualStrings("graph", try string(out.items, 7));
    try std.testing.expectEqualSlices(u8, out.items[0..5], (try field(out.items, 999)).?.raw);
}

test "packed and unpacked tensor dimensions" {
    const values = try ints(std.testing.allocator, &.{ 8, 2, 10, 2, 3, 4, 8, 5 }, 1);
    defer std.testing.allocator.free(values);
    try std.testing.expectEqualSlices(i64, &.{ 2, 3, 4, 5 }, values);
}
