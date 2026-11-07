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
