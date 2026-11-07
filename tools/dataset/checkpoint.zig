//! Read this project's PyTorch ZIP checkpoints without importing or executing Python.
//!   zig run -O ReleaseFast tools/dataset/checkpoint.zig -- <checkpoint.pt|checkpoint.pth>
//! The JSON report includes every tensor's logical-byte hash and all checkpoint metadata.
//! Allocate with an arena: values, storage payloads and dictionary aliases share its lifetime.
const std = @import("std");
const u = @import("util.zig");
const Allocator = std.mem.Allocator;
const Error = error{ InvalidPickle, UnsupportedPickleProtocol, UnsupportedPickleOpcode, UnsupportedGlobal, UnsupportedReduction, UnsupportedStorage, InvalidTensor, InvalidDictionaryKey, MetadataTooDeep };

pub const Dtype = enum {
    float32,
    int64,

    pub fn size(self: Dtype) usize {
        return switch (self) {
            .float32 => 4,
            .int64 => 8,
        };
    }
};

pub const Storage = struct {
    key: []const u8,
    dtype: Dtype,
    location: []const u8,
    count: usize,
    bytes: []const u8 = &.{},
};

fn elementCount(shape: []const usize) Error!usize {
    if (std.mem.findScalar(usize, shape, 0) != null) return 0;
    var count: usize = 1;
    for (shape) |dim| count = std.math.mul(usize, count, dim) catch return error.InvalidTensor;
    return count;
}

pub const Tensor = struct {
    storage: *Storage,
    offset: usize,
    shape: []const usize,
    strides: []const usize,
    requires_grad: bool,
    elements: usize,

    pub fn validate(self: Tensor) Error!void {
        if (self.shape.len != self.strides.len or self.elements != try elementCount(self.shape)) return error.InvalidTensor;
        if (self.elements == 0) {
            if (self.offset > self.storage.count) return error.InvalidTensor;
            return;
        }
        var last = self.offset;
        for (self.shape, self.strides) |dim, stride| {
            const delta = std.math.mul(usize, dim - 1, stride) catch return error.InvalidTensor;
            last = std.math.add(usize, last, delta) catch return error.InvalidTensor;
        }
        if (last >= self.storage.count) return error.InvalidTensor;
    }

    fn element(self: Tensor, flat: usize) []const u8 {
        std.debug.assert(flat < self.elements);
        var index = flat;
        var offset = self.offset;
        var axis = self.shape.len;
        while (axis > 0) {
            axis -= 1;
            offset += (index % self.shape[axis]) * self.strides[axis];
            index /= self.shape[axis];
        }
        const size = self.storage.dtype.size();
        return self.storage.bytes[offset * size ..][0..size];
    }

    pub fn floats(self: Tensor, gpa: Allocator) ![]f32 {
        if (self.storage.dtype != .float32) return error.ExpectedFloatTensor;
        const values = try gpa.alloc(f32, self.elements);
        for (values, 0..) |*v, i| v.* = @bitCast(std.mem.readInt(u32, self.element(i)[0..4], .little));
        return values;
    }

    pub fn sha256(self: Tensor) [32]u8 {
        var hash: std.crypto.hash.sha2.Sha256 = .init(.{});
        for (0..self.elements) |i| hash.update(self.element(i));
        return hash.finalResult();
    }
};

const Global = enum { ordered_dict, rebuild_tensor_v2, float_storage, long_storage };
const Sequence = std.ArrayList(Value);
pub const Entry = struct { key: Value, value: Value };
pub const Dict = struct {
    items: std.ArrayList(Entry) = .empty,
    attributes: ?*Value = null,

    pub fn get(self: Dict, key: []const u8) ?Value {
        for (self.items.items) |e| if (e.key == .string and std.mem.eql(u8, e.key.string, key)) return e.value;
        return null;
    }

    fn put(self: *Dict, gpa: Allocator, key: Value, value: Value) !void {
        if (key != .string and key != .integer) return error.InvalidDictionaryKey;
        for (self.items.items) |*e| {
            const equal = if (key == .string) e.key == .string and std.mem.eql(u8, e.key.string, key.string) else e.key == .integer and e.key.integer == key.integer;
            if (equal) {
                e.value = value;
                return;
            }
        }
        try self.items.append(gpa, .{ .key = key, .value = value });
    }
};

pub const Value = union(enum) {
    none,
    boolean: bool,
    integer: i64,
    float: f64,
    string: []const u8,
    tuple: *Sequence,
    list: *Sequence,
    dict: *Dict,
    tensor: *Tensor,
    storage: *Storage,
    global: Global,
    mark,
};

const Pickle = struct {
    gpa: Allocator,
    data: []const u8,
    pos: usize = 0,
    stack: std.ArrayList(Value) = .empty,
    memo: std.AutoHashMapUnmanaged(u32, Value) = .empty,
    storages: std.StringHashMapUnmanaged(*Storage) = .empty,

    fn take(self: *Pickle, n: usize) Error![]const u8 {
        if (n > self.data.len - self.pos) return error.InvalidPickle;
        const result = self.data[self.pos..][0..n];
        self.pos += n;
        return result;
    }

    fn int(self: *Pickle, comptime T: type) Error!T {
        return std.mem.readInt(T, (try self.take(@sizeOf(T)))[0..@sizeOf(T)], .little);
    }

    fn line(self: *Pickle) Error![]const u8 {
        const end = std.mem.findScalar(u8, self.data[self.pos..], '\n') orelse return error.InvalidPickle;
        const result = try self.take(end);
        _ = try self.take(1);
        return result;
    }

    fn push(self: *Pickle, value: Value) !void {
        try self.stack.append(self.gpa, value);
    }
    fn pop(self: *Pickle) Error!Value {
        return self.stack.pop() orelse error.InvalidPickle;
    }
    fn top(self: *Pickle) Error!Value {
        return if (self.stack.items.len > 0) self.stack.items[self.stack.items.len - 1] else error.InvalidPickle;
    }

    fn marked(self: *Pickle) Error![]Value {
        var i = self.stack.items.len;
        while (i > 0) {
            i -= 1;
            if (self.stack.items[i] == .mark) {
                const values = self.stack.items[i + 1 ..];
                self.stack.items.len = i;
                return values;
            }
        }
        return error.InvalidPickle;
    }

    fn sequence(self: *Pickle, values: []const Value) !*Sequence {
        const seq = try self.gpa.create(Sequence);
        seq.* = .empty;
        try seq.appendSlice(self.gpa, values);
        return seq;
    }

    fn dictionary(self: *Pickle) !*Dict {
        const dict = try self.gpa.create(Dict);
        dict.* = .{};
        return dict;
    }

    fn natural(value: Value) Error!usize {
        if (value != .integer) return error.InvalidTensor;
        return std.math.cast(usize, value.integer) orelse error.InvalidTensor;
    }

    fn dims(self: *Pickle, value: Value) ![]const usize {
        if (value != .tuple or value.tuple.items.len > 32) return error.InvalidTensor;
        const result = try self.gpa.alloc(usize, value.tuple.items.len);
        for (result, value.tuple.items) |*dim, v| dim.* = try natural(v);
        return result;
    }

    fn persistent(self: *Pickle, value: Value) !Value {
        if (value != .tuple or value.tuple.items.len != 5) return error.UnsupportedStorage;
        const args = value.tuple.items;
        if (args[0] != .string or !std.mem.eql(u8, args[0].string, "storage") or args[1] != .global or args[2] != .string or args[3] != .string) return error.UnsupportedStorage;
        const dtype: Dtype = switch (args[1].global) {
            .float_storage => .float32,
            .long_storage => .int64,
            else => return error.UnsupportedStorage,
        };
        const key = args[2].string;
        if (key.len == 0) return error.UnsupportedStorage;
        for (key) |c| if (!std.ascii.isAlphanumeric(c) and c != '_' and c != '-') return error.UnsupportedStorage;
        const count = try natural(args[4]);
        _ = std.math.mul(usize, count, dtype.size()) catch return error.InvalidTensor;
        if (self.storages.get(key)) |storage| {
            if (storage.dtype != dtype or storage.count != count or !std.mem.eql(u8, storage.location, args[3].string)) return error.UnsupportedStorage;
            return .{ .storage = storage };
        }
        const storage = try self.gpa.create(Storage);
        storage.* = .{ .key = key, .dtype = dtype, .count = count, .location = args[3].string };
        try self.storages.put(self.gpa, key, storage);
        return .{ .storage = storage };
    }

    fn reduce(self: *Pickle, callable: Value, value: Value) !Value {
        if (callable != .global or value != .tuple) return error.UnsupportedReduction;
        const args = value.tuple.items;
        switch (callable.global) {
            .ordered_dict => {
                if (args.len != 0) return error.UnsupportedReduction;
                return .{ .dict = try self.dictionary() };
            },
            .rebuild_tensor_v2 => {
                if (args.len != 6 or args[0] != .storage or args[4] != .boolean or (args[5] != .none and (args[5] != .dict or args[5].dict.items.items.len != 0))) return error.UnsupportedReduction;
                const shape = try self.dims(args[2]);
                const tensor = try self.gpa.create(Tensor);
                tensor.* = .{ .storage = args[0].storage, .offset = try natural(args[1]), .shape = shape, .strides = try self.dims(args[3]), .requires_grad = args[4].boolean, .elements = try elementCount(shape) };
                try tensor.validate();
                return .{ .tensor = tensor };
            },
            else => return error.UnsupportedReduction,
        }
    }

    fn global(self: *Pickle) !Global {
        const module = try self.line();
        const name = try self.line();
        if (std.mem.eql(u8, module, "collections") and std.mem.eql(u8, name, "OrderedDict")) return .ordered_dict;
        if (std.mem.eql(u8, module, "torch._utils") and std.mem.eql(u8, name, "_rebuild_tensor_v2")) return .rebuild_tensor_v2;
        if (std.mem.eql(u8, module, "torch")) {
            if (std.mem.eql(u8, name, "FloatStorage")) return .float_storage;
            if (std.mem.eql(u8, name, "LongStorage")) return .long_storage;
        }
        return error.UnsupportedGlobal;
    }

    fn parse(self: *Pickle) !Value {
        if (!std.mem.eql(u8, try self.take(2), &.{ 0x80, 2 })) return error.UnsupportedPickleProtocol;
        while (self.pos < self.data.len) {
            const op = (try self.take(1))[0];
            switch (op) {
                '.' => {
                    if (self.stack.items.len != 1 or self.pos != self.data.len) return error.InvalidPickle;
                    return self.pop();
                },
                '(' => try self.push(.mark),
                'N' => try self.push(.none),
                0x88, 0x89 => try self.push(.{ .boolean = op == 0x88 }),
                'K' => try self.push(.{ .integer = try self.int(u8) }),
                'M' => try self.push(.{ .integer = try self.int(u16) }),
                'J' => try self.push(.{ .integer = try self.int(i32) }),
                0x8a => {
                    const bytes = try self.take(try self.int(u8));
                    if (bytes.len > 8) return error.InvalidPickle;
                    var bits: u64 = 0;
                    for (bytes, 0..) |b, i| bits |= @as(u64, b) << @as(u6, @intCast(i * 8));
                    if (bytes.len > 0 and bytes.len < 8 and bytes[bytes.len - 1] & 128 != 0) bits |= @as(u64, std.math.maxInt(u64)) << @as(u6, @intCast(bytes.len * 8));
                    try self.push(.{ .integer = @bitCast(bits) });
                },
                'G' => try self.push(.{ .float = @bitCast(std.mem.readInt(u64, (try self.take(8))[0..8], .big)) }),
                'X' => {
                    const text = try self.take(try self.int(u32));
                    if (!std.unicode.utf8ValidateSlice(text)) return error.InvalidPickle;
                    try self.push(.{ .string = text });
                },
                'c' => try self.push(.{ .global = try self.global() }),
                ')' => try self.push(.{ .tuple = try self.sequence(&.{}) }),
                ']' => try self.push(.{ .list = try self.sequence(&.{}) }),
                '}' => try self.push(.{ .dict = try self.dictionary() }),
                't' => {
                    const values = try self.marked();
                    const seq = try self.sequence(values);
                    try self.push(.{ .tuple = seq });
                },
                0x85...0x87 => {
                    const count = op - 0x84;
                    if (self.stack.items.len < count) return error.InvalidPickle;
                    const start = self.stack.items.len - count;
                    const seq = try self.sequence(self.stack.items[start..]);
                    self.stack.items.len = start;
                    try self.push(.{ .tuple = seq });
                },
                's' => {
                    const value = try self.pop();
                    const key = try self.pop();
                    const dict = try self.top();
                    if (dict != .dict) return error.InvalidPickle;
                    try dict.dict.put(self.gpa, key, value);
                },
                'u' => {
                    const pairs = try self.marked();
                    const dict = try self.top();
                    if (dict != .dict or pairs.len % 2 != 0) return error.InvalidPickle;
                    var i: usize = 0;
                    while (i < pairs.len) : (i += 2) try dict.dict.put(self.gpa, pairs[i], pairs[i + 1]);
                },
                'a' => {
                    const value = try self.pop();
                    const list = try self.top();
                    if (list != .list) return error.InvalidPickle;
                    try list.list.append(self.gpa, value);
                },
                'e' => {
                    const values = try self.marked();
                    const list = try self.top();
                    if (list != .list) return error.InvalidPickle;
                    try list.list.appendSlice(self.gpa, values);
                },
                'q', 'r' => {
                    const index: u32 = if (op == 'q') try self.int(u8) else try self.int(u32);
                    const value = try self.top();
                    if (value == .mark) return error.InvalidPickle;
                    try self.memo.put(self.gpa, index, value);
                },
                'h', 'j' => {
                    const index: u32 = if (op == 'h') try self.int(u8) else try self.int(u32);
                    try self.push(self.memo.get(index) orelse return error.InvalidPickle);
                },
                'Q' => {
                    const value = try self.pop();
                    try self.push(try self.persistent(value));
                },
                'R' => {
                    const args = try self.pop();
                    const callable = try self.pop();
                    try self.push(try self.reduce(callable, args));
                },
                'b' => {
                    const attributes = try self.pop();
                    const instance = try self.top();
                    if (instance != .dict or attributes != .dict) return error.UnsupportedReduction;
                    const state = try self.gpa.create(Value);
                    state.* = attributes;
                    instance.dict.attributes = state;
                },
                else => return error.UnsupportedPickleOpcode,
            }
        }
        return error.InvalidPickle;
    }
};

fn extract(gpa: Allocator, reader: *std.Io.File.Reader, entry: std.zip.Iterator.Entry, archive_size: u64) ![]const u8 {
    // shortcut: torch.save uses stored ZIP members; add bounded decompression for repacked archives.
    if (entry.compression_method != .store) return error.UnsupportedCheckpointCompression;
    if (entry.uncompressed_size != entry.compressed_size or entry.uncompressed_size > archive_size) return error.InvalidCheckpointMember;
    var out: std.Io.Writer.Allocating = .init(gpa);
    try entry.extractTo(reader, &out.writer);
    const bytes = try out.toOwnedSlice();
    if (std.hash.Crc32.hash(bytes) != entry.crc32) return error.CheckpointCrcMismatch;
    return bytes;
}

pub fn load(gpa: Allocator, io: std.Io, path: []const u8) !Value {
    var file = try std.Io.Dir.cwd().openFile(io, path, .{});
    defer file.close(io);
    const archive_size = (try file.stat(io)).size;
    var buffer: [4096]u8 = undefined;
    var reader = file.reader(io, &buffer);
    var entries: std.StringHashMapUnmanaged(std.zip.Iterator.Entry) = .empty;
    var it = try std.zip.Iterator.init(&reader);
    var name_buffer: [std.fs.max_path_bytes]u8 = undefined;
    var pickle_name: ?[]const u8 = null;
    while (try it.next()) |entry| {
        const name = try gpa.dupe(u8, try entry.getFilename(&reader, &name_buffer, .{}));
        if (entries.contains(name)) return error.DuplicateCheckpointMember;
        try entries.put(gpa, name, entry);
        if (std.mem.endsWith(u8, name, "/data.pkl")) {
            if (pickle_name != null) return error.MultipleCheckpointRoots;
            pickle_name = name;
        }
    }
    const name = pickle_name orelse return error.MissingCheckpointPickle;
    const prefix = name[0 .. name.len - "data.pkl".len];
    if (entries.get(try std.fmt.allocPrint(gpa, "{s}byteorder", .{prefix}))) |entry| {
        if (!std.mem.eql(u8, try extract(gpa, &reader, entry, archive_size), "little")) return error.UnsupportedCheckpointByteOrder;
    }
    var pickle: Pickle = .{ .gpa = gpa, .data = try extract(gpa, &reader, entries.get(name).?, archive_size) };
    const root = try pickle.parse();
    if (root != .dict) return error.InvalidCheckpointRoot;
    var storages = pickle.storages.valueIterator();
    while (storages.next()) |value| {
        const storage = value.*;
        const member = try std.fmt.allocPrint(gpa, "{s}data/{s}", .{ prefix, storage.key });
        const entry = entries.get(member) orelse return error.MissingCheckpointStorage;
        if (entry.uncompressed_size != storage.count * storage.dtype.size()) return error.InvalidCheckpointStorageSize;
        storage.bytes = try extract(gpa, &reader, entry, archive_size);
    }
    return root;
}

/// Tagged JSON preserves integer dictionary keys, tuples, tensor views and state_dict attributes.
pub fn report(w: *std.Io.Writer, value: Value) !void {
    var budget: usize = 64 * 1024 * 1024;
    try reportValue(w, value, 0, &budget);
}

fn reportValue(w: *std.Io.Writer, value: Value, depth: usize, budget: *usize) anyerror!void {
    if (depth > 64) return error.MetadataTooDeep;
    // Charge an upper bound for JSON escaping and tensor descriptors to stop alias expansion.
    const charge = switch (value) {
        .string => |v| std.math.add(usize, std.math.mul(usize, v.len, 6) catch return error.MetadataTooLarge, 2) catch return error.MetadataTooLarge,
        .tensor => |v| std.math.add(usize, std.math.mul(usize, std.math.add(usize, v.storage.key.len, v.storage.location.len) catch return error.MetadataTooLarge, 6) catch return error.MetadataTooLarge, 2048) catch return error.MetadataTooLarge,
        else => @as(usize, 64),
    };
    if (charge > budget.*) return error.MetadataTooLarge;
    budget.* -= charge;
    switch (value) {
        .none => try w.writeAll("null"),
        .boolean => |v| try std.json.Stringify.value(v, .{}, w),
        .integer => |v| try std.json.Stringify.value(v, .{}, w),
        .float => |v| {
            try w.writeAll("{\"float\":");
            if (std.math.isFinite(v)) {
                try std.json.Stringify.value(v, .{}, w);
            } else {
                try std.json.Stringify.value(if (std.math.isNan(v)) "nan" else if (v < 0) "-inf" else "inf", .{}, w);
            }
            try w.writeByte('}');
        },
        .string => |v| try std.json.Stringify.value(v, .{}, w),
        .list, .tuple => |seq| {
            try w.print("{{\"{s}\":[", .{if (value == .list) "list" else "tuple"});
            for (seq.items, 0..) |item, i| {
                if (i > 0) try w.writeByte(',');
                try reportValue(w, item, depth + 1, budget);
            }
            try w.writeAll("]}");
        },
        .dict => |dict| {
            try w.writeAll("{\"dict\":[");
            for (dict.items.items, 0..) |e, i| {
                if (i > 0) try w.writeByte(',');
                try w.writeByte('[');
                try reportValue(w, e.key, depth + 1, budget);
                try w.writeByte(',');
                try reportValue(w, e.value, depth + 1, budget);
                try w.writeByte(']');
            }
            try w.writeAll("],\"attributes\":");
            if (dict.attributes) |attrs| try reportValue(w, attrs.*, depth + 1, budget) else try w.writeAll("null");
            try w.writeByte('}');
        },
        .tensor => |tensor| {
            try w.writeAll("{\"tensor\":");
            try std.json.Stringify.value(.{
                .dtype = @tagName(tensor.storage.dtype),
                .shape = tensor.shape,
                .strides = tensor.strides,
                .offset = tensor.offset,
                .storage = tensor.storage.key,
                .location = tensor.storage.location,
                .sha256 = std.fmt.bytesToHex(tensor.sha256(), .lower),
                .requires_grad = tensor.requires_grad,
            }, .{}, w);
            try w.writeByte('}');
        },
        else => return error.InvalidCheckpointValue,
    }
}

pub fn main(init: std.process.Init) !void {
    const gpa = init.arena.allocator();
    const args = try init.minimal.args.toSlice(gpa);
    if (args.len != 2) {
        std.debug.print("usage: checkpoint <checkpoint.pt|checkpoint.pth>\n", .{});
        return error.Usage;
    }
    const checkpoint = try load(gpa, init.io, args[1]);
    // Render before writing so a malformed or cyclic value never produces a partial report.
    var output: std.Io.Writer.Allocating = .init(gpa);
    try report(&output.writer, checkpoint);
    var stdout: u.Stdout = undefined;
    stdout.init(init.io);
    try stdout.w().writeAll(output.written());
    try stdout.w().writeByte('\n');
    try stdout.w().flush();
}

test "restricted pickle rejects executable globals and malformed input" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const cases = [_]struct { bytes: []const u8, err: anyerror }{
        .{ .bytes = "\x80\x02cos\nsystem\n.", .err = error.UnsupportedGlobal },
        .{ .bytes = "\x80\x05N.", .err = error.UnsupportedPickleProtocol },
        .{ .bytes = "\x80\x02h\x01.", .err = error.InvalidPickle },
        .{ .bytes = "\x80\x02R.", .err = error.InvalidPickle },
        .{ .bytes = "\x80\x02u.", .err = error.InvalidPickle },
        .{ .bytes = "\x80\x02X\xff\xff\xff\xff", .err = error.InvalidPickle },
        .{ .bytes = "\x80\x02NN.", .err = error.InvalidPickle },
        .{ .bytes = "\x80\x02N.N", .err = error.InvalidPickle },
    };
    for (cases) |case| {
        var pickle: Pickle = .{ .gpa = arena.allocator(), .data = case.bytes };
        try std.testing.expectError(case.err, pickle.parse());
    }
}

test "memoized containers retain mutations; cycles cannot recurse forever in reports" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    var pickle: Pickle = .{ .gpa = arena.allocator(), .data = "\x80\x02]q\x00h\x00a." };
    const root = try pickle.parse();
    try std.testing.expect(root.list.items[0].list == root.list);
    var out: std.Io.Writer.Allocating = .init(arena.allocator());
    try std.testing.expectError(error.MetadataTooDeep, report(&out.writer, root));
    var budget: usize = 0;
    try std.testing.expectError(error.MetadataTooLarge, reportValue(&out.writer, .none, 0, &budget));
}

test "tensor views validate extents and read logical elements with storage offsets" {
    var storage: Storage = .{ .key = "0", .location = "cpu", .dtype = .float32, .count = 4, .bytes = &.{ 0, 0, 128, 63, 0, 0, 0, 64, 0, 0, 64, 64, 0, 0, 128, 64 } };
    var tensor: Tensor = .{ .storage = &storage, .offset = 0, .shape = &.{ 2, 2 }, .strides = &.{ 1, 2 }, .requires_grad = false, .elements = 4 };
    try tensor.validate();
    const values = try tensor.floats(std.testing.allocator);
    defer std.testing.allocator.free(values);
    try std.testing.expectEqualSlices(f32, &.{ 1, 3, 2, 4 }, values);
    tensor.offset = 1;
    try std.testing.expectError(error.InvalidTensor, tensor.validate());
    tensor.shape = &.{0};
    tensor.strides = &.{1};
    tensor.elements = 0;
    tensor.offset = storage.count;
    try tensor.validate();
    tensor.offset += 1;
    try std.testing.expectError(error.InvalidTensor, tensor.validate());
}

test "synthetic checkpoint retains shared views, metadata and integer-keyed optimizer state" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const gpa = arena.allocator();
    const root = (try load(gpa, std.testing.io, "tools/dataset/testdata/checkpoint.pt")).dict;
    const model = root.get("model").?.dict;
    const weight = model.get("weight").?.tensor;
    const transposed = model.get("transposed").?.tensor;
    const offset = model.get("offset").?.tensor;
    try std.testing.expectEqual(@as(i64, 112), root.get("size").?.integer);
    try std.testing.expectEqual(@as(i64, 17), root.get("epoch").?.integer);
    try std.testing.expectEqual(@as(i64, -8589934592), root.get("signed").?.integer);
    try std.testing.expect(weight.storage == transposed.storage and weight.storage == offset.storage);
    try std.testing.expectEqualSlices(f32, &.{ 1, 4, 2, 5, 3, 6 }, try transposed.floats(gpa));
    try std.testing.expectEqualSlices(f32, &.{ 2, 4 }, try offset.floats(gpa));
    try std.testing.expectEqual(@as(u64, 1099511627776), std.mem.readInt(u64, model.get("counter").?.tensor.element(0)[0..8], .little));
    try std.testing.expectEqual(@as(usize, 0), model.get("empty").?.tensor.elements);
    try std.testing.expectEqual(@as(f64, 1.25), root.get("val_stats").?.dict.get("angle_median").?.float);
    const metadata = model.attributes.?.dict.get("_metadata").?.dict;
    try std.testing.expectEqual(@as(i64, 2), metadata.get("features").?.dict.get("version").?.integer);
    const optimizer = root.get("optimizer").?.dict;
    const state = optimizer.get("state").?.dict.items.items[0];
    try std.testing.expectEqual(@as(i64, 7), state.key.integer);
    try std.testing.expect(state.value.dict.get("momentum_buffer").?.tensor.storage == weight.storage);
    try std.testing.expectEqual(@as(i64, 7), optimizer.get("param_groups").?.list.items[0].dict.get("params").?.list.items[0].integer);
    const provenance = root.get("provenance").?.tuple.items;
    try std.testing.expectEqualStrings("eyes \xe2\x9c\x93", provenance[0].string);
    try std.testing.expect(provenance[1].boolean and provenance[2] == .none);
    const memo_probe = root.get("memo_probe").?.list.items;
    try std.testing.expectEqual(@as(usize, 301), memo_probe.len);
    try std.testing.expectEqualStrings("record_299", memo_probe[300].string);
}

test "corrupt stored tensor payloads fail CRC validation" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const gpa = arena.allocator();
    const fixture = try u.readFile(gpa, std.testing.io, "tools/dataset/testdata/checkpoint.pt");
    const pattern = [_]u8{ 0, 0, 128, 63, 0, 0, 0, 64, 0, 0, 64, 64 };
    const index = std.mem.indexOf(u8, fixture, &pattern) orelse return error.MissingSyntheticTensor;
    fixture[index] ^= 1;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(std.testing.io, .{ .sub_path = "corrupt.pt", .data = fixture });
    const path = try tmp.dir.realPathFileAlloc(std.testing.io, "corrupt.pt", gpa);
    try std.testing.expectError(error.CheckpointCrcMismatch, load(gpa, std.testing.io, path));
}

test "non-finite scalar metadata still produces valid tagged JSON" {
    const cases = [_]struct { value: f64, expected: []const u8 }{
        .{ .value = std.math.inf(f64), .expected = "{\"float\":\"inf\"}" },
        .{ .value = -std.math.inf(f64), .expected = "{\"float\":\"-inf\"}" },
        .{ .value = std.math.nan(f64), .expected = "{\"float\":\"nan\"}" },
    };
    for (cases) |case| {
        var out: std.Io.Writer.Allocating = .init(std.testing.allocator);
        defer out.deinit();
        try report(&out.writer, .{ .float = case.value });
        try std.testing.expectEqualStrings(case.expected, out.written());
    }
}
