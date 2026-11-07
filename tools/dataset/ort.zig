//! Development-only ONNX Runtime oracle. Mirrors the ABI prefix in fcs-ort/src/sys.rs.
//! Link libc on Unix (`zig run -lc ...`) and set ORT_DYLIB_PATH if needed.
const std = @import("std");
const builtin = @import("builtin");
const Allocator = std.mem.Allocator;
const cc: std.builtin.CallingConvention = if (builtin.os.tag == .windows) .winapi else .c;
const Status = opaque {};
const Env = opaque {};
const SessionOptions = opaque {};
const Session = opaque {};
const MemoryInfo = opaque {};
const Value = opaque {};
const Shape = opaque {};
const Result = ?*Status;
const Unused = *const anyopaque;

// ONNX Runtime's table is append-only. Every omitted field still occupies one pointer.
const Api = extern struct {
    reserved_0: [2]Unused,
    error_message: *const fn (*const Status) callconv(cc) [*:0]const u8,
    create_env: *const fn (c_int, [*:0]const u8, *?*Env) callconv(cc) Result,
    reserved_4: [4]Unused,
    create_session: *const fn (*const Env, [*]const u8, usize, *const SessionOptions, *?*Session) callconv(cc) Result,
    run: *const fn (*Session, ?*const anyopaque, [*]const [*:0]const u8, [*]const *const Value, usize, [*]const [*:0]const u8, usize, [*]?*Value) callconv(cc) Result,
    create_options: *const fn (*?*SessionOptions) callconv(cc) Result,
    reserved_11: [12]Unused,
    set_optimization: *const fn (*SessionOptions, c_int) callconv(cc) Result,
    set_threads: *const fn (*SessionOptions, c_int) callconv(cc) Result,
    reserved_25: [24]Unused,
    create_tensor: *const fn (*const MemoryInfo, *anyopaque, usize, [*]const i64, usize, c_int, *?*Value) callconv(cc) Result,
    reserved_50: Unused,
    tensor_data: *const fn (*Value, *?*anyopaque) callconv(cc) Result,
    reserved_52: [8]Unused,
    element_type: *const fn (*const Shape, *c_int) callconv(cc) Result,
    rank: *const fn (*const Shape, *usize) callconv(cc) Result,
    dimensions: *const fn (*const Shape, [*]i64, usize) callconv(cc) Result,
    reserved_63: Unused,
    elements: *const fn (*const Shape, *usize) callconv(cc) Result,
    tensor_shape: *const fn (*const Value, *?*Shape) callconv(cc) Result,
    reserved_66: [3]Unused,
    create_memory: *const fn (c_int, c_int, *?*MemoryInfo) callconv(cc) Result,
    reserved_70: [22]Unused,
    release_env: *const fn (*Env) callconv(cc) void,
    release_status: *const fn (*Status) callconv(cc) void,
    release_memory: *const fn (*MemoryInfo) callconv(cc) void,
    release_session: *const fn (*Session) callconv(cc) void,
    release_value: *const fn (*Value) callconv(cc) void,
    reserved_97: [2]Unused,
    release_shape: *const fn (*Shape) callconv(cc) void,
    release_options: *const fn (*SessionOptions) callconv(cc) void,

    fn check(self: *const Api, result: Result) !void {
        if (result) |status| {
            defer self.release_status(status);
            std.debug.print("ONNX Runtime: {s}\n", .{std.mem.span(self.error_message(status))});
            return error.OnnxRuntime;
        }
    }
};

const ApiBase = extern struct {
    get_api: *const fn (u32) callconv(cc) ?*const Api,
    version: *const fn () callconv(cc) [*:0]const u8,
};

extern "kernel32" fn LoadLibraryW([*:0]const u16) callconv(.winapi) ?*anyopaque;
extern "kernel32" fn GetProcAddress(*anyopaque, [*:0]const u8) callconv(.winapi) ?*anyopaque;
extern "kernel32" fn FreeLibrary(*anyopaque) callconv(.winapi) c_int;

const Library = struct {
    handle: if (builtin.os.tag == .windows) *anyopaque else std.DynLib,

    fn open(gpa: Allocator, path: []const u8) !Library {
        if (builtin.os.tag == .windows) {
            const wide = try std.unicode.wtf8ToWtf16LeAllocZ(gpa, path);
            defer gpa.free(wide);
            return .{ .handle = LoadLibraryW(wide) orelse return error.RuntimeNotFound };
        } else {
            if (!builtin.link_libc) @compileError("ONNX Runtime requires libc: add -lc to the Zig command");
            return .{ .handle = try std.DynLib.open(path) };
        }
    }

    fn base(self: *Library) !*const ApiBase {
        const Entry = *const fn () callconv(cc) ?*const ApiBase;
        const entry: Entry = if (builtin.os.tag == .windows)
            @ptrCast(GetProcAddress(self.handle, "OrtGetApiBase") orelse return error.MissingOrtApi)
        else
            self.handle.lookup(Entry, "OrtGetApiBase") orelse return error.MissingOrtApi;
        return entry() orelse error.MissingOrtApi;
    }

    fn close(self: *Library) void {
        if (builtin.os.tag == .windows) {
            _ = FreeLibrary(self.handle);
        } else self.handle.close();
    }
};

pub const Output = struct { shape: []const i64, data: []const f32 };

pub const Oracle = struct {
    library: Library,
    api: *const Api,
    env: *Env,
    session: *Session,
    memory: *MemoryInfo,

    pub fn open(gpa: Allocator, path: []const u8, model: []const u8) !Oracle {
        var library = try Library.open(gpa, path);
        errdefer library.close();
        const base = try library.base();
        const api = base.get_api(24) orelse return error.RuntimeApiTooOld;
        var env: ?*Env = null;
        try api.check(api.create_env(2, "fcs-dataset", &env));
        const environment = env orelse return error.NullOrtHandle;
        errdefer api.release_env(environment);
        var options: ?*SessionOptions = null;
        try api.check(api.create_options(&options));
        const opts = options orelse return error.NullOrtHandle;
        defer api.release_options(opts);
        try api.check(api.set_optimization(opts, 99));
        try api.check(api.set_threads(opts, 4));
        var session: ?*Session = null;
        try api.check(api.create_session(environment, model.ptr, model.len, opts, &session));
        const s = session orelse return error.NullOrtHandle;
        errdefer api.release_session(s);
        var memory: ?*MemoryInfo = null;
        try api.check(api.create_memory(0, 0, &memory));
        return .{ .library = library, .api = api, .env = environment, .session = s, .memory = memory orelse return error.NullOrtHandle };
    }

    pub fn close(self: *Oracle) void {
        self.api.release_memory(self.memory);
        self.api.release_session(self.session);
        self.api.release_env(self.env);
        self.library.close();
        self.* = undefined;
    }

    pub fn run(self: *Oracle, gpa: Allocator, input: []const f32, shape: []const i64, input_name: [:0]const u8, output_name: [:0]const u8) !Output {
        return (try self.runAll(gpa, input, shape, input_name, &.{output_name}))[0];
    }

    /// One input, every named output, in the order named.
    pub fn runAll(self: *Oracle, gpa: Allocator, input: []const f32, shape: []const i64, input_name: [:0]const u8, output_names: []const [:0]const u8) ![]Output {
        if (try elements(shape) != input.len) return error.InvalidTensorShape;
        const bytes = std.math.mul(usize, input.len, @sizeOf(f32)) catch return error.InvalidTensorShape;
        var tensor: ?*Value = null;
        try self.api.check(self.api.create_tensor(self.memory, @ptrCast(@constCast(input.ptr)), bytes, shape.ptr, shape.len, 1, &tensor));
        const t = tensor orelse return error.NullOrtHandle;
        defer self.api.release_value(t);
        const names = try gpa.alloc([*:0]const u8, output_names.len);
        defer gpa.free(names);
        for (names, output_names) |*n, name| n.* = name.ptr;
        const values = try gpa.alloc(?*Value, output_names.len);
        defer gpa.free(values);
        @memset(values, null);
        defer for (values) |value| if (value) |v| self.api.release_value(v);
        try self.api.check(self.api.run(self.session, null, &.{input_name.ptr}, &.{t}, 1, names.ptr, names.len, values.ptr));
        const results = try gpa.alloc(Output, values.len);
        for (results, values) |*r, value| r.* = try self.read(gpa, value orelse return error.NullOrtHandle);
        return results;
    }

    fn read(self: *Oracle, gpa: Allocator, output: *Value) !Output {
        var info: ?*Shape = null;
        try self.api.check(self.api.tensor_shape(output, &info));
        const dims = info orelse return error.NullOrtHandle;
        defer self.api.release_shape(dims);
        var dtype: c_int = 0;
        try self.api.check(self.api.element_type(dims, &dtype));
        if (dtype != 1) return error.ExpectedFloatTensor;
        var rank: usize = 0;
        try self.api.check(self.api.rank(dims, &rank));
        if (rank > 32) return error.InvalidTensorShape;
        const output_shape = try gpa.alloc(i64, rank);
        errdefer gpa.free(output_shape);
        try self.api.check(self.api.dimensions(dims, output_shape.ptr, rank));
        const count = try elements(output_shape);
        var actual_count: usize = 0;
        try self.api.check(self.api.elements(dims, &actual_count));
        if (actual_count != count) return error.InvalidTensorShape;
        var data: ?*anyopaque = null;
        try self.api.check(self.api.tensor_data(output, &data));
        const values: []const f32 = if (count == 0) &.{} else @as([*]const f32, @ptrCast(@alignCast(data orelse return error.NullOrtHandle)))[0..count];
        return .{ .shape = output_shape, .data = try gpa.dupe(f32, values) };
    }
};

/// `ORT_DYLIB_PATH`, as for the Rust parity tests, else the platform's library name.
pub fn runtimePath(environ: *const std.process.Environ.Map) []const u8 {
    return environ.get("ORT_DYLIB_PATH") orelse switch (builtin.os.tag) {
        .windows => "onnxruntime.dll",
        .macos => "libonnxruntime.dylib",
        else => "libonnxruntime.so",
    };
}

/// A model's graph input and output names, as ONNX Runtime's `get_inputs`/`get_outputs` list
/// them: older exports also list initializers among the inputs, and those are skipped.
pub fn graphNames(gpa: Allocator, model: []const u8) !struct { inputs: []const [:0]const u8, outputs: []const [:0]const u8 } {
    const pb = @import("onnx.zig");
    const graph = try (try pb.field(model, 7) orelse return error.MissingGraph).bytes();
    var initializers: std.StringHashMapUnmanaged(void) = .empty;
    defer initializers.deinit(gpa);
    var inputs: std.ArrayList([:0]const u8) = .empty;
    var outputs: std.ArrayList([:0]const u8) = .empty;
    var r: pb.Reader = .{ .data = graph };
    while (try r.next()) |f| if (f.tag == 5) try initializers.put(gpa, try pb.string(try f.bytes(), 8), {});
    r = .{ .data = graph };
    while (try r.next()) |f| {
        if (f.tag != 11 and f.tag != 12) continue;
        const name = try pb.string(try f.bytes(), 1);
        if (f.tag == 11 and initializers.contains(name)) continue;
        try (if (f.tag == 11) &inputs else &outputs).append(gpa, try gpa.dupeSentinel(u8, name, 0));
    }
    return .{ .inputs = try inputs.toOwnedSlice(gpa), .outputs = try outputs.toOwnedSlice(gpa) };
}

fn elements(shape: []const i64) !usize {
    if (shape.len > 32) return error.InvalidTensorShape;
    for (shape) |dim| if (dim < 0) return error.InvalidTensorShape;
    if (std.mem.findScalar(i64, shape, 0) != null) return 0;
    var n: usize = 1;
    for (shape) |dim| n = std.math.mul(usize, n, std.math.cast(usize, dim) orelse return error.InvalidTensorShape) catch return error.InvalidTensorShape;
    _ = std.math.mul(usize, n, @sizeOf(f32)) catch return error.InvalidTensorShape;
    return n;
}

test "ORT ABI prefix offsets match the supported C API and Rust oracle" {
    try std.testing.expectEqual(@as(usize, 101 * @sizeOf(Unused)), @sizeOf(Api));
    inline for (.{ .{ "create_session", 8 }, .{ "run", 9 }, .{ "create_tensor", 49 }, .{ "tensor_data", 51 }, .{ "elements", 64 }, .{ "tensor_shape", 65 }, .{ "release_env", 92 }, .{ "release_options", 100 } }) |entry| {
        try std.testing.expectEqual(@as(usize, entry[1] * @sizeOf(Unused)), @offsetOf(Api, entry[0]));
    }
}

test "tensor shape checks reject negative dimensions and size overflow" {
    try std.testing.expectEqual(@as(usize, 12), try elements(&.{ 3, 4 }));
    try std.testing.expectEqual(@as(usize, 0), try elements(&.{ 0, 4 }));
    try std.testing.expectError(error.InvalidTensorShape, elements(&.{ 0, -1 }));
    try std.testing.expectError(error.InvalidTensorShape, elements(&.{ std.math.maxInt(i64), 8 }));
}
