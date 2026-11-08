//! The NVIDIA driver API, loaded at run time: no CUDA toolkit, only the driver every NVIDIA
//! install has (nvcuda.dll / libcuda.so.1). Kernels are PTX the driver JIT-compiles for the GPU
//! present; `gpu/*.zig` hold their Zig sources. Link libc on Unix (`zig run -lc`).
const std = @import("std");
const builtin = @import("builtin");
const Library = @import("dynlib.zig").Library;

const Result = c_int;
pub const Ptr = u64; // CUdeviceptr
const Context = *opaque {};
const Module = *opaque {};
pub const Function = *opaque {};

const Api = struct {
    init: *const fn (c_uint) callconv(.c) Result,
    device_get: *const fn (*c_int, c_int) callconv(.c) Result,
    device_name: *const fn ([*]u8, c_int, c_int) callconv(.c) Result,
    retain_primary: *const fn (*?Context, c_int) callconv(.c) Result,
    set_current: *const fn (?Context) callconv(.c) Result,
    load_module: *const fn (*?Module, *const anyopaque, c_uint, ?[*]const c_int, ?[*]const ?*anyopaque) callconv(.c) Result,
    get_function: *const fn (*?Function, Module, [*:0]const u8) callconv(.c) Result,
    alloc: *const fn (*Ptr, usize) callconv(.c) Result,
    free: *const fn (Ptr) callconv(.c) Result,
    upload: *const fn (Ptr, *const anyopaque, usize) callconv(.c) Result,
    download: *const fn (*anyopaque, Ptr, usize) callconv(.c) Result,
    copy: *const fn (Ptr, Ptr, usize) callconv(.c) Result,
    memset32: *const fn (Ptr, c_uint, usize) callconv(.c) Result,
    launch: *const fn (Function, c_uint, c_uint, c_uint, c_uint, c_uint, c_uint, c_uint, ?*anyopaque, [*]const ?*anyopaque, ?*anyopaque) callconv(.c) Result,
    synchronize: *const fn () callconv(.c) Result,
    error_name: *const fn (Result, *?[*:0]const u8) callconv(.c) Result,
};

const symbols = .{
    .{ "init", "cuInit" },                      .{ "device_get", "cuDeviceGet" },
    .{ "device_name", "cuDeviceGetName" },      .{ "retain_primary", "cuDevicePrimaryCtxRetain" },
    .{ "set_current", "cuCtxSetCurrent" },      .{ "load_module", "cuModuleLoadDataEx" },
    .{ "get_function", "cuModuleGetFunction" }, .{ "alloc", "cuMemAlloc_v2" },
    .{ "free", "cuMemFree_v2" },                .{ "upload", "cuMemcpyHtoD_v2" },
    .{ "download", "cuMemcpyDtoH_v2" },         .{ "copy", "cuMemcpyDtoD_v2" },
    .{ "memset32", "cuMemsetD32_v2" },          .{ "launch", "cuLaunchKernel" },
    .{ "synchronize", "cuCtxSynchronize" },     .{ "error_name", "cuGetErrorName" },
};

pub const Device = struct {
    library: Library,
    api: Api,
    module: Module,
    name: [128]u8 = @splat(0),

    /// Initialise device 0 and load `ptx` (NUL-terminated PTX text).
    pub fn open(gpa: std.mem.Allocator, ptx: [:0]const u8) !Device {
        var library = Library.open(gpa, if (builtin.os.tag == .windows) "nvcuda.dll" else "libcuda.so.1") catch return error.NvidiaDriverNotFound;
        errdefer library.close();
        var api: Api = undefined;
        inline for (symbols) |s| @field(api, s[0]) = library.lookup(@FieldType(Api, s[0]), s[1]) orelse return error.MissingDriverSymbol;
        var self: Device = .{ .library = library, .api = api, .module = undefined };
        try self.check(api.init(0));
        var device: c_int = 0;
        try self.check(api.device_get(&device, 0));
        try self.check(api.device_name(&self.name, self.name.len, device));
        var context: ?Context = null;
        try self.check(api.retain_primary(&context, device));
        try self.check(api.set_current(context));
        // Ask the JIT for its error log, so a PTX the driver rejects says why.
        var log: [8192]u8 = @splat(0);
        const options = [_]c_int{ 5, 6 }; // CU_JIT_ERROR_LOG_BUFFER, CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES
        const values = [_]?*anyopaque{ &log, @ptrFromInt(log.len) };
        var module: ?Module = null;
        check_jit: {
            const r = api.load_module(&module, ptx.ptr, options.len, &options, &values);
            if (r == 0) break :check_jit;
            std.debug.print("PTX rejected by the driver JIT:\n{s}\n", .{std.mem.sliceTo(&log, 0)});
            try self.check(r);
        }
        self.module = module orelse return error.CudaFailed;
        return self;
    }

    pub fn deviceName(self: *const Device) []const u8 {
        return std.mem.sliceTo(&self.name, 0);
    }

    fn check(self: *const Device, r: Result) !void {
        if (r == 0) return;
        var name: ?[*:0]const u8 = null;
        _ = self.api.error_name(r, &name);
        std.debug.print("CUDA driver error {d}: {s}\n", .{ r, if (name) |n| std.mem.span(n) else "unknown" });
        return error.CudaFailed;
    }

    pub fn function(self: *const Device, name: [:0]const u8) !Function {
        var f: ?Function = null;
        try self.check(self.api.get_function(&f, self.module, name.ptr));
        return f orelse error.CudaFailed;
    }

    pub fn alloc(self: *const Device, comptime T: type, count: usize) !Buffer(T) {
        var ptr: Ptr = 0;
        try self.check(self.api.alloc(&ptr, @max(1, count) * @sizeOf(T)));
        return .{ .ptr = ptr, .len = count };
    }

    pub fn free(self: *const Device, buffer: anytype) void {
        _ = self.api.free(buffer.ptr);
    }

    pub fn upload(self: *const Device, comptime T: type, dst: Buffer(T), src: []const T) !void {
        if (src.len > dst.len) return error.BufferTooSmall;
        if (src.len > 0) try self.check(self.api.upload(dst.ptr, src.ptr, src.len * @sizeOf(T)));
    }

    pub fn download(self: *const Device, comptime T: type, dst: []T, src: Buffer(T)) !void {
        if (dst.len > src.len) return error.BufferTooSmall;
        if (dst.len > 0) try self.check(self.api.download(dst.ptr, src.ptr, dst.len * @sizeOf(T)));
    }

    pub fn copy(self: *const Device, comptime T: type, dst: Buffer(T), src: Buffer(T), count: usize) !void {
        if (count > dst.len or count > src.len) return error.BufferTooSmall;
        if (count > 0) try self.check(self.api.copy(dst.ptr, src.ptr, count * @sizeOf(T)));
    }

    /// Fill with 32-bit zeros (0.0 for f32).
    pub fn zero(self: *const Device, buffer: anytype) !void {
        if (buffer.len > 0) try self.check(self.api.memset32(buffer.ptr, 0, buffer.len));
    }

    /// Launch `f` over `threads` work items in blocks of 256. `args` is a tuple of the kernel's
    /// parameters: u32/i32/f32 scalars and `Buffer`s, which pass as device pointers.
    pub fn launch(self: *const Device, f: Function, threads: usize, args: anytype) !void {
        if (threads == 0) return;
        const Args = @TypeOf(args);
        const info = @typeInfo(Args).@"struct";
        var values: Lowered(Args) = undefined;
        var pointers: [info.field_names.len]?*anyopaque = undefined;
        inline for (info.field_names, info.field_types, 0..) |name, T, i| {
            const v = @field(args, name);
            values[i] = if (comptime isBuffer(T)) v.ptr else v;
            pointers[i] = @ptrCast(&values[i]);
        }
        const block = 256;
        const grid = std.math.cast(c_uint, (threads + block - 1) / block) orelse return error.LaunchTooLarge;
        try self.check(self.api.launch(f, grid, 1, 1, block, 1, 1, 0, null, &pointers, null));
    }

    pub fn synchronize(self: *const Device) !void {
        try self.check(self.api.synchronize());
    }
};

pub fn Buffer(comptime T: type) type {
    return struct {
        ptr: Ptr,
        len: usize,
        pub const Elem = T;
    };
}

/// The argument tuple with every `Buffer` replaced by its device pointer.
fn Lowered(comptime Args: type) type {
    const field_types = @typeInfo(Args).@"struct".field_types;
    var types: [field_types.len]type = undefined;
    for (field_types, 0..) |T, i| types[i] = if (isBuffer(T)) Ptr else switch (T) {
        comptime_int => u32,
        comptime_float => f32,
        else => T,
    };
    return @Tuple(&types);
}

fn isBuffer(comptime T: type) bool {
    return @typeInfo(T) == .@"struct" and @hasField(T, "ptr");
}
