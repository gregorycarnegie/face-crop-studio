//! A shared library opened at run time (ONNX Runtime, the NVIDIA driver). Link libc on Unix.
const std = @import("std");
const builtin = @import("builtin");

extern "kernel32" fn LoadLibraryW([*:0]const u16) callconv(.winapi) ?*anyopaque;
extern "kernel32" fn GetProcAddress(*anyopaque, [*:0]const u8) callconv(.winapi) ?*anyopaque;
extern "kernel32" fn FreeLibrary(*anyopaque) callconv(.winapi) c_int;

pub const Library = struct {
    handle: if (builtin.os.tag == .windows) *anyopaque else std.DynLib,

    pub fn open(gpa: std.mem.Allocator, path: []const u8) !Library {
        if (builtin.os.tag == .windows) {
            const wide = try std.unicode.wtf8ToWtf16LeAllocZ(gpa, path);
            defer gpa.free(wide);
            return .{ .handle = LoadLibraryW(wide) orelse return error.LibraryNotFound };
        } else {
            if (!builtin.link_libc) @compileError("loading a shared library requires libc: add -lc to the Zig command");
            return .{ .handle = std.DynLib.open(path) catch return error.LibraryNotFound };
        }
    }

    /// The symbol `name` as a function pointer of type `T`.
    pub fn lookup(self: *Library, comptime T: type, name: [:0]const u8) ?T {
        if (builtin.os.tag == .windows) return @ptrCast(GetProcAddress(self.handle, name) orelse return null);
        return self.handle.lookup(T, name);
    }

    pub fn close(self: *Library) void {
        if (builtin.os.tag == .windows) {
            _ = FreeLibrary(self.handle);
        } else self.handle.close();
    }
};
