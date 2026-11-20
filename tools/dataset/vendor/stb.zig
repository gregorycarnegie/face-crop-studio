//! Zig port of the active stb_image 2.30 / stb_image_write 1.16 codecs.
//! Upstream: nothings/stb, commit 013ac3beddff3dbffafd5177e7972067cd2b5083.
//! See LICENSE for Sean Barrett's MIT/public-domain license.
//! Scalar decoding and JPEG/PNG filtering follow stb; DEFLATE uses Zig's standard library.

const std = @import("std");
fn abs(value: i32) i32 {
    return @intCast(@abs(value));
}
pub threadlocal var allocator: std.mem.Allocator = std.heap.page_allocator;
const Header = extern struct { size: usize, padding: usize = 0 };
fn malloc(size: usize) ?*anyopaque {
    const total = std.math.add(usize, size, @sizeOf(Header)) catch return null;
    const block = allocator.alignedAlloc(u8, .@"16", total) catch return null;
    const header: *Header = @ptrCast(block.ptr);
    header.* = .{ .size = size };
    return block.ptr + @sizeOf(Header);
}
fn free(ptr: ?*anyopaque) void {
    const raw: [*]align(16) u8 = @alignCast(@as([*]u8, @ptrCast(ptr orelse return)) - @sizeOf(Header));
    const header: *Header = @ptrCast(raw);
    allocator.free(raw[0 .. header.size + @sizeOf(Header)]);
}
fn realloc(ptr: ?*anyopaque, size: usize) ?*anyopaque {
    const old = ptr orelse return malloc(size);
    const total = std.math.add(usize, size, @sizeOf(Header)) catch return null;
    const raw: [*]align(16) u8 = @alignCast(@as([*]u8, @ptrCast(old)) - @sizeOf(Header));
    const header: *Header = @ptrCast(raw);
    const old_slice: []align(16) u8 = raw[0 .. header.size + @sizeOf(Header)];
    const resized = allocator.realloc(old_slice, total) catch return null;
    const new_header: *Header = @ptrCast(resized.ptr);
    new_header.* = .{ .size = size };
    return resized.ptr + @sizeOf(Header);
}
const stbi_io_callbacks = extern struct {
    read: ?*const fn (user: ?*anyopaque, data: [*c]u8, size: i32) callconv(.c) i32 = null,
    skip: ?*const fn (user: ?*anyopaque, n: i32) callconv(.c) void = null,
    eof: ?*const fn (user: ?*anyopaque) callconv(.c) i32 = null,
};
fn memcpy(dst: ?*anyopaque, src: ?*const anyopaque, size: usize) ?*anyopaque {
    if (size != 0) @memcpy(@as([*]u8, @ptrCast(dst.?))[0..size], @as([*]const u8, @ptrCast(src.?))[0..size]);
    return dst;
}
fn memset(dst: ?*anyopaque, value: i32, size: usize) ?*anyopaque {
    if (size != 0) @memset(@as([*]u8, @ptrCast(dst.?))[0..size], @as(u8, @truncate(@as(u32, @bitCast(value)))));
    return dst;
}
fn memmove(dst: ?*anyopaque, src: ?*const anyopaque, size: usize) ?*anyopaque {
    if (size != 0) {
        const dest = @as([*]u8, @ptrCast(dst.?))[0..size];
        const source = @as([*]const u8, @ptrCast(src.?))[0..size];
        if (@intFromPtr(dst.?) < @intFromPtr(src.?)) std.mem.copyForwards(u8, dest, source) else std.mem.copyBackwards(u8, dest, source);
    }
    return dst;
}
fn _assert(message: [*c]const u8, _: [*c]const u8, line: u32) noreturn {
    std.debug.panic("stb invariant at upstream line {d}: {s}", .{ line, std.mem.span(message) });
}
const stbi__context = extern struct {
    img_x: u32 = 0,
    img_y: u32 = 0,
    img_n: i32 = 0,
    img_out_n: i32 = 0,
    io: stbi_io_callbacks = std.mem.zeroes(stbi_io_callbacks),
    io_user_data: ?*anyopaque = null,
    read_from_callbacks: i32 = 0,
    buflen: i32 = 0,
    buffer_start: [128]u8 = std.mem.zeroes([128]u8),
    callback_already_read: i32 = 0,
    img_buffer: [*c]u8 = null,
    img_buffer_end: [*c]u8 = null,
    img_buffer_original: [*c]u8 = null,
    img_buffer_original_end: [*c]u8 = null,
};
fn stbi__start_mem(s: [*c]stbi__context, buffer: [*c]const u8, len: i32) callconv(.c) void {
    s.*.io.read = null;
    s.*.read_from_callbacks = 0;
    s.*.callback_already_read = 0;
    s.*.img_buffer = blk: {
        const tmp = @as([*c]u8, @ptrCast(@alignCast(@constCast(buffer))));
        s.*.img_buffer_original = tmp;
        break :blk tmp;
    };
    s.*.img_buffer_end = blk: {
        const tmp = @as([*c]u8, @ptrCast(@alignCast(@constCast(buffer)))) + @as(usize, @bitCast(@as(isize, @intCast(len))));
        s.*.img_buffer_original_end = tmp;
        break :blk tmp;
    };
}
fn stbi__rewind(s: [*c]stbi__context) callconv(.c) void {
    s.*.img_buffer = s.*.img_buffer_original;
    s.*.img_buffer_end = s.*.img_buffer_original_end;
}
const STBI_ORDER_RGB: i32 = 0;
const stbi__result_info = extern struct {
    bits_per_channel: i32 = 0,
    num_channels: i32 = 0,
    channel_order: i32 = 0,
};
threadlocal var stbi__g_failure_reason: [*c]const u8 = null;
fn stbi__err(str: [*c]const u8) callconv(.c) i32 {
    stbi__g_failure_reason = str;
    return 0;
}
fn stbi__malloc(size: usize) callconv(.c) ?*anyopaque {
    return malloc(size);
}
fn stbi__addsizes_valid(a: i32, b: i32) callconv(.c) i32 {
    if (b < @as(i32, 0)) return 0;
    return @intFromBool(a <= (__INT_MAX__ - b));
}
fn stbi__mul2sizes_valid(a: i32, b: i32) callconv(.c) i32 {
    if ((a < @as(i32, 0)) or (b < @as(i32, 0))) return 0;
    if (b == @as(i32, 0)) return 1;
    return @intFromBool(a <= @divTrunc(__INT_MAX__, b));
}
fn stbi__mad2sizes_valid(a: i32, b: i32, add: i32) callconv(.c) i32 {
    return @intFromBool((stbi__mul2sizes_valid(a, b) != 0) and (stbi__addsizes_valid(a * b, add) != 0));
}
fn stbi__mad3sizes_valid(a: i32, b: i32, c: i32, add: i32) callconv(.c) i32 {
    return @intFromBool(((stbi__mul2sizes_valid(a, b) != 0) and (stbi__mul2sizes_valid(a * b, c) != 0)) and (stbi__addsizes_valid((a * b) * c, add) != 0));
}
fn stbi__malloc_mad2(a: i32, b: i32, add: i32) callconv(.c) ?*anyopaque {
    if (!(stbi__mad2sizes_valid(a, b, add) != 0)) return null;
    return stbi__malloc(@bitCast(@as(i64, (a * b) + add)));
}
fn stbi__malloc_mad3(a: i32, b: i32, c: i32, add: i32) callconv(.c) ?*anyopaque {
    if (!(stbi__mad3sizes_valid(a, b, c, add) != 0)) return null;
    return stbi__malloc(@bitCast(@as(i64, ((a * b) * c) + add)));
}
fn stbi__addints_valid(a: i32, b: i32) callconv(.c) i32 {
    if (@intFromBool(a >= @as(i32, 0)) != @intFromBool(b >= @as(i32, 0))) return 1;
    if ((a < @as(i32, 0)) and (b < @as(i32, 0))) return @intFromBool(a >= ((-__INT_MAX__ - @as(i32, 1)) - b));
    return @intFromBool(a <= (__INT_MAX__ - b));
}
fn stbi__mul2shorts_valid(a: i32, b: i32) callconv(.c) i32 {
    if ((b == @as(i32, 0)) or (b == -@as(i32, 1))) return 1;
    if (@intFromBool(a >= @as(i32, 0)) == @intFromBool(b >= @as(i32, 0))) return @intFromBool(a <= @divTrunc(__SHRT_MAX__, b));
    if (b < @as(i32, 0)) return @intFromBool(a <= @divTrunc(-__SHRT_MAX__ - @as(i32, 1), b));
    return @intFromBool(a >= @divTrunc(-__SHRT_MAX__ - @as(i32, 1), b));
}
pub fn stbi_image_free(retval_from_stbi_load: ?*anyopaque) void {
    free(retval_from_stbi_load);
}
var stbi__vertically_flip_on_load_global: i32 = 0;
threadlocal var stbi__vertically_flip_on_load_local: i32 = 0;
threadlocal var stbi__vertically_flip_on_load_set: i32 = 0;
fn stbi__load_main(s: [*c]stbi__context, x: [*c]i32, y: [*c]i32, comp: [*c]i32, req_comp: i32, ri: [*c]stbi__result_info, bpc: i32) callconv(.c) ?*anyopaque {
    _ = memset(@ptrCast(@alignCast(ri)), 0, @sizeOf(@TypeOf(ri.*)));
    ri.*.bits_per_channel = 8;
    ri.*.channel_order = STBI_ORDER_RGB;
    ri.*.num_channels = 0;
    if (stbi__png_test(s) != 0) return stbi__png_load(s, x, y, comp, req_comp, ri);
    if (stbi__bmp_test(s) != 0) return stbi__bmp_load(s, x, y, comp, req_comp, ri);
    _ = @sizeOf(@TypeOf(bpc));
    if (stbi__jpeg_test(s) != 0) return stbi__jpeg_load(s, x, y, comp, req_comp, ri);
    return @ptrCast(@alignCast(@as([*c]u8, @ptrFromInt(@as(usize, @intCast(@intFromPtr(if (stbi__err("unknown image type") != 0) @as(?*anyopaque, null) else @as(?*anyopaque, null))))))));
}
fn stbi__convert_16_to_8(orig: [*c]u16, w: i32, h: i32, channels: i32) callconv(.c) [*c]u8 {
    var i: i32 = undefined;
    const img_len: i32 = (w * h) * channels;
    var reduced: [*c]u8 = undefined;
    reduced = @ptrCast(@alignCast(stbi__malloc(@bitCast(@as(i64, img_len)))));
    if (@as(?*anyopaque, @ptrCast(@alignCast(reduced))) == @as(?*anyopaque, null)) return @as([*c]u8, @ptrFromInt(@as(usize, @intCast(@intFromPtr(if (stbi__err("outofmem") != 0) @as(?*anyopaque, null) else @as(?*anyopaque, null))))));
    {
        i = 0;
        while (i < img_len) : (i += 1) {
            reduced[@bitCast(@as(isize, @intCast(i)))] = @bitCast(@as(i8, @truncate((@as(i32, orig[@bitCast(@as(isize, @intCast(i)))]) >> @intCast(@as(i32, 8))) & @as(i32, 255))));
        }
    }
    free(@ptrCast(@alignCast(orig)));
    return reduced;
}
fn stbi__vertical_flip(image: ?*anyopaque, w: i32, h: i32, bytes_per_pixel: i32) callconv(.c) void {
    var row: i32 = undefined;
    const bytes_per_row: usize = @as(usize, @bitCast(@as(i64, w))) *% @as(usize, @bitCast(@as(i64, bytes_per_pixel)));
    var temp: [2048]u8 = undefined;
    const bytes: [*c]u8 = @ptrCast(@alignCast(image));
    {
        row = 0;
        while (row < (h >> @intCast(@as(i32, 1)))) : (row += 1) {
            var row0: [*c]u8 = bytes + (@as(usize, @bitCast(@as(i64, row))) *% bytes_per_row);
            var row1: [*c]u8 = bytes + (@as(usize, @bitCast(@as(i64, (h - row) - @as(i32, 1)))) *% bytes_per_row);
            var bytes_left: usize = bytes_per_row;
            while (bytes_left != 0) {
                const bytes_copy: usize = if (bytes_left < @sizeOf(@TypeOf(temp))) bytes_left else @sizeOf(@TypeOf(temp));
                _ = memcpy(@ptrCast(@alignCast(@as([*c]u8, @ptrCast(@alignCast(&temp))))), @ptrCast(@alignCast(row0)), bytes_copy);
                _ = memcpy(@ptrCast(@alignCast(row0)), @ptrCast(@alignCast(row1)), bytes_copy);
                _ = memcpy(@ptrCast(@alignCast(row1)), @ptrCast(@alignCast(@as([*c]u8, @ptrCast(@alignCast(&temp))))), bytes_copy);
                row0 += bytes_copy;
                row1 += bytes_copy;
                bytes_left -%= bytes_copy;
            }
        }
    }
}
fn stbi__load_and_postprocess_8bit(s: [*c]stbi__context, x: [*c]i32, y: [*c]i32, comp: [*c]i32, req_comp: i32) callconv(.c) [*c]u8 {
    var ri: stbi__result_info = undefined;
    var result: ?*anyopaque = stbi__load_main(s, x, y, comp, req_comp, &ri, 8);
    if (result == @as(?*anyopaque, null)) return null;
    _ = !!((ri.bits_per_channel == @as(i32, 8)) or (ri.bits_per_channel == @as(i32, 16))) or ((blk: {
        _assert("ri.bits_per_channel == 8 || ri.bits_per_channel == 16", "tools\\dataset\\vendor\\stb_image.h", 1269);
        break :blk 0;
    }) != 0);
    if (ri.bits_per_channel != @as(i32, 8)) {
        result = @ptrCast(@alignCast(stbi__convert_16_to_8(@ptrCast(@alignCast(result)), x.*, y.*, if (req_comp == @as(i32, 0)) comp.* else req_comp)));
        ri.bits_per_channel = 8;
    }
    if ((if (stbi__vertically_flip_on_load_set != 0) stbi__vertically_flip_on_load_local else stbi__vertically_flip_on_load_global) != 0) {
        const channels: i32 = if (req_comp != 0) req_comp else comp.*;
        stbi__vertical_flip(result, x.*, y.*, @bitCast(@as(u32, @truncate(@as(u64, @bitCast(@as(i64, channels))) *% @sizeOf(u8)))));
    }
    return @ptrCast(@alignCast(result));
}
pub fn stbi_load_from_memory(buffer: [*c]const u8, len: i32, x: [*c]i32, y: [*c]i32, comp: [*c]i32, req_comp: i32) [*c]u8 {
    var s: stbi__context = undefined;
    stbi__start_mem(&s, buffer, len);
    return stbi__load_and_postprocess_8bit(&s, x, y, comp, req_comp);
}
const STBI__SCAN_load: i32 = 0;
const STBI__SCAN_type: i32 = 1;
const STBI__SCAN_header: i32 = 2;
fn stbi__refill_buffer(s: [*c]stbi__context) callconv(.c) void {
    const n: i32 = s.*.io.read.?(s.*.io_user_data, @ptrCast(@alignCast(@as([*c]u8, @ptrCast(@alignCast(&s.*.buffer_start))))), s.*.buflen);
    s.*.callback_already_read += @truncate(@divExact(@as(i64, @bitCast(@intFromPtr(s.*.img_buffer) -% @intFromPtr(s.*.img_buffer_original))), @sizeOf(u8)));
    if (n == @as(i32, 0)) {
        s.*.read_from_callbacks = 0;
        s.*.img_buffer = @ptrCast(@alignCast(&s.*.buffer_start));
        s.*.img_buffer_end = @as([*c]u8, @ptrCast(@alignCast(&s.*.buffer_start))) + @as(usize, @bitCast(@as(isize, @intCast(@as(i32, 1)))));
        s.*.img_buffer.* = 0;
    } else {
        s.*.img_buffer = @ptrCast(@alignCast(&s.*.buffer_start));
        s.*.img_buffer_end = @as([*c]u8, @ptrCast(@alignCast(&s.*.buffer_start))) + @as(usize, @bitCast(@as(isize, @intCast(n))));
    }
}
fn stbi__get8(s: [*c]stbi__context) callconv(.c) u8 {
    if (s.*.img_buffer < s.*.img_buffer_end) return (blk: {
        const ref = &s.*.img_buffer;
        const tmp = ref.*;
        ref.* += 1;
        break :blk tmp;
    }).*;
    if (s.*.read_from_callbacks != 0) {
        stbi__refill_buffer(s);
        return (blk: {
            const ref = &s.*.img_buffer;
            const tmp = ref.*;
            ref.* += 1;
            break :blk tmp;
        }).*;
    }
    return 0;
}
fn stbi__at_eof(s: [*c]stbi__context) callconv(.c) i32 {
    if (s.*.io.read != null) {
        if (!(s.*.io.eof.?(s.*.io_user_data) != 0)) return 0;
        if (s.*.read_from_callbacks == @as(i32, 0)) return 1;
    }
    return @intFromBool(s.*.img_buffer >= s.*.img_buffer_end);
}
fn stbi__skip(s: [*c]stbi__context, n: i32) callconv(.c) void {
    if (n == @as(i32, 0)) return;
    if (n < @as(i32, 0)) {
        s.*.img_buffer = s.*.img_buffer_end;
        return;
    }
    if (s.*.io.read != null) {
        const blen: i32 = @truncate(@divExact(@as(i64, @bitCast(@intFromPtr(s.*.img_buffer_end) -% @intFromPtr(s.*.img_buffer))), @sizeOf(u8)));
        if (blen < n) {
            s.*.img_buffer = s.*.img_buffer_end;
            s.*.io.skip.?(s.*.io_user_data, n - blen);
            return;
        }
    }
    s.*.img_buffer += @as(usize, @bitCast(@as(isize, @intCast(n))));
}
fn stbi__getn(s: [*c]stbi__context, buffer: [*c]u8, n: i32) callconv(.c) i32 {
    if (s.*.io.read != null) {
        const blen: i32 = @truncate(@divExact(@as(i64, @bitCast(@intFromPtr(s.*.img_buffer_end) -% @intFromPtr(s.*.img_buffer))), @sizeOf(u8)));
        if (blen < n) {
            var res: i32 = undefined;
            var count: i32 = undefined;
            _ = memcpy(@ptrCast(@alignCast(buffer)), @ptrCast(@alignCast(s.*.img_buffer)), @bitCast(@as(i64, blen)));
            count = s.*.io.read.?(s.*.io_user_data, @as([*c]u8, @ptrCast(@alignCast(buffer))) + @as(usize, @bitCast(@as(isize, @intCast(blen)))), n - blen);
            res = @intFromBool(count == (n - blen));
            s.*.img_buffer = s.*.img_buffer_end;
            return res;
        }
    }
    if ((s.*.img_buffer + @as(usize, @bitCast(@as(isize, @intCast(n))))) <= s.*.img_buffer_end) {
        _ = memcpy(@ptrCast(@alignCast(buffer)), @ptrCast(@alignCast(s.*.img_buffer)), @bitCast(@as(i64, n)));
        s.*.img_buffer += @as(usize, @bitCast(@as(isize, @intCast(n))));
        return 1;
    } else return 0;
}
fn stbi__get16be(s: [*c]stbi__context) callconv(.c) i32 {
    const z: i32 = stbi__get8(s);
    return (z << @intCast(@as(i32, 8))) + @as(i32, stbi__get8(s));
}
fn stbi__get32be(s: [*c]stbi__context) callconv(.c) u32 {
    const z: u32 = @bitCast(@as(i32, stbi__get16be(s)));
    return (z << @intCast(@as(u32, 16))) +% @as(u32, @bitCast(@as(i32, stbi__get16be(s))));
}
fn stbi__get16le(s: [*c]stbi__context) callconv(.c) i32 {
    const z: i32 = stbi__get8(s);
    return z + (@as(i32, stbi__get8(s)) << @intCast(@as(i32, 8)));
}
fn stbi__get32le(s: [*c]stbi__context) callconv(.c) u32 {
    var z: u32 = @bitCast(@as(i32, stbi__get16le(s)));
    z +%= @as(u32, @bitCast(@as(i32, stbi__get16le(s)))) << @intCast(@as(u32, 16));
    return z;
}
fn stbi__compute_y(r: i32, g: i32, b: i32) callconv(.c) u8 {
    return @bitCast(@as(i8, @truncate((((r * @as(i32, 77)) + (g * @as(i32, 150))) + (@as(i32, 29) * b)) >> @intCast(@as(i32, 8)))));
}
fn stbi__convert_format(data: [*c]u8, img_n: i32, req_comp: i32, x: u32, y: u32) callconv(.c) [*c]u8 {
    var i: i32 = undefined;
    var j: i32 = undefined;
    var good: [*c]u8 = undefined;
    if (req_comp == img_n) return data;
    _ = !!((req_comp >= @as(i32, 1)) and (req_comp <= @as(i32, 4))) or ((blk: {
        _assert("req_comp >= 1 && req_comp <= 4", "tools\\dataset\\vendor\\stb_image.h", 1761);
        break :blk 0;
    }) != 0);
    good = @ptrCast(@alignCast(stbi__malloc_mad3(req_comp, @bitCast(@as(u32, @truncate(x))), @bitCast(@as(u32, @truncate(y))), 0)));
    if (@as(?*anyopaque, @ptrCast(@alignCast(good))) == @as(?*anyopaque, null)) {
        free(@ptrCast(@alignCast(data)));
        return @as([*c]u8, @ptrFromInt(@as(usize, @intCast(@intFromPtr(if (stbi__err("outofmem") != 0) @as(?*anyopaque, null) else @as(?*anyopaque, null))))));
    }
    {
        j = 0;
        while (j < @as(i32, @bitCast(@as(u32, @truncate(y))))) : (j += 1) {
            var src: [*c]u8 = data + ((@as(u32, @bitCast(@as(i32, j))) *% x) *% @as(u32, @bitCast(@as(i32, img_n))));
            var dest: [*c]u8 = good + ((@as(u32, @bitCast(@as(i32, j))) *% x) *% @as(u32, @bitCast(@as(i32, req_comp))));
            blk: {
                switch ((img_n * @as(i32, 8)) + req_comp) {
                    (@as(i32, 1) * @as(i32, 8)) + @as(i32, 2) => {
                        {
                            i = @bitCast(@as(u32, @truncate(x -% @as(u32, 1))));
                            while (i >= @as(i32, 0)) : ({
                                i -= 1;
                                src += @as(usize, @bitCast(@as(isize, @intCast(@as(i32, 1)))));
                                dest += @as(usize, @bitCast(@as(isize, @intCast(@as(i32, 2)))));
                            }) {
                                dest[@as(i32, 0)] = src[@as(i32, 0)];
                                dest[@as(i32, 1)] = 255;
                            }
                        }
                        break :blk;
                    },
                    (@as(i32, 1) * @as(i32, 8)) + @as(i32, 3) => {
                        {
                            i = @bitCast(@as(u32, @truncate(x -% @as(u32, 1))));
                            while (i >= @as(i32, 0)) : ({
                                i -= 1;
                                src += @as(usize, @bitCast(@as(isize, @intCast(@as(i32, 1)))));
                                dest += @as(usize, @bitCast(@as(isize, @intCast(@as(i32, 3)))));
                            }) {
                                dest[@as(i32, 0)] = blk_1: {
                                    const tmp = blk_2: {
                                        const tmp_3 = src[@as(i32, 0)];
                                        dest[@as(i32, 2)] = tmp_3;
                                        break :blk_2 tmp_3;
                                    };
                                    dest[@as(i32, 1)] = tmp;
                                    break :blk_1 tmp;
                                };
                            }
                        }
                        break :blk;
                    },
                    (@as(i32, 1) * @as(i32, 8)) + @as(i32, 4) => {
                        {
                            i = @bitCast(@as(u32, @truncate(x -% @as(u32, 1))));
                            while (i >= @as(i32, 0)) : ({
                                i -= 1;
                                src += @as(usize, @bitCast(@as(isize, @intCast(@as(i32, 1)))));
                                dest += @as(usize, @bitCast(@as(isize, @intCast(@as(i32, 4)))));
                            }) {
                                dest[@as(i32, 0)] = blk_1: {
                                    const tmp = blk_2: {
                                        const tmp_3 = src[@as(i32, 0)];
                                        dest[@as(i32, 2)] = tmp_3;
                                        break :blk_2 tmp_3;
                                    };
                                    dest[@as(i32, 1)] = tmp;
                                    break :blk_1 tmp;
                                };
                                dest[@as(i32, 3)] = 255;
                            }
                        }
                        break :blk;
                    },
                    (@as(i32, 2) * @as(i32, 8)) + @as(i32, 1) => {
                        {
                            i = @bitCast(@as(u32, @truncate(x -% @as(u32, 1))));
                            while (i >= @as(i32, 0)) : ({
                                i -= 1;
                                src += @as(usize, @bitCast(@as(isize, @intCast(@as(i32, 2)))));
                                dest += @as(usize, @bitCast(@as(isize, @intCast(@as(i32, 1)))));
                            }) {
                                dest[@as(i32, 0)] = src[@as(i32, 0)];
                            }
                        }
                        break :blk;
                    },
                    (@as(i32, 2) * @as(i32, 8)) + @as(i32, 3) => {
                        {
                            i = @bitCast(@as(u32, @truncate(x -% @as(u32, 1))));
                            while (i >= @as(i32, 0)) : ({
                                i -= 1;
                                src += @as(usize, @bitCast(@as(isize, @intCast(@as(i32, 2)))));
                                dest += @as(usize, @bitCast(@as(isize, @intCast(@as(i32, 3)))));
                            }) {
                                dest[@as(i32, 0)] = blk_1: {
                                    const tmp = blk_2: {
                                        const tmp_3 = src[@as(i32, 0)];
                                        dest[@as(i32, 2)] = tmp_3;
                                        break :blk_2 tmp_3;
                                    };
                                    dest[@as(i32, 1)] = tmp;
                                    break :blk_1 tmp;
                                };
                            }
                        }
                        break :blk;
                    },
                    (@as(i32, 2) * @as(i32, 8)) + @as(i32, 4) => {
                        {
                            i = @bitCast(@as(u32, @truncate(x -% @as(u32, 1))));
                            while (i >= @as(i32, 0)) : ({
                                i -= 1;
                                src += @as(usize, @bitCast(@as(isize, @intCast(@as(i32, 2)))));
                                dest += @as(usize, @bitCast(@as(isize, @intCast(@as(i32, 4)))));
                            }) {
                                dest[@as(i32, 0)] = blk_1: {
                                    const tmp = blk_2: {
                                        const tmp_3 = src[@as(i32, 0)];
                                        dest[@as(i32, 2)] = tmp_3;
                                        break :blk_2 tmp_3;
                                    };
                                    dest[@as(i32, 1)] = tmp;
                                    break :blk_1 tmp;
                                };
                                dest[@as(i32, 3)] = src[@as(i32, 1)];
                            }
                        }
                        break :blk;
                    },
                    (@as(i32, 3) * @as(i32, 8)) + @as(i32, 4) => {
                        {
                            i = @bitCast(@as(u32, @truncate(x -% @as(u32, 1))));
                            while (i >= @as(i32, 0)) : ({
                                i -= 1;
                                src += @as(usize, @bitCast(@as(isize, @intCast(@as(i32, 3)))));
                                dest += @as(usize, @bitCast(@as(isize, @intCast(@as(i32, 4)))));
                            }) {
                                dest[@as(i32, 0)] = src[@as(i32, 0)];
                                dest[@as(i32, 1)] = src[@as(i32, 1)];
                                dest[@as(i32, 2)] = src[@as(i32, 2)];
                                dest[@as(i32, 3)] = 255;
                            }
                        }
                        break :blk;
                    },
                    (@as(i32, 3) * @as(i32, 8)) + @as(i32, 1) => {
                        {
                            i = @bitCast(@as(u32, @truncate(x -% @as(u32, 1))));
                            while (i >= @as(i32, 0)) : ({
                                i -= 1;
                                src += @as(usize, @bitCast(@as(isize, @intCast(@as(i32, 3)))));
                                dest += @as(usize, @bitCast(@as(isize, @intCast(@as(i32, 1)))));
                            }) {
                                dest[@as(i32, 0)] = stbi__compute_y(src[@as(i32, 0)], src[@as(i32, 1)], src[@as(i32, 2)]);
                            }
                        }
                        break :blk;
                    },
                    (@as(i32, 3) * @as(i32, 8)) + @as(i32, 2) => {
                        {
                            i = @bitCast(@as(u32, @truncate(x -% @as(u32, 1))));
                            while (i >= @as(i32, 0)) : ({
                                i -= 1;
                                src += @as(usize, @bitCast(@as(isize, @intCast(@as(i32, 3)))));
                                dest += @as(usize, @bitCast(@as(isize, @intCast(@as(i32, 2)))));
                            }) {
                                dest[@as(i32, 0)] = stbi__compute_y(src[@as(i32, 0)], src[@as(i32, 1)], src[@as(i32, 2)]);
                                dest[@as(i32, 1)] = 255;
                            }
                        }
                        break :blk;
                    },
                    (@as(i32, 4) * @as(i32, 8)) + @as(i32, 1) => {
                        {
                            i = @bitCast(@as(u32, @truncate(x -% @as(u32, 1))));
                            while (i >= @as(i32, 0)) : ({
                                i -= 1;
                                src += @as(usize, @bitCast(@as(isize, @intCast(@as(i32, 4)))));
                                dest += @as(usize, @bitCast(@as(isize, @intCast(@as(i32, 1)))));
                            }) {
                                dest[@as(i32, 0)] = stbi__compute_y(src[@as(i32, 0)], src[@as(i32, 1)], src[@as(i32, 2)]);
                            }
                        }
                        break :blk;
                    },
                    (@as(i32, 4) * @as(i32, 8)) + @as(i32, 2) => {
                        {
                            i = @bitCast(@as(u32, @truncate(x -% @as(u32, 1))));
                            while (i >= @as(i32, 0)) : ({
                                i -= 1;
                                src += @as(usize, @bitCast(@as(isize, @intCast(@as(i32, 4)))));
                                dest += @as(usize, @bitCast(@as(isize, @intCast(@as(i32, 2)))));
                            }) {
                                dest[@as(i32, 0)] = stbi__compute_y(src[@as(i32, 0)], src[@as(i32, 1)], src[@as(i32, 2)]);
                                dest[@as(i32, 1)] = src[@as(i32, 3)];
                            }
                        }
                        break :blk;
                    },
                    (@as(i32, 4) * @as(i32, 8)) + @as(i32, 3) => {
                        {
                            i = @bitCast(@as(u32, @truncate(x -% @as(u32, 1))));
                            while (i >= @as(i32, 0)) : ({
                                i -= 1;
                                src += @as(usize, @bitCast(@as(isize, @intCast(@as(i32, 4)))));
                                dest += @as(usize, @bitCast(@as(isize, @intCast(@as(i32, 3)))));
                            }) {
                                dest[@as(i32, 0)] = src[@as(i32, 0)];
                                dest[@as(i32, 1)] = src[@as(i32, 1)];
                                dest[@as(i32, 2)] = src[@as(i32, 2)];
                            }
                        }
                        break :blk;
                    },
                    else => {
                        _ = !!(@as(i32, 0) != 0) or ((blk_1: {
                            _assert("0", "tools\\dataset\\vendor\\stb_image.h", 1790);
                            break :blk_1 0;
                        }) != 0);
                        free(@ptrCast(@alignCast(data)));
                        free(@ptrCast(@alignCast(good)));
                        return @as([*c]u8, @ptrFromInt(@as(usize, @intCast(@intFromPtr(if (stbi__err("unsupported") != 0) @as(?*anyopaque, null) else @as(?*anyopaque, null))))));
                    },
                }
            }
        }
    }
    free(@ptrCast(@alignCast(data)));
    return good;
}
fn stbi__compute_y_16(r: i32, g: i32, b: i32) callconv(.c) u16 {
    return @bitCast(@as(i16, @truncate((((r * @as(i32, 77)) + (g * @as(i32, 150))) + (@as(i32, 29) * b)) >> @intCast(@as(i32, 8)))));
}
fn stbi__convert_format16(data: [*c]u16, img_n: i32, req_comp: i32, x: u32, y: u32) callconv(.c) [*c]u16 {
    var i: i32 = undefined;
    var j: i32 = undefined;
    var good: [*c]u16 = undefined;
    if (req_comp == img_n) return data;
    _ = !!((req_comp >= @as(i32, 1)) and (req_comp <= @as(i32, 4))) or ((blk: {
        _assert("req_comp >= 1 && req_comp <= 4", "tools\\dataset\\vendor\\stb_image.h", 1818);
        break :blk 0;
    }) != 0);
    good = @ptrCast(@alignCast(stbi__malloc(((@as(u32, @bitCast(@as(i32, req_comp))) *% x) *% y) *% @as(u32, 2))));
    if (@as(?*anyopaque, @ptrCast(@alignCast(good))) == @as(?*anyopaque, null)) {
        free(@ptrCast(@alignCast(data)));
        return @ptrCast(@alignCast(@as([*c]u8, @ptrFromInt(@as(usize, @intCast(@intFromPtr(if (stbi__err("outofmem") != 0) @as(?*anyopaque, null) else @as(?*anyopaque, null))))))));
    }
    {
        j = 0;
        while (j < @as(i32, @bitCast(@as(u32, @truncate(y))))) : (j += 1) {
            var src: [*c]u16 = data + ((@as(u32, @bitCast(@as(i32, j))) *% x) *% @as(u32, @bitCast(@as(i32, img_n))));
            var dest: [*c]u16 = good + ((@as(u32, @bitCast(@as(i32, j))) *% x) *% @as(u32, @bitCast(@as(i32, req_comp))));
            blk: {
                switch ((img_n * @as(i32, 8)) + req_comp) {
                    (@as(i32, 1) * @as(i32, 8)) + @as(i32, 2) => {
                        {
                            i = @bitCast(@as(u32, @truncate(x -% @as(u32, 1))));
                            while (i >= @as(i32, 0)) : ({
                                i -= 1;
                                src += @as(usize, @bitCast(@as(isize, @intCast(@as(i32, 1)))));
                                dest += @as(usize, @bitCast(@as(isize, @intCast(@as(i32, 2)))));
                            }) {
                                dest[@as(i32, 0)] = src[@as(i32, 0)];
                                dest[@as(i32, 1)] = 65535;
                            }
                        }
                        break :blk;
                    },
                    (@as(i32, 1) * @as(i32, 8)) + @as(i32, 3) => {
                        {
                            i = @bitCast(@as(u32, @truncate(x -% @as(u32, 1))));
                            while (i >= @as(i32, 0)) : ({
                                i -= 1;
                                src += @as(usize, @bitCast(@as(isize, @intCast(@as(i32, 1)))));
                                dest += @as(usize, @bitCast(@as(isize, @intCast(@as(i32, 3)))));
                            }) {
                                dest[@as(i32, 0)] = blk_1: {
                                    const tmp = blk_2: {
                                        const tmp_3 = src[@as(i32, 0)];
                                        dest[@as(i32, 2)] = tmp_3;
                                        break :blk_2 tmp_3;
                                    };
                                    dest[@as(i32, 1)] = tmp;
                                    break :blk_1 tmp;
                                };
                            }
                        }
                        break :blk;
                    },
                    (@as(i32, 1) * @as(i32, 8)) + @as(i32, 4) => {
                        {
                            i = @bitCast(@as(u32, @truncate(x -% @as(u32, 1))));
                            while (i >= @as(i32, 0)) : ({
                                i -= 1;
                                src += @as(usize, @bitCast(@as(isize, @intCast(@as(i32, 1)))));
                                dest += @as(usize, @bitCast(@as(isize, @intCast(@as(i32, 4)))));
                            }) {
                                dest[@as(i32, 0)] = blk_1: {
                                    const tmp = blk_2: {
                                        const tmp_3 = src[@as(i32, 0)];
                                        dest[@as(i32, 2)] = tmp_3;
                                        break :blk_2 tmp_3;
                                    };
                                    dest[@as(i32, 1)] = tmp;
                                    break :blk_1 tmp;
                                };
                                dest[@as(i32, 3)] = 65535;
                            }
                        }
                        break :blk;
                    },
                    (@as(i32, 2) * @as(i32, 8)) + @as(i32, 1) => {
                        {
                            i = @bitCast(@as(u32, @truncate(x -% @as(u32, 1))));
                            while (i >= @as(i32, 0)) : ({
                                i -= 1;
                                src += @as(usize, @bitCast(@as(isize, @intCast(@as(i32, 2)))));
                                dest += @as(usize, @bitCast(@as(isize, @intCast(@as(i32, 1)))));
                            }) {
                                dest[@as(i32, 0)] = src[@as(i32, 0)];
                            }
                        }
                        break :blk;
                    },
                    (@as(i32, 2) * @as(i32, 8)) + @as(i32, 3) => {
                        {
                            i = @bitCast(@as(u32, @truncate(x -% @as(u32, 1))));
                            while (i >= @as(i32, 0)) : ({
                                i -= 1;
                                src += @as(usize, @bitCast(@as(isize, @intCast(@as(i32, 2)))));
                                dest += @as(usize, @bitCast(@as(isize, @intCast(@as(i32, 3)))));
                            }) {
                                dest[@as(i32, 0)] = blk_1: {
                                    const tmp = blk_2: {
                                        const tmp_3 = src[@as(i32, 0)];
                                        dest[@as(i32, 2)] = tmp_3;
                                        break :blk_2 tmp_3;
                                    };
                                    dest[@as(i32, 1)] = tmp;
                                    break :blk_1 tmp;
                                };
                            }
                        }
                        break :blk;
                    },
                    (@as(i32, 2) * @as(i32, 8)) + @as(i32, 4) => {
                        {
                            i = @bitCast(@as(u32, @truncate(x -% @as(u32, 1))));
                            while (i >= @as(i32, 0)) : ({
                                i -= 1;
                                src += @as(usize, @bitCast(@as(isize, @intCast(@as(i32, 2)))));
                                dest += @as(usize, @bitCast(@as(isize, @intCast(@as(i32, 4)))));
                            }) {
                                dest[@as(i32, 0)] = blk_1: {
                                    const tmp = blk_2: {
                                        const tmp_3 = src[@as(i32, 0)];
                                        dest[@as(i32, 2)] = tmp_3;
                                        break :blk_2 tmp_3;
                                    };
                                    dest[@as(i32, 1)] = tmp;
                                    break :blk_1 tmp;
                                };
                                dest[@as(i32, 3)] = src[@as(i32, 1)];
                            }
                        }
                        break :blk;
                    },
                    (@as(i32, 3) * @as(i32, 8)) + @as(i32, 4) => {
                        {
                            i = @bitCast(@as(u32, @truncate(x -% @as(u32, 1))));
                            while (i >= @as(i32, 0)) : ({
                                i -= 1;
                                src += @as(usize, @bitCast(@as(isize, @intCast(@as(i32, 3)))));
                                dest += @as(usize, @bitCast(@as(isize, @intCast(@as(i32, 4)))));
                            }) {
                                dest[@as(i32, 0)] = src[@as(i32, 0)];
                                dest[@as(i32, 1)] = src[@as(i32, 1)];
                                dest[@as(i32, 2)] = src[@as(i32, 2)];
                                dest[@as(i32, 3)] = 65535;
                            }
                        }
                        break :blk;
                    },
                    (@as(i32, 3) * @as(i32, 8)) + @as(i32, 1) => {
                        {
                            i = @bitCast(@as(u32, @truncate(x -% @as(u32, 1))));
                            while (i >= @as(i32, 0)) : ({
                                i -= 1;
                                src += @as(usize, @bitCast(@as(isize, @intCast(@as(i32, 3)))));
                                dest += @as(usize, @bitCast(@as(isize, @intCast(@as(i32, 1)))));
                            }) {
                                dest[@as(i32, 0)] = stbi__compute_y_16(src[@as(i32, 0)], src[@as(i32, 1)], src[@as(i32, 2)]);
                            }
                        }
                        break :blk;
                    },
                    (@as(i32, 3) * @as(i32, 8)) + @as(i32, 2) => {
                        {
                            i = @bitCast(@as(u32, @truncate(x -% @as(u32, 1))));
                            while (i >= @as(i32, 0)) : ({
                                i -= 1;
                                src += @as(usize, @bitCast(@as(isize, @intCast(@as(i32, 3)))));
                                dest += @as(usize, @bitCast(@as(isize, @intCast(@as(i32, 2)))));
                            }) {
                                dest[@as(i32, 0)] = stbi__compute_y_16(src[@as(i32, 0)], src[@as(i32, 1)], src[@as(i32, 2)]);
                                dest[@as(i32, 1)] = 65535;
                            }
                        }
                        break :blk;
                    },
                    (@as(i32, 4) * @as(i32, 8)) + @as(i32, 1) => {
                        {
                            i = @bitCast(@as(u32, @truncate(x -% @as(u32, 1))));
                            while (i >= @as(i32, 0)) : ({
                                i -= 1;
                                src += @as(usize, @bitCast(@as(isize, @intCast(@as(i32, 4)))));
                                dest += @as(usize, @bitCast(@as(isize, @intCast(@as(i32, 1)))));
                            }) {
                                dest[@as(i32, 0)] = stbi__compute_y_16(src[@as(i32, 0)], src[@as(i32, 1)], src[@as(i32, 2)]);
                            }
                        }
                        break :blk;
                    },
                    (@as(i32, 4) * @as(i32, 8)) + @as(i32, 2) => {
                        {
                            i = @bitCast(@as(u32, @truncate(x -% @as(u32, 1))));
                            while (i >= @as(i32, 0)) : ({
                                i -= 1;
                                src += @as(usize, @bitCast(@as(isize, @intCast(@as(i32, 4)))));
                                dest += @as(usize, @bitCast(@as(isize, @intCast(@as(i32, 2)))));
                            }) {
                                dest[@as(i32, 0)] = stbi__compute_y_16(src[@as(i32, 0)], src[@as(i32, 1)], src[@as(i32, 2)]);
                                dest[@as(i32, 1)] = src[@as(i32, 3)];
                            }
                        }
                        break :blk;
                    },
                    (@as(i32, 4) * @as(i32, 8)) + @as(i32, 3) => {
                        {
                            i = @bitCast(@as(u32, @truncate(x -% @as(u32, 1))));
                            while (i >= @as(i32, 0)) : ({
                                i -= 1;
                                src += @as(usize, @bitCast(@as(isize, @intCast(@as(i32, 4)))));
                                dest += @as(usize, @bitCast(@as(isize, @intCast(@as(i32, 3)))));
                            }) {
                                dest[@as(i32, 0)] = src[@as(i32, 0)];
                                dest[@as(i32, 1)] = src[@as(i32, 1)];
                                dest[@as(i32, 2)] = src[@as(i32, 2)];
                            }
                        }
                        break :blk;
                    },
                    else => {
                        _ = !!(@as(i32, 0) != 0) or ((blk_1: {
                            _assert("0", "tools\\dataset\\vendor\\stb_image.h", 1847);
                            break :blk_1 0;
                        }) != 0);
                        free(@ptrCast(@alignCast(data)));
                        free(@ptrCast(@alignCast(good)));
                        return @ptrCast(@alignCast(@as([*c]u8, @ptrFromInt(@as(usize, @intCast(@intFromPtr(if (stbi__err("unsupported") != 0) @as(?*anyopaque, null) else @as(?*anyopaque, null))))))));
                    },
                }
            }
        }
    }
    free(@ptrCast(@alignCast(data)));
    return good;
}
const stbi__huffman = extern struct {
    fast: [512]u8 = std.mem.zeroes([512]u8),
    code: [256]u16 = std.mem.zeroes([256]u16),
    values: [256]u8 = std.mem.zeroes([256]u8),
    size: [257]u8 = std.mem.zeroes([257]u8),
    maxcode: [18]u32 = std.mem.zeroes([18]u32),
    delta: [17]i32 = std.mem.zeroes([17]i32),
};
const struct_unnamed_6 = extern struct {
    id: i32 = 0,
    h: i32 = 0,
    v: i32 = 0,
    tq: i32 = 0,
    hd: i32 = 0,
    ha: i32 = 0,
    dc_pred: i32 = 0,
    x: i32 = 0,
    y: i32 = 0,
    w2: i32 = 0,
    h2: i32 = 0,
    data: [*c]u8 = null,
    raw_data: ?*anyopaque = null,
    raw_coeff: ?*anyopaque = null,
    linebuf: [*c]u8 = null,
    coeff: [*c]i16 = null,
    coeff_w: i32 = 0,
    coeff_h: i32 = 0,
};
const stbi__jpeg = extern struct {
    s: [*c]stbi__context = null,
    huff_dc: [4]stbi__huffman = std.mem.zeroes([4]stbi__huffman),
    huff_ac: [4]stbi__huffman = std.mem.zeroes([4]stbi__huffman),
    dequant: [4][64]u16 = std.mem.zeroes([4][64]u16),
    fast_ac: [4][512]i16 = std.mem.zeroes([4][512]i16),
    img_h_max: i32 = 0,
    img_v_max: i32 = 0,
    img_mcu_x: i32 = 0,
    img_mcu_y: i32 = 0,
    img_mcu_w: i32 = 0,
    img_mcu_h: i32 = 0,
    img_comp: [4]struct_unnamed_6 = std.mem.zeroes([4]struct_unnamed_6),
    code_buffer: u32 = 0,
    code_bits: i32 = 0,
    marker: u8 = 0,
    nomore: i32 = 0,
    progressive: i32 = 0,
    spec_start: i32 = 0,
    spec_end: i32 = 0,
    succ_high: i32 = 0,
    succ_low: i32 = 0,
    eob_run: i32 = 0,
    jfif: i32 = 0,
    app14_color_transform: i32 = 0,
    rgb: i32 = 0,
    scan_n: i32 = 0,
    order: [4]i32 = std.mem.zeroes([4]i32),
    restart_interval: i32 = 0,
    todo: i32 = 0,
    idct_block_kernel: ?*const fn (out: [*c]u8, out_stride: i32, data: [*c]i16) callconv(.c) void = null,
    YCbCr_to_RGB_kernel: ?*const fn (out: [*c]u8, y: [*c]const u8, pcb: [*c]const u8, pcr: [*c]const u8, count: i32, step: i32) callconv(.c) void = null,
    resample_row_hv_2_kernel: ?*const fn (out: [*c]u8, in_near: [*c]u8, in_far: [*c]u8, w: i32, hs: i32) callconv(.c) [*c]u8 = null,
};
fn stbi__build_huffman(h: [*c]stbi__huffman, count: [*c]i32) callconv(.c) i32 {
    var i: i32 = undefined;
    var j: i32 = undefined;
    var k: i32 = 0;
    var code: u32 = undefined;
    {
        i = 0;
        while (i < @as(i32, 16)) : (i += 1) {
            {
                j = 0;
                while (j < count[@bitCast(@as(isize, @intCast(i)))]) : (j += 1) {
                    h.*.size[
                        @bitCast(@as(isize, @intCast(blk: {
                            const ref = &k;
                            const tmp = ref.*;
                            ref.* += 1;
                            break :blk tmp;
                        })))
                    ] = @bitCast(@as(i8, @truncate(i + @as(i32, 1))));
                    if (k >= @as(i32, 257)) return stbi__err("bad size list");
                }
            }
        }
    }
    h.*.size[@bitCast(@as(isize, @intCast(k)))] = 0;
    code = 0;
    k = 0;
    {
        j = 1;
        while (j <= @as(i32, 16)) : (j += 1) {
            h.*.delta[@bitCast(@as(isize, @intCast(j)))] = @bitCast(@as(u32, @truncate(@as(u32, @bitCast(@as(i32, k))) -% code)));
            if (@as(i32, h.*.size[@bitCast(@as(isize, @intCast(k)))]) == j) {
                while (@as(i32, h.*.size[@bitCast(@as(isize, @intCast(k)))]) == j) {
                    h.*.code[
                        @bitCast(@as(isize, @intCast(blk: {
                            const ref = &k;
                            const tmp = ref.*;
                            ref.* += 1;
                            break :blk tmp;
                        })))
                    ] = @truncate(blk: {
                        const ref = &code;
                        const tmp = ref.*;
                        ref.* +%= 1;
                        break :blk tmp;
                    });
                }
                if ((code -% @as(u32, 1)) >= (@as(u32, 1) << @intCast(@as(u32, @bitCast(@as(i32, j)))))) return stbi__err("bad code lengths");
            }
            h.*.maxcode[@bitCast(@as(isize, @intCast(j)))] = code << @intCast(@as(u32, @bitCast(@as(i32, @as(i32, 16) - j))));
            code <<= @intCast(@as(u32, 1));
        }
    }
    h.*.maxcode[@bitCast(@as(isize, @intCast(j)))] = 4294967295;
    _ = memset(@ptrCast(@alignCast(@as([*c]u8, @ptrCast(@alignCast(&h.*.fast))))), 255, @bitCast(@as(i64, @as(i32, 1) << @intCast(FAST_BITS))));
    {
        i = 0;
        while (i < k) : (i += 1) {
            const s: i32 = h.*.size[@bitCast(@as(isize, @intCast(i)))];
            if (s <= FAST_BITS) {
                const c: i32 = @as(i32, h.*.code[@bitCast(@as(isize, @intCast(i)))]) << @intCast(FAST_BITS - s);
                const m: i32 = @as(i32, 1) << @intCast(FAST_BITS - s);
                {
                    j = 0;
                    while (j < m) : (j += 1) {
                        h.*.fast[@bitCast(@as(isize, @intCast(c + j)))] = @bitCast(@as(i8, @truncate(i)));
                    }
                }
            }
        }
    }
    return 1;
}
fn stbi__build_fast_ac(arg_fast_ac: [*c]i16, h: [*c]stbi__huffman) callconv(.c) void {
    var fast_ac = arg_fast_ac;
    var i: i32 = undefined;
    {
        i = 0;
        while (i < (@as(i32, 1) << @intCast(FAST_BITS))) : (i += 1) {
            const fast: u8 = h.*.fast[@bitCast(@as(isize, @intCast(i)))];
            fast_ac[@bitCast(@as(isize, @intCast(i)))] = 0;
            if (@as(i32, fast) < @as(i32, 255)) {
                const rs: i32 = h.*.values[fast];
                const run: i32 = (rs >> @intCast(@as(i32, 4))) & @as(i32, 15);
                const magbits: i32 = rs & @as(i32, 15);
                const len: i32 = h.*.size[fast];
                if ((magbits != 0) and ((len + magbits) <= FAST_BITS)) {
                    var k: i32 = ((i << @intCast(len)) & ((@as(i32, 1) << @intCast(FAST_BITS)) - @as(i32, 1))) >> @intCast(FAST_BITS - magbits);
                    const m: i32 = @as(i32, 1) << @intCast(magbits - @as(i32, 1));
                    if (k < m) {
                        {
                            const ref = &k;
                            ref.* = @bitCast(@as(u32, @truncate(@as(u32, @bitCast(@as(i32, ref.*))) +% ((~@as(u32, 0) << @intCast(@as(u32, @bitCast(@as(i32, magbits))))) +% @as(u32, 1)))));
                        }
                    }
                    if ((k >= -@as(i32, 128)) and (k <= @as(i32, 127))) {
                        fast_ac[@bitCast(@as(isize, @intCast(i)))] = @truncate(((k * @as(i32, 256)) + (run * @as(i32, 16))) + (len + magbits));
                    }
                }
            }
        }
    }
}
fn stbi__grow_buffer_unsafe(j: [*c]stbi__jpeg) callconv(.c) void {
    while (true) {
        const b: u32 = @bitCast(@as(i32, if (j.*.nomore != 0) @as(i32, 0) else @as(i32, stbi__get8(j.*.s))));
        if (b == @as(u32, 255)) {
            var c: i32 = stbi__get8(j.*.s);
            while (c == @as(i32, 255)) {
                c = stbi__get8(j.*.s);
            }
            if (c != @as(i32, 0)) {
                j.*.marker = @bitCast(@as(i8, @truncate(c)));
                j.*.nomore = 1;
                return;
            }
        }
        j.*.code_buffer |= b << @intCast(@as(u32, @bitCast(@as(i32, @as(i32, 24) - j.*.code_bits))));
        j.*.code_bits += 8;
        if (!(j.*.code_bits <= @as(i32, 24))) break;
    }
}
const stbi__bmask: [17]u32 = [17]u32{
    0,     1, 3, 7, 15, 31, 63, 127, 255, 511, 1023, 2047, 4095, 8191, 16383, 32767,
    65535,
};
fn stbi__jpeg_huff_decode(j: [*c]stbi__jpeg, h: [*c]stbi__huffman) callconv(.c) i32 {
    var temp: u32 = undefined;
    var c: i32 = undefined;
    var k: i32 = undefined;
    if (j.*.code_bits < @as(i32, 16)) {
        stbi__grow_buffer_unsafe(j);
    }
    c = @bitCast(@as(u32, @truncate((j.*.code_buffer >> @intCast(@as(u32, @bitCast(@as(i32, @as(i32, 32) - FAST_BITS))))) & @as(u32, @bitCast(@as(i32, (@as(i32, 1) << @intCast(FAST_BITS)) - @as(i32, 1)))))));
    k = h.*.fast[@bitCast(@as(isize, @intCast(c)))];
    if (k < @as(i32, 255)) {
        const s: i32 = h.*.size[@bitCast(@as(isize, @intCast(k)))];
        if (s > j.*.code_bits) return -@as(i32, 1);
        j.*.code_buffer <<= @intCast(@as(u32, @bitCast(@as(i32, s))));
        j.*.code_bits -= s;
        return h.*.values[@bitCast(@as(isize, @intCast(k)))];
    }
    temp = j.*.code_buffer >> @intCast(@as(u32, 16));
    {
        k = FAST_BITS + @as(i32, 1);
        while (true) : (k += 1) if (temp < h.*.maxcode[@bitCast(@as(isize, @intCast(k)))]) break;
    }
    if (k == @as(i32, 17)) {
        j.*.code_bits -= 16;
        return -@as(i32, 1);
    }
    if (k > j.*.code_bits) return -@as(i32, 1);
    c = @bitCast(@as(u32, @truncate(((j.*.code_buffer >> @intCast(@as(u32, @bitCast(@as(i32, @as(i32, 32) - k))))) & stbi__bmask[@bitCast(@as(isize, @intCast(k)))]) +% @as(u32, @bitCast(@as(i32, h.*.delta[@bitCast(@as(isize, @intCast(k)))]))))));
    if ((c < @as(i32, 0)) or (c >= @as(i32, 256))) return -@as(i32, 1);
    _ = !!(((j.*.code_buffer >> @intCast(@as(u32, @bitCast(@as(i32, @as(i32, 32) - @as(i32, h.*.size[@bitCast(@as(isize, @intCast(c)))])))))) & stbi__bmask[h.*.size[@bitCast(@as(isize, @intCast(c)))]]) == @as(u32, @bitCast(@as(i32, @as(i32, h.*.code[@bitCast(@as(isize, @intCast(c)))]))))) or ((blk: {
        _assert("(((j->code_buffer) >> (32 - h->size[c])) & stbi__bmask[h->size[c]]) == h->code[c]", "tools\\dataset\\vendor\\stb_image.h", 2140);
        break :blk 0;
    }) != 0);
    j.*.code_bits -= k;
    j.*.code_buffer <<= @intCast(@as(u32, @bitCast(@as(i32, k))));
    return h.*.values[@bitCast(@as(isize, @intCast(c)))];
}
const stbi__jbias: [16]i32 = [16]i32{
    0,
    -@as(i32, 1),
    -@as(i32, 3),
    -@as(i32, 7),
    -@as(i32, 15),
    -@as(i32, 31),
    -@as(i32, 63),
    -@as(i32, 127),
    -@as(i32, 255),
    -@as(i32, 511),
    -@as(i32, 1023),
    -@as(i32, 2047),
    -@as(i32, 4095),
    -@as(i32, 8191),
    -@as(i32, 16383),
    -@as(i32, 32767),
};
fn stbi__extend_receive(j: [*c]stbi__jpeg, n: i32) callconv(.c) i32 {
    var k: u32 = undefined;
    var sgn: i32 = undefined;
    if (j.*.code_bits < n) {
        stbi__grow_buffer_unsafe(j);
    }
    if (j.*.code_bits < n) return 0;
    sgn = @bitCast(@as(u32, @truncate(j.*.code_buffer >> @intCast(@as(u32, 31)))));
    k = (j.*.code_buffer << @intCast(@as(u32, @bitCast(@as(i32, n))))) | (j.*.code_buffer >> @intCast(@as(u32, @bitCast(@as(i32, -n & @as(i32, 31))))));
    j.*.code_buffer = k & ~stbi__bmask[@bitCast(@as(isize, @intCast(n)))];
    k &= stbi__bmask[@bitCast(@as(isize, @intCast(n)))];
    j.*.code_bits -= n;
    return @bitCast(@as(u32, @truncate(k +% @as(u32, @bitCast(@as(i32, stbi__jbias[@bitCast(@as(isize, @intCast(n)))] & (sgn - @as(i32, 1))))))));
}
fn stbi__jpeg_get_bits(j: [*c]stbi__jpeg, n: i32) callconv(.c) i32 {
    var k: u32 = undefined;
    if (j.*.code_bits < n) {
        stbi__grow_buffer_unsafe(j);
    }
    if (j.*.code_bits < n) return 0;
    k = (j.*.code_buffer << @intCast(@as(u32, @bitCast(@as(i32, n))))) | (j.*.code_buffer >> @intCast(@as(u32, @bitCast(@as(i32, -n & @as(i32, 31))))));
    j.*.code_buffer = k & ~stbi__bmask[@bitCast(@as(isize, @intCast(n)))];
    k &= stbi__bmask[@bitCast(@as(isize, @intCast(n)))];
    j.*.code_bits -= n;
    return @bitCast(@as(u32, @truncate(k)));
}
fn stbi__jpeg_get_bit(j: [*c]stbi__jpeg) callconv(.c) i32 {
    var k: u32 = undefined;
    if (j.*.code_bits < @as(i32, 1)) {
        stbi__grow_buffer_unsafe(j);
    }
    if (j.*.code_bits < @as(i32, 1)) return 0;
    k = j.*.code_buffer;
    j.*.code_buffer <<= @intCast(@as(u32, 1));
    j.*.code_bits -= 1;
    return @bitCast(@as(u32, @truncate(k & @as(u32, 2147483648))));
}
const stbi__jpeg_dezigzag: [79]u8 = [79]u8{
    0,  1,  8,  16, 9,  2,  3,  10, 17, 24, 32, 25, 18, 11, 4,  5,
    12, 19, 26, 33, 40, 48, 41, 34, 27, 20, 13, 6,  7,  14, 21, 28,
    35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51,
    58, 59, 52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63,
    63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63,
};
fn stbi__jpeg_decode_block(j: [*c]stbi__jpeg, arg_data: [*c]i16, hdc: [*c]stbi__huffman, hac: [*c]stbi__huffman, fac: [*c]i16, b: i32, dequant: [*c]u16) callconv(.c) i32 {
    var data = arg_data;
    var diff: i32 = undefined;
    var dc: i32 = undefined;
    var k: i32 = undefined;
    var t: i32 = undefined;
    if (j.*.code_bits < @as(i32, 16)) {
        stbi__grow_buffer_unsafe(j);
    }
    t = stbi__jpeg_huff_decode(j, hdc);
    if ((t < @as(i32, 0)) or (t > @as(i32, 15))) return stbi__err("bad huffman code");
    _ = memset(@ptrCast(@alignCast(data)), 0, @as(u64, 64) *% @sizeOf(@TypeOf(data[@as(i32, 0)])));
    diff = if (t != 0) stbi__extend_receive(j, t) else @as(i32, 0);
    if (!(stbi__addints_valid(j.*.img_comp[@bitCast(@as(isize, @intCast(b)))].dc_pred, diff) != 0)) return stbi__err("bad delta");
    dc = j.*.img_comp[@bitCast(@as(isize, @intCast(b)))].dc_pred + diff;
    j.*.img_comp[@bitCast(@as(isize, @intCast(b)))].dc_pred = dc;
    if (!(stbi__mul2shorts_valid(dc, dequant[@as(i32, 0)]) != 0)) return stbi__err("can't merge dc and ac");
    data[@as(i32, 0)] = @truncate(dc * @as(i32, dequant[@as(i32, 0)]));
    k = 1;
    while (true) {
        var zig: u32 = undefined;
        var c: i32 = undefined;
        var r: i32 = undefined;
        var s: i32 = undefined;
        if (j.*.code_bits < @as(i32, 16)) {
            stbi__grow_buffer_unsafe(j);
        }
        c = @bitCast(@as(u32, @truncate((j.*.code_buffer >> @intCast(@as(u32, @bitCast(@as(i32, @as(i32, 32) - FAST_BITS))))) & @as(u32, @bitCast(@as(i32, (@as(i32, 1) << @intCast(FAST_BITS)) - @as(i32, 1)))))));
        r = fac[@bitCast(@as(isize, @intCast(c)))];
        if (r != 0) {
            k += (r >> @intCast(@as(i32, 4))) & @as(i32, 15);
            s = r & @as(i32, 15);
            if (s > j.*.code_bits) return stbi__err("bad huffman code");
            j.*.code_buffer <<= @intCast(@as(u32, @bitCast(@as(i32, s))));
            j.*.code_bits -= s;
            zig = stbi__jpeg_dezigzag[
                @bitCast(@as(isize, @intCast(blk: {
                    const ref = &k;
                    const tmp = ref.*;
                    ref.* += 1;
                    break :blk tmp;
                })))
            ];
            data[zig] = @truncate((r >> @intCast(@as(i32, 8))) * @as(i32, dequant[zig]));
        } else {
            const rs: i32 = stbi__jpeg_huff_decode(j, hac);
            if (rs < @as(i32, 0)) return stbi__err("bad huffman code");
            s = rs & @as(i32, 15);
            r = rs >> @intCast(@as(i32, 4));
            if (s == @as(i32, 0)) {
                if (rs != @as(i32, 240)) break;
                k += 16;
            } else {
                k += r;
                zig = stbi__jpeg_dezigzag[
                    @bitCast(@as(isize, @intCast(blk: {
                        const ref = &k;
                        const tmp = ref.*;
                        ref.* += 1;
                        break :blk tmp;
                    })))
                ];
                data[zig] = @truncate(stbi__extend_receive(j, s) * @as(i32, dequant[zig]));
            }
        }
        if (!(k < @as(i32, 64))) break;
    }
    return 1;
}
fn stbi__jpeg_decode_block_prog_dc(j: [*c]stbi__jpeg, arg_data: [*c]i16, hdc: [*c]stbi__huffman, b: i32) callconv(.c) i32 {
    var data = arg_data;
    var diff: i32 = undefined;
    var dc: i32 = undefined;
    var t: i32 = undefined;
    if (j.*.spec_end != @as(i32, 0)) return stbi__err("can't merge dc and ac");
    if (j.*.code_bits < @as(i32, 16)) {
        stbi__grow_buffer_unsafe(j);
    }
    if (j.*.succ_high == @as(i32, 0)) {
        _ = memset(@ptrCast(@alignCast(data)), 0, @as(u64, 64) *% @sizeOf(@TypeOf(data[@as(i32, 0)])));
        t = stbi__jpeg_huff_decode(j, hdc);
        if ((t < @as(i32, 0)) or (t > @as(i32, 15))) return stbi__err("can't merge dc and ac");
        diff = if (t != 0) stbi__extend_receive(j, t) else @as(i32, 0);
        if (!(stbi__addints_valid(j.*.img_comp[@bitCast(@as(isize, @intCast(b)))].dc_pred, diff) != 0)) return stbi__err("bad delta");
        dc = j.*.img_comp[@bitCast(@as(isize, @intCast(b)))].dc_pred + diff;
        j.*.img_comp[@bitCast(@as(isize, @intCast(b)))].dc_pred = dc;
        if (!(stbi__mul2shorts_valid(dc, @as(i32, 1) << @intCast(j.*.succ_low)) != 0)) return stbi__err("can't merge dc and ac");
        data[@as(i32, 0)] = @truncate(dc * (@as(i32, 1) << @intCast(j.*.succ_low)));
    } else {
        if (stbi__jpeg_get_bit(j) != 0) {
            {
                const ref = &data[@as(i32, 0)];
                ref.* = @truncate(@as(i32, ref.*) + @as(i32, @as(i16, @truncate(@as(i32, 1) << @intCast(j.*.succ_low)))));
            }
        }
    }
    return 1;
}
fn stbi__jpeg_decode_block_prog_ac(j: [*c]stbi__jpeg, arg_data: [*c]i16, hac: [*c]stbi__huffman, fac: [*c]i16) callconv(.c) i32 {
    var data = arg_data;
    var k: i32 = undefined;
    if (j.*.spec_start == @as(i32, 0)) return stbi__err("can't merge dc and ac");
    if (j.*.succ_high == @as(i32, 0)) {
        const shift: i32 = j.*.succ_low;
        if (j.*.eob_run != 0) {
            j.*.eob_run -= 1;
            return 1;
        }
        k = j.*.spec_start;
        while (true) {
            var zig: u32 = undefined;
            var c: i32 = undefined;
            var r: i32 = undefined;
            var s: i32 = undefined;
            if (j.*.code_bits < @as(i32, 16)) {
                stbi__grow_buffer_unsafe(j);
            }
            c = @bitCast(@as(u32, @truncate((j.*.code_buffer >> @intCast(@as(u32, @bitCast(@as(i32, @as(i32, 32) - FAST_BITS))))) & @as(u32, @bitCast(@as(i32, (@as(i32, 1) << @intCast(FAST_BITS)) - @as(i32, 1)))))));
            r = fac[@bitCast(@as(isize, @intCast(c)))];
            if (r != 0) {
                k += (r >> @intCast(@as(i32, 4))) & @as(i32, 15);
                s = r & @as(i32, 15);
                if (s > j.*.code_bits) return stbi__err("bad huffman code");
                j.*.code_buffer <<= @intCast(@as(u32, @bitCast(@as(i32, s))));
                j.*.code_bits -= s;
                zig = stbi__jpeg_dezigzag[
                    @bitCast(@as(isize, @intCast(blk: {
                        const ref = &k;
                        const tmp = ref.*;
                        ref.* += 1;
                        break :blk tmp;
                    })))
                ];
                data[zig] = @truncate((r >> @intCast(@as(i32, 8))) * (@as(i32, 1) << @intCast(shift)));
            } else {
                const rs: i32 = stbi__jpeg_huff_decode(j, hac);
                if (rs < @as(i32, 0)) return stbi__err("bad huffman code");
                s = rs & @as(i32, 15);
                r = rs >> @intCast(@as(i32, 4));
                if (s == @as(i32, 0)) {
                    if (r < @as(i32, 15)) {
                        j.*.eob_run = @as(i32, 1) << @intCast(r);
                        if (r != 0) {
                            j.*.eob_run += stbi__jpeg_get_bits(j, r);
                        }
                        j.*.eob_run -= 1;
                        break;
                    }
                    k += 16;
                } else {
                    k += r;
                    zig = stbi__jpeg_dezigzag[
                        @bitCast(@as(isize, @intCast(blk: {
                            const ref = &k;
                            const tmp = ref.*;
                            ref.* += 1;
                            break :blk tmp;
                        })))
                    ];
                    data[zig] = @truncate(stbi__extend_receive(j, s) * (@as(i32, 1) << @intCast(shift)));
                }
            }
            if (!(k <= j.*.spec_end)) break;
        }
    } else {
        const bit: i16 = @truncate(@as(i32, 1) << @intCast(j.*.succ_low));
        if (j.*.eob_run != 0) {
            j.*.eob_run -= 1;
            {
                k = j.*.spec_start;
                while (k <= j.*.spec_end) : (k += 1) {
                    const p: [*c]i16 = &data[stbi__jpeg_dezigzag[@bitCast(@as(isize, @intCast(k)))]];
                    if (@as(i32, p.*) != @as(i32, 0)) if (stbi__jpeg_get_bit(j) != 0) if ((@as(i32, p.*) & @as(i32, bit)) == @as(i32, 0)) {
                        if (@as(i32, p.*) > @as(i32, 0)) {
                            {
                                const ref = &p.*;
                                ref.* = @truncate(@as(i32, ref.*) + @as(i32, bit));
                            }
                        } else {
                            {
                                const ref = &p.*;
                                ref.* = @truncate(@as(i32, ref.*) - @as(i32, bit));
                            }
                        }
                    };
                }
            }
        } else {
            k = j.*.spec_start;
            while (true) {
                var r: i32 = undefined;
                var s: i32 = undefined;
                const rs: i32 = stbi__jpeg_huff_decode(j, hac);
                if (rs < @as(i32, 0)) return stbi__err("bad huffman code");
                s = rs & @as(i32, 15);
                r = rs >> @intCast(@as(i32, 4));
                if (s == @as(i32, 0)) {
                    if (r < @as(i32, 15)) {
                        j.*.eob_run = (@as(i32, 1) << @intCast(r)) - @as(i32, 1);
                        if (r != 0) {
                            j.*.eob_run += stbi__jpeg_get_bits(j, r);
                        }
                        r = 64;
                    } else {}
                } else {
                    if (s != @as(i32, 1)) return stbi__err("bad huffman code");
                    if (stbi__jpeg_get_bit(j) != 0) {
                        s = bit;
                    } else {
                        s = -@as(i32, bit);
                    }
                }
                while (k <= j.*.spec_end) {
                    const p: [*c]i16 = &data[
                        stbi__jpeg_dezigzag[
                            @bitCast(@as(isize, @intCast(blk: {
                                const ref = &k;
                                const tmp = ref.*;
                                ref.* += 1;
                                break :blk tmp;
                            })))
                        ]
                    ];
                    if (@as(i32, p.*) != @as(i32, 0)) {
                        if (stbi__jpeg_get_bit(j) != 0) if ((@as(i32, p.*) & @as(i32, bit)) == @as(i32, 0)) {
                            if (@as(i32, p.*) > @as(i32, 0)) {
                                {
                                    const ref = &p.*;
                                    ref.* = @truncate(@as(i32, ref.*) + @as(i32, bit));
                                }
                            } else {
                                {
                                    const ref = &p.*;
                                    ref.* = @truncate(@as(i32, ref.*) - @as(i32, bit));
                                }
                            }
                        };
                    } else {
                        if (r == @as(i32, 0)) {
                            p.* = @truncate(s);
                            break;
                        }
                        r -= 1;
                    }
                }
                if (!(k <= j.*.spec_end)) break;
            }
        }
    }
    return 1;
}
fn stbi__clamp(x: i32) callconv(.c) u8 {
    if (@as(u32, @bitCast(@as(i32, x))) > @as(u32, 255)) {
        if (x < @as(i32, 0)) return 0;
        if (x > @as(i32, 255)) return 255;
    }
    return @bitCast(@as(i8, @truncate(x)));
}
inline fn idctF2F(comptime x: f32) @Vector(8, i32) {
    return @splat(@as(i32, @intFromFloat(@as(f64, @floatCast(x * @as(f32, 4096.0))) + 0.5)));
}

inline fn transpose8x8(rows: [8]@Vector(8, i32)) [8]@Vector(8, i32) {
    const V8 = @Vector(8, i32);
    // Stage 1: interleave adjacent pairs (2x2 blocks)
    const t0 = @shuffle(i32, rows[0], rows[1], [8]i32{ 0, ~@as(i32, 0), 1, ~@as(i32, 1), 4, ~@as(i32, 4), 5, ~@as(i32, 5) });
    const t1 = @shuffle(i32, rows[0], rows[1], [8]i32{ 2, ~@as(i32, 2), 3, ~@as(i32, 3), 6, ~@as(i32, 6), 7, ~@as(i32, 7) });
    const t2 = @shuffle(i32, rows[2], rows[3], [8]i32{ 0, ~@as(i32, 0), 1, ~@as(i32, 1), 4, ~@as(i32, 4), 5, ~@as(i32, 5) });
    const t3 = @shuffle(i32, rows[2], rows[3], [8]i32{ 2, ~@as(i32, 2), 3, ~@as(i32, 3), 6, ~@as(i32, 6), 7, ~@as(i32, 7) });
    const t4 = @shuffle(i32, rows[4], rows[5], [8]i32{ 0, ~@as(i32, 0), 1, ~@as(i32, 1), 4, ~@as(i32, 4), 5, ~@as(i32, 5) });
    const t5 = @shuffle(i32, rows[4], rows[5], [8]i32{ 2, ~@as(i32, 2), 3, ~@as(i32, 3), 6, ~@as(i32, 6), 7, ~@as(i32, 7) });
    const t6 = @shuffle(i32, rows[6], rows[7], [8]i32{ 0, ~@as(i32, 0), 1, ~@as(i32, 1), 4, ~@as(i32, 4), 5, ~@as(i32, 5) });
    const t7 = @shuffle(i32, rows[6], rows[7], [8]i32{ 2, ~@as(i32, 2), 3, ~@as(i32, 3), 6, ~@as(i32, 6), 7, ~@as(i32, 7) });
    // Stage 2: interleave 2-element groups (4x4 blocks)
    const s0 = @shuffle(i32, t0, t2, [8]i32{ 0, 1, ~@as(i32, 0), ~@as(i32, 1), 4, 5, ~@as(i32, 4), ~@as(i32, 5) });
    const s1 = @shuffle(i32, t0, t2, [8]i32{ 2, 3, ~@as(i32, 2), ~@as(i32, 3), 6, 7, ~@as(i32, 6), ~@as(i32, 7) });
    const s2 = @shuffle(i32, t1, t3, [8]i32{ 0, 1, ~@as(i32, 0), ~@as(i32, 1), 4, 5, ~@as(i32, 4), ~@as(i32, 5) });
    const s3 = @shuffle(i32, t1, t3, [8]i32{ 2, 3, ~@as(i32, 2), ~@as(i32, 3), 6, 7, ~@as(i32, 6), ~@as(i32, 7) });
    const s4 = @shuffle(i32, t4, t6, [8]i32{ 0, 1, ~@as(i32, 0), ~@as(i32, 1), 4, 5, ~@as(i32, 4), ~@as(i32, 5) });
    const s5 = @shuffle(i32, t4, t6, [8]i32{ 2, 3, ~@as(i32, 2), ~@as(i32, 3), 6, 7, ~@as(i32, 6), ~@as(i32, 7) });
    const s6 = @shuffle(i32, t5, t7, [8]i32{ 0, 1, ~@as(i32, 0), ~@as(i32, 1), 4, 5, ~@as(i32, 4), ~@as(i32, 5) });
    const s7 = @shuffle(i32, t5, t7, [8]i32{ 2, 3, ~@as(i32, 2), ~@as(i32, 3), 6, 7, ~@as(i32, 6), ~@as(i32, 7) });
    // Stage 3: combine halves (8x8)
    return [8]V8{
        @shuffle(i32, s0, s4, [8]i32{ 0, 1, 2, 3, ~@as(i32, 0), ~@as(i32, 1), ~@as(i32, 2), ~@as(i32, 3) }),
        @shuffle(i32, s1, s5, [8]i32{ 0, 1, 2, 3, ~@as(i32, 0), ~@as(i32, 1), ~@as(i32, 2), ~@as(i32, 3) }),
        @shuffle(i32, s2, s6, [8]i32{ 0, 1, 2, 3, ~@as(i32, 0), ~@as(i32, 1), ~@as(i32, 2), ~@as(i32, 3) }),
        @shuffle(i32, s3, s7, [8]i32{ 0, 1, 2, 3, ~@as(i32, 0), ~@as(i32, 1), ~@as(i32, 2), ~@as(i32, 3) }),
        @shuffle(i32, s0, s4, [8]i32{ 4, 5, 6, 7, ~@as(i32, 4), ~@as(i32, 5), ~@as(i32, 6), ~@as(i32, 7) }),
        @shuffle(i32, s1, s5, [8]i32{ 4, 5, 6, 7, ~@as(i32, 4), ~@as(i32, 5), ~@as(i32, 6), ~@as(i32, 7) }),
        @shuffle(i32, s2, s6, [8]i32{ 4, 5, 6, 7, ~@as(i32, 4), ~@as(i32, 5), ~@as(i32, 6), ~@as(i32, 7) }),
        @shuffle(i32, s3, s7, [8]i32{ 4, 5, 6, 7, ~@as(i32, 4), ~@as(i32, 5), ~@as(i32, 6), ~@as(i32, 7) }),
    };
}

inline fn idct1D(s0: @Vector(8, i32), s1: @Vector(8, i32), s2: @Vector(8, i32), s3: @Vector(8, i32), s4: @Vector(8, i32), s5: @Vector(8, i32), s6: @Vector(8, i32), s7: @Vector(8, i32), comptime bias: i32, comptime shift: u5) [8]@Vector(8, i32) {
    const V8 = @Vector(8, i32);
    const p1_e = (s2 +% s6) *% idctF2F(0.541196);
    const t2_e = p1_e +% s6 *% idctF2F(-1.847759);
    const t3_e = p1_e +% s2 *% idctF2F(0.765367);
    const t0_e = (s0 +% s4) *% @as(V8, @splat(4096));
    const t1_e = (s0 -% s4) *% @as(V8, @splat(4096));
    const b: V8 = @splat(bias);
    const x0 = t0_e +% t3_e +% b;
    const x3 = t0_e -% t3_e +% b;
    const x1 = t1_e +% t2_e +% b;
    const x2 = t1_e -% t2_e +% b;

    const p3 = s7 +% s3;
    const p4 = s5 +% s1;
    const p1 = s7 +% s1;
    const p2 = s5 +% s3;
    const p5 = (p3 +% p4) *% idctF2F(1.175876);
    var t0 = s7 *% idctF2F(0.298631);
    var t1 = s5 *% idctF2F(2.05312);
    var t2 = s3 *% idctF2F(3.072711);
    var t3 = s1 *% idctF2F(1.501321);
    const p1_o = p5 +% p1 *% idctF2F(-0.899976);
    const p2_o = p5 +% p2 *% idctF2F(-2.562916);
    const p3_o = p3 *% idctF2F(-1.961571);
    const p4_o = p4 *% idctF2F(-0.390181);
    t3 +%= p1_o +% p4_o;
    t2 +%= p2_o +% p3_o;
    t1 +%= p2_o +% p4_o;
    t0 +%= p1_o +% p3_o;

    const sh: @Vector(8, u5) = @splat(shift);
    return .{
        (x0 +% t3) >> sh,
        (x1 +% t2) >> sh,
        (x2 +% t1) >> sh,
        (x3 +% t0) >> sh,
        (x3 -% t0) >> sh,
        (x2 -% t1) >> sh,
        (x1 -% t2) >> sh,
        (x0 -% t3) >> sh,
    };
}

fn stbi__idct_block(out: [*c]u8, out_stride: i32, data: [*c]i16) callconv(.c) void {
    const V8 = @Vector(8, i32);
    const d: [*]const i16 = @ptrCast(data);
    var r: [8]V8 = undefined;
    inline for (0..8) |i| {
        const row16: @Vector(8, i16) = d[i * 8 ..][0..8].*;
        r[i] = row16;
    }
    // Pass 1: 8 columns in parallel (matches scalar DC-only shortcut bit-for-bit).
    const ac_or = r[1] | r[2] | r[3] | r[4] | r[5] | r[6] | r[7];
    const col_out = if (@reduce(.Or, ac_or) == 0) blk: {
        const dc = r[0] *% @as(V8, @splat(4));
        break :blk [8]V8{ dc, dc, dc, dc, dc, dc, dc, dc };
    } else blk: {
        const raw = idct1D(r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7], 512, 10);
        const dc = r[0] *% @as(V8, @splat(4));
        const is_dc = ac_or == @as(V8, @splat(0));
        var masked: [8]V8 = undefined;
        inline for (0..8) |i| masked[i] = @select(i32, is_dc, dc, raw[i]);
        break :blk masked;
    };
    // Transpose 8x8 so pass 2 operates on all 8 rows in parallel.
    const t = transpose8x8(col_out);
    const row_out = transpose8x8(idct1D(t[0], t[1], t[2], t[3], t[4], t[5], t[6], t[7], 65536 + (128 << 17), 17));
    const zero: V8 = @splat(0);
    const maxv: V8 = @splat(255);
    const stride: usize = @intCast(out_stride);
    var o: [*]u8 = @ptrCast(out);
    inline for (0..8) |i| {
        const clamped = @min(@max(row_out[i], zero), maxv);
        const bytes: @Vector(8, u8) = @intCast(clamped);
        o[0..8].* = bytes;
        o += stride;
    }
}
fn stbi__get_marker(j: [*c]stbi__jpeg) callconv(.c) u8 {
    var x: u8 = undefined;
    if (@as(i32, j.*.marker) != STBI__MARKER_none) {
        x = j.*.marker;
        j.*.marker = STBI__MARKER_none;
        return x;
    }
    x = stbi__get8(j.*.s);
    if (@as(i32, x) != @as(i32, 255)) return STBI__MARKER_none;
    while (@as(i32, x) == @as(i32, 255)) {
        x = stbi__get8(j.*.s);
    }
    return x;
}
fn stbi__jpeg_reset(j: [*c]stbi__jpeg) callconv(.c) void {
    j.*.code_bits = 0;
    j.*.code_buffer = 0;
    j.*.nomore = 0;
    j.*.img_comp[@as(i32, 0)].dc_pred = blk: {
        const tmp = blk_1: {
            const tmp_2 = blk_2: {
                const tmp_3 = @as(i32, 0);
                j.*.img_comp[@as(i32, 3)].dc_pred = tmp_3;
                break :blk_2 tmp_3;
            };
            j.*.img_comp[@as(i32, 2)].dc_pred = tmp_2;
            break :blk_1 tmp_2;
        };
        j.*.img_comp[@as(i32, 1)].dc_pred = tmp;
        break :blk tmp;
    };
    j.*.marker = STBI__MARKER_none;
    j.*.todo = if (j.*.restart_interval != 0) j.*.restart_interval else @as(i32, 2147483647);
    j.*.eob_run = 0;
}
fn stbi__parse_entropy_coded_data(z: [*c]stbi__jpeg) callconv(.c) i32 {
    stbi__jpeg_reset(z);
    if (!(z.*.progressive != 0)) {
        if (z.*.scan_n == @as(i32, 1)) {
            var i: i32 = undefined;
            var j: i32 = undefined;
            var data: [64]i16 = undefined;
            const n: i32 = z.*.order[@as(i32, 0)];
            const w: i32 = (z.*.img_comp[@bitCast(@as(isize, @intCast(n)))].x + @as(i32, 7)) >> @intCast(@as(i32, 3));
            const h: i32 = (z.*.img_comp[@bitCast(@as(isize, @intCast(n)))].y + @as(i32, 7)) >> @intCast(@as(i32, 3));
            {
                j = 0;
                while (j < h) : (j += 1) {
                    {
                        i = 0;
                        while (i < w) : (i += 1) {
                            const ha: i32 = z.*.img_comp[@bitCast(@as(isize, @intCast(n)))].ha;
                            if (!(stbi__jpeg_decode_block(z, @ptrCast(@alignCast(&data)), @as([*c]stbi__huffman, @ptrCast(@alignCast(&z.*.huff_dc))) + @as(usize, @bitCast(@as(isize, @intCast(z.*.img_comp[@bitCast(@as(isize, @intCast(n)))].hd)))), @as([*c]stbi__huffman, @ptrCast(@alignCast(&z.*.huff_ac))) + @as(usize, @bitCast(@as(isize, @intCast(ha)))), @ptrCast(@alignCast(&z.*.fast_ac[@bitCast(@as(isize, @intCast(ha)))])), n, @ptrCast(@alignCast(&z.*.dequant[@bitCast(@as(isize, @intCast(z.*.img_comp[@bitCast(@as(isize, @intCast(n)))].tq)))]))) != 0)) return 0;
                            z.*.idct_block_kernel.?((z.*.img_comp[@bitCast(@as(isize, @intCast(n)))].data + @as(usize, @bitCast(@as(isize, @intCast((z.*.img_comp[@bitCast(@as(isize, @intCast(n)))].w2 * j) * @as(i32, 8)))))) + @as(usize, @bitCast(@as(isize, @intCast(i * @as(i32, 8))))), z.*.img_comp[@bitCast(@as(isize, @intCast(n)))].w2, @ptrCast(@alignCast(&data)));
                            if ((blk: {
                                const ref = &z.*.todo;
                                ref.* -= 1;
                                break :blk ref.*;
                            }) <= @as(i32, 0)) {
                                if (z.*.code_bits < @as(i32, 24)) {
                                    stbi__grow_buffer_unsafe(z);
                                }
                                if (!((@as(i32, z.*.marker) >= @as(i32, 208)) and (@as(i32, z.*.marker) <= @as(i32, 215)))) return 1;
                                stbi__jpeg_reset(z);
                            }
                        }
                    }
                }
            }
            return 1;
        } else {
            var i: i32 = undefined;
            var j: i32 = undefined;
            var k: i32 = undefined;
            var x: i32 = undefined;
            var y: i32 = undefined;
            var data: [64]i16 = undefined;
            {
                j = 0;
                while (j < z.*.img_mcu_y) : (j += 1) {
                    {
                        i = 0;
                        while (i < z.*.img_mcu_x) : (i += 1) {
                            {
                                k = 0;
                                while (k < z.*.scan_n) : (k += 1) {
                                    const n: i32 = z.*.order[@bitCast(@as(isize, @intCast(k)))];
                                    {
                                        y = 0;
                                        while (y < z.*.img_comp[@bitCast(@as(isize, @intCast(n)))].v) : (y += 1) {
                                            {
                                                x = 0;
                                                while (x < z.*.img_comp[@bitCast(@as(isize, @intCast(n)))].h) : (x += 1) {
                                                    const x2: i32 = ((i * z.*.img_comp[@bitCast(@as(isize, @intCast(n)))].h) + x) * @as(i32, 8);
                                                    const y2: i32 = ((j * z.*.img_comp[@bitCast(@as(isize, @intCast(n)))].v) + y) * @as(i32, 8);
                                                    const ha: i32 = z.*.img_comp[@bitCast(@as(isize, @intCast(n)))].ha;
                                                    if (!(stbi__jpeg_decode_block(z, @ptrCast(@alignCast(&data)), @as([*c]stbi__huffman, @ptrCast(@alignCast(&z.*.huff_dc))) + @as(usize, @bitCast(@as(isize, @intCast(z.*.img_comp[@bitCast(@as(isize, @intCast(n)))].hd)))), @as([*c]stbi__huffman, @ptrCast(@alignCast(&z.*.huff_ac))) + @as(usize, @bitCast(@as(isize, @intCast(ha)))), @ptrCast(@alignCast(&z.*.fast_ac[@bitCast(@as(isize, @intCast(ha)))])), n, @ptrCast(@alignCast(&z.*.dequant[@bitCast(@as(isize, @intCast(z.*.img_comp[@bitCast(@as(isize, @intCast(n)))].tq)))]))) != 0)) return 0;
                                                    z.*.idct_block_kernel.?((z.*.img_comp[@bitCast(@as(isize, @intCast(n)))].data + @as(usize, @bitCast(@as(isize, @intCast(z.*.img_comp[@bitCast(@as(isize, @intCast(n)))].w2 * y2))))) + @as(usize, @bitCast(@as(isize, @intCast(x2)))), z.*.img_comp[@bitCast(@as(isize, @intCast(n)))].w2, @ptrCast(@alignCast(&data)));
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                            if ((blk: {
                                const ref = &z.*.todo;
                                ref.* -= 1;
                                break :blk ref.*;
                            }) <= @as(i32, 0)) {
                                if (z.*.code_bits < @as(i32, 24)) {
                                    stbi__grow_buffer_unsafe(z);
                                }
                                if (!((@as(i32, z.*.marker) >= @as(i32, 208)) and (@as(i32, z.*.marker) <= @as(i32, 215)))) return 1;
                                stbi__jpeg_reset(z);
                            }
                        }
                    }
                }
            }
            return 1;
        }
    } else {
        if (z.*.scan_n == @as(i32, 1)) {
            var i: i32 = undefined;
            var j: i32 = undefined;
            const n: i32 = z.*.order[@as(i32, 0)];
            const w: i32 = (z.*.img_comp[@bitCast(@as(isize, @intCast(n)))].x + @as(i32, 7)) >> @intCast(@as(i32, 3));
            const h: i32 = (z.*.img_comp[@bitCast(@as(isize, @intCast(n)))].y + @as(i32, 7)) >> @intCast(@as(i32, 3));
            {
                j = 0;
                while (j < h) : (j += 1) {
                    {
                        i = 0;
                        while (i < w) : (i += 1) {
                            const data: [*c]i16 = z.*.img_comp[@bitCast(@as(isize, @intCast(n)))].coeff + @as(usize, @bitCast(@as(isize, @intCast(@as(i32, 64) * (i + (j * z.*.img_comp[@bitCast(@as(isize, @intCast(n)))].coeff_w))))));
                            if (z.*.spec_start == @as(i32, 0)) {
                                if (!(stbi__jpeg_decode_block_prog_dc(z, data, &z.*.huff_dc[@bitCast(@as(isize, @intCast(z.*.img_comp[@bitCast(@as(isize, @intCast(n)))].hd)))], n) != 0)) return 0;
                            } else {
                                const ha: i32 = z.*.img_comp[@bitCast(@as(isize, @intCast(n)))].ha;
                                if (!(stbi__jpeg_decode_block_prog_ac(z, data, &z.*.huff_ac[@bitCast(@as(isize, @intCast(ha)))], @ptrCast(@alignCast(&z.*.fast_ac[@bitCast(@as(isize, @intCast(ha)))]))) != 0)) return 0;
                            }
                            if ((blk: {
                                const ref = &z.*.todo;
                                ref.* -= 1;
                                break :blk ref.*;
                            }) <= @as(i32, 0)) {
                                if (z.*.code_bits < @as(i32, 24)) {
                                    stbi__grow_buffer_unsafe(z);
                                }
                                if (!((@as(i32, z.*.marker) >= @as(i32, 208)) and (@as(i32, z.*.marker) <= @as(i32, 215)))) return 1;
                                stbi__jpeg_reset(z);
                            }
                        }
                    }
                }
            }
            return 1;
        } else {
            var i: i32 = undefined;
            var j: i32 = undefined;
            var k: i32 = undefined;
            var x: i32 = undefined;
            var y: i32 = undefined;
            {
                j = 0;
                while (j < z.*.img_mcu_y) : (j += 1) {
                    {
                        i = 0;
                        while (i < z.*.img_mcu_x) : (i += 1) {
                            {
                                k = 0;
                                while (k < z.*.scan_n) : (k += 1) {
                                    const n: i32 = z.*.order[@bitCast(@as(isize, @intCast(k)))];
                                    {
                                        y = 0;
                                        while (y < z.*.img_comp[@bitCast(@as(isize, @intCast(n)))].v) : (y += 1) {
                                            {
                                                x = 0;
                                                while (x < z.*.img_comp[@bitCast(@as(isize, @intCast(n)))].h) : (x += 1) {
                                                    const x2: i32 = (i * z.*.img_comp[@bitCast(@as(isize, @intCast(n)))].h) + x;
                                                    const y2: i32 = (j * z.*.img_comp[@bitCast(@as(isize, @intCast(n)))].v) + y;
                                                    const data: [*c]i16 = z.*.img_comp[@bitCast(@as(isize, @intCast(n)))].coeff + @as(usize, @bitCast(@as(isize, @intCast(@as(i32, 64) * (x2 + (y2 * z.*.img_comp[@bitCast(@as(isize, @intCast(n)))].coeff_w))))));
                                                    if (!(stbi__jpeg_decode_block_prog_dc(z, data, &z.*.huff_dc[@bitCast(@as(isize, @intCast(z.*.img_comp[@bitCast(@as(isize, @intCast(n)))].hd)))], n) != 0)) return 0;
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                            if ((blk: {
                                const ref = &z.*.todo;
                                ref.* -= 1;
                                break :blk ref.*;
                            }) <= @as(i32, 0)) {
                                if (z.*.code_bits < @as(i32, 24)) {
                                    stbi__grow_buffer_unsafe(z);
                                }
                                if (!((@as(i32, z.*.marker) >= @as(i32, 208)) and (@as(i32, z.*.marker) <= @as(i32, 215)))) return 1;
                                stbi__jpeg_reset(z);
                            }
                        }
                    }
                }
            }
            return 1;
        }
    }
    return undefined;
}
fn stbi__jpeg_dequantize(arg_data: [*c]i16, dequant: [*c]u16) callconv(.c) void {
    var data = arg_data;
    var i: i32 = undefined;
    {
        i = 0;
        while (i < @as(i32, 64)) : (i += 1) {
            {
                const ref = &data[@bitCast(@as(isize, @intCast(i)))];
                ref.* = @truncate(@as(i32, ref.*) * @as(i32, dequant[@bitCast(@as(isize, @intCast(i)))]));
            }
        }
    }
}
fn stbi__jpeg_finish(z: [*c]stbi__jpeg) callconv(.c) void {
    if (z.*.progressive != 0) {
        var i: i32 = undefined;
        var j: i32 = undefined;
        var n: i32 = undefined;
        {
            n = 0;
            while (n < z.*.s.*.img_n) : (n += 1) {
                const w: i32 = (z.*.img_comp[@bitCast(@as(isize, @intCast(n)))].x + @as(i32, 7)) >> @intCast(@as(i32, 3));
                const h: i32 = (z.*.img_comp[@bitCast(@as(isize, @intCast(n)))].y + @as(i32, 7)) >> @intCast(@as(i32, 3));
                {
                    j = 0;
                    while (j < h) : (j += 1) {
                        {
                            i = 0;
                            while (i < w) : (i += 1) {
                                const data: [*c]i16 = z.*.img_comp[@bitCast(@as(isize, @intCast(n)))].coeff + @as(usize, @bitCast(@as(isize, @intCast(@as(i32, 64) * (i + (j * z.*.img_comp[@bitCast(@as(isize, @intCast(n)))].coeff_w))))));
                                stbi__jpeg_dequantize(data, @ptrCast(@alignCast(&z.*.dequant[@bitCast(@as(isize, @intCast(z.*.img_comp[@bitCast(@as(isize, @intCast(n)))].tq)))])));
                                z.*.idct_block_kernel.?((z.*.img_comp[@bitCast(@as(isize, @intCast(n)))].data + @as(usize, @bitCast(@as(isize, @intCast((z.*.img_comp[@bitCast(@as(isize, @intCast(n)))].w2 * j) * @as(i32, 8)))))) + @as(usize, @bitCast(@as(isize, @intCast(i * @as(i32, 8))))), z.*.img_comp[@bitCast(@as(isize, @intCast(n)))].w2, data);
                            }
                        }
                    }
                }
            }
        }
    }
}
fn stbi__process_marker(z: [*c]stbi__jpeg, m: i32) callconv(.c) i32 {
    var L: i32 = undefined;
    {
        switch (m) {
            STBI__MARKER_none => {
                return stbi__err("expected marker");
            },
            @as(i32, 221) => {
                if (stbi__get16be(z.*.s) != @as(i32, 4)) return stbi__err("bad DRI len");
                z.*.restart_interval = stbi__get16be(z.*.s);
                return 1;
            },
            @as(i32, 219) => {
                L = stbi__get16be(z.*.s) - @as(i32, 2);
                while (L > @as(i32, 0)) {
                    const q: i32 = stbi__get8(z.*.s);
                    const p: i32 = q >> @intCast(@as(i32, 4));
                    const sixteen: i32 = @intFromBool(p != @as(i32, 0));
                    const t: i32 = q & @as(i32, 15);
                    var i: i32 = undefined;
                    if ((p != @as(i32, 0)) and (p != @as(i32, 1))) return stbi__err("bad DQT type");
                    if (t > @as(i32, 3)) return stbi__err("bad DQT table");
                    {
                        i = 0;
                        while (i < @as(i32, 64)) : (i += 1) {
                            z.*.dequant[@bitCast(@as(isize, @intCast(t)))][stbi__jpeg_dezigzag[@bitCast(@as(isize, @intCast(i)))]] = @bitCast(@as(i16, @truncate(if (sixteen != 0) stbi__get16be(z.*.s) else @as(i32, stbi__get8(z.*.s)))));
                        }
                    }
                    L -= if (sixteen != 0) @as(i32, 129) else @as(i32, 65);
                }
                return @intFromBool(L == @as(i32, 0));
            },
            @as(i32, 196) => {
                L = stbi__get16be(z.*.s) - @as(i32, 2);
                while (L > @as(i32, 0)) {
                    var v: [*c]u8 = undefined;
                    var sizes: [16]i32 = undefined;
                    var i: i32 = undefined;
                    var n: i32 = 0;
                    const q: i32 = stbi__get8(z.*.s);
                    const tc: i32 = q >> @intCast(@as(i32, 4));
                    const th: i32 = q & @as(i32, 15);
                    if ((tc > @as(i32, 1)) or (th > @as(i32, 3))) return stbi__err("bad DHT header");
                    {
                        i = 0;
                        while (i < @as(i32, 16)) : (i += 1) {
                            sizes[@bitCast(@as(isize, @intCast(i)))] = stbi__get8(z.*.s);
                            n += sizes[@bitCast(@as(isize, @intCast(i)))];
                        }
                    }
                    if (n > @as(i32, 256)) return stbi__err("bad DHT header");
                    L -= 17;
                    if (tc == @as(i32, 0)) {
                        if (!(stbi__build_huffman(@as([*c]stbi__huffman, @ptrCast(@alignCast(&z.*.huff_dc))) + @as(usize, @bitCast(@as(isize, @intCast(th)))), @ptrCast(@alignCast(&sizes))) != 0)) return 0;
                        v = @ptrCast(@alignCast(&z.*.huff_dc[@bitCast(@as(isize, @intCast(th)))].values));
                    } else {
                        if (!(stbi__build_huffman(@as([*c]stbi__huffman, @ptrCast(@alignCast(&z.*.huff_ac))) + @as(usize, @bitCast(@as(isize, @intCast(th)))), @ptrCast(@alignCast(&sizes))) != 0)) return 0;
                        v = @ptrCast(@alignCast(&z.*.huff_ac[@bitCast(@as(isize, @intCast(th)))].values));
                    }
                    {
                        i = 0;
                        while (i < n) : (i += 1) {
                            v[@bitCast(@as(isize, @intCast(i)))] = stbi__get8(z.*.s);
                        }
                    }
                    if (tc != @as(i32, 0)) {
                        stbi__build_fast_ac(@ptrCast(@alignCast(&z.*.fast_ac[@bitCast(@as(isize, @intCast(th)))])), @as([*c]stbi__huffman, @ptrCast(@alignCast(&z.*.huff_ac))) + @as(usize, @bitCast(@as(isize, @intCast(th)))));
                    }
                    L -= n;
                }
                return @intFromBool(L == @as(i32, 0));
            },
            else => {},
        }
    }
    if (((m >= @as(i32, 224)) and (m <= @as(i32, 239))) or (m == @as(i32, 254))) {
        L = stbi__get16be(z.*.s);
        if (L < @as(i32, 2)) {
            if (m == @as(i32, 254)) return stbi__err("bad COM len") else return stbi__err("bad APP len");
        }
        L -= 2;
        if ((m == @as(i32, 224)) and (L >= @as(i32, 5))) {
            const static_local_tag = struct {
                const tag: [5]u8 = [5]u8{
                    'J',
                    'F',
                    'I',
                    'F',
                    '\x00',
                };
            };
            var ok: i32 = 1;
            var i: i32 = undefined;
            {
                i = 0;
                while (i < @as(i32, 5)) : (i += 1) if (@as(i32, stbi__get8(z.*.s)) != @as(i32, static_local_tag.tag[@bitCast(@as(isize, @intCast(i)))])) {
                    ok = 0;
                };
            }
            L -= 5;
            if (ok != 0) {
                z.*.jfif = 1;
            }
        } else if ((m == @as(i32, 238)) and (L >= @as(i32, 12))) {
            const static_local_tag = struct {
                const tag: [6]u8 = [6]u8{
                    'A',
                    'd',
                    'o',
                    'b',
                    'e',
                    '\x00',
                };
            };
            var ok: i32 = 1;
            var i: i32 = undefined;
            {
                i = 0;
                while (i < @as(i32, 6)) : (i += 1) if (@as(i32, stbi__get8(z.*.s)) != @as(i32, static_local_tag.tag[@bitCast(@as(isize, @intCast(i)))])) {
                    ok = 0;
                };
            }
            L -= 6;
            if (ok != 0) {
                _ = stbi__get8(z.*.s);
                _ = stbi__get16be(z.*.s);
                _ = stbi__get16be(z.*.s);
                z.*.app14_color_transform = stbi__get8(z.*.s);
                L -= 6;
            }
        }
        stbi__skip(z.*.s, L);
        return 1;
    }
    return stbi__err("unknown marker");
}
fn stbi__process_scan_header(z: [*c]stbi__jpeg) callconv(.c) i32 {
    var i: i32 = undefined;
    const Ls: i32 = stbi__get16be(z.*.s);
    z.*.scan_n = stbi__get8(z.*.s);
    if (((z.*.scan_n < @as(i32, 1)) or (z.*.scan_n > @as(i32, 4))) or (z.*.scan_n > z.*.s.*.img_n)) return stbi__err("bad SOS component count");
    if (Ls != (@as(i32, 6) + (@as(i32, 2) * z.*.scan_n))) return stbi__err("bad SOS len");
    {
        i = 0;
        while (i < z.*.scan_n) : (i += 1) {
            const id: i32 = stbi__get8(z.*.s);
            var which: i32 = undefined;
            const q: i32 = stbi__get8(z.*.s);
            {
                which = 0;
                while (which < z.*.s.*.img_n) : (which += 1) if (z.*.img_comp[@bitCast(@as(isize, @intCast(which)))].id == id) break;
            }
            if (which == z.*.s.*.img_n) return 0;
            z.*.img_comp[@bitCast(@as(isize, @intCast(which)))].hd = q >> @intCast(@as(i32, 4));
            if (z.*.img_comp[@bitCast(@as(isize, @intCast(which)))].hd > @as(i32, 3)) return stbi__err("bad DC huff");
            z.*.img_comp[@bitCast(@as(isize, @intCast(which)))].ha = q & @as(i32, 15);
            if (z.*.img_comp[@bitCast(@as(isize, @intCast(which)))].ha > @as(i32, 3)) return stbi__err("bad AC huff");
            z.*.order[@bitCast(@as(isize, @intCast(i)))] = which;
        }
    }
    {
        var aa: i32 = undefined;
        z.*.spec_start = stbi__get8(z.*.s);
        z.*.spec_end = stbi__get8(z.*.s);
        aa = stbi__get8(z.*.s);
        z.*.succ_high = aa >> @intCast(@as(i32, 4));
        z.*.succ_low = aa & @as(i32, 15);
        if (z.*.progressive != 0) {
            if (((((z.*.spec_start > @as(i32, 63)) or (z.*.spec_end > @as(i32, 63))) or (z.*.spec_start > z.*.spec_end)) or (z.*.succ_high > @as(i32, 13))) or (z.*.succ_low > @as(i32, 13))) return stbi__err("bad SOS");
        } else {
            if (z.*.spec_start != @as(i32, 0)) return stbi__err("bad SOS");
            if ((z.*.succ_high != @as(i32, 0)) or (z.*.succ_low != @as(i32, 0))) return stbi__err("bad SOS");
            z.*.spec_end = 63;
        }
    }
    return 1;
}
fn stbi__free_jpeg_components(z: [*c]stbi__jpeg, ncomp: i32, why: i32) callconv(.c) i32 {
    var i: i32 = undefined;
    {
        i = 0;
        while (i < ncomp) : (i += 1) {
            if (z.*.img_comp[@bitCast(@as(isize, @intCast(i)))].raw_data != null) {
                free(z.*.img_comp[@bitCast(@as(isize, @intCast(i)))].raw_data);
                z.*.img_comp[@bitCast(@as(isize, @intCast(i)))].raw_data = null;
                z.*.img_comp[@bitCast(@as(isize, @intCast(i)))].data = null;
            }
            if (z.*.img_comp[@bitCast(@as(isize, @intCast(i)))].raw_coeff != null) {
                free(z.*.img_comp[@bitCast(@as(isize, @intCast(i)))].raw_coeff);
                z.*.img_comp[@bitCast(@as(isize, @intCast(i)))].raw_coeff = null;
                z.*.img_comp[@bitCast(@as(isize, @intCast(i)))].coeff = null;
            }
            if (z.*.img_comp[@bitCast(@as(isize, @intCast(i)))].linebuf != null) {
                free(@ptrCast(@alignCast(z.*.img_comp[@bitCast(@as(isize, @intCast(i)))].linebuf)));
                z.*.img_comp[@bitCast(@as(isize, @intCast(i)))].linebuf = null;
            }
        }
    }
    return why;
}
fn stbi__process_frame_header(z: [*c]stbi__jpeg, scan: i32) callconv(.c) i32 {
    const s: [*c]stbi__context = z.*.s;
    var Lf: i32 = undefined;
    var p: i32 = undefined;
    var i: i32 = undefined;
    var q: i32 = undefined;
    var h_max: i32 = 1;
    var v_max: i32 = 1;
    var c: i32 = undefined;
    Lf = stbi__get16be(s);
    if (Lf < @as(i32, 11)) return stbi__err("bad SOF len");
    p = stbi__get8(s);
    if (p != @as(i32, 8)) return stbi__err("only 8-bit");
    s.*.img_y = @bitCast(@as(i32, stbi__get16be(s)));
    if (s.*.img_y == @as(u32, 0)) return stbi__err("no header height");
    s.*.img_x = @bitCast(@as(i32, stbi__get16be(s)));
    if (s.*.img_x == @as(u32, 0)) return stbi__err("0 width");
    if (s.*.img_y > @as(u32, @bitCast(@as(i32, @as(i32, 1) << @intCast(@as(i32, 24)))))) return stbi__err("too large");
    if (s.*.img_x > @as(u32, @bitCast(@as(i32, @as(i32, 1) << @intCast(@as(i32, 24)))))) return stbi__err("too large");
    c = stbi__get8(s);
    if (((c != @as(i32, 3)) and (c != @as(i32, 1))) and (c != @as(i32, 4))) return stbi__err("bad component count");
    s.*.img_n = c;
    {
        i = 0;
        while (i < c) : (i += 1) {
            z.*.img_comp[@bitCast(@as(isize, @intCast(i)))].data = null;
            z.*.img_comp[@bitCast(@as(isize, @intCast(i)))].linebuf = null;
        }
    }
    if (Lf != (@as(i32, 8) + (@as(i32, 3) * s.*.img_n))) return stbi__err("bad SOF len");
    z.*.rgb = 0;
    {
        i = 0;
        while (i < s.*.img_n) : (i += 1) {
            const static_local_rgb = struct {
                const rgb: [3]u8 = [3]u8{
                    'R',
                    'G',
                    'B',
                };
            };
            z.*.img_comp[@bitCast(@as(isize, @intCast(i)))].id = stbi__get8(s);
            if ((s.*.img_n == @as(i32, 3)) and (z.*.img_comp[@bitCast(@as(isize, @intCast(i)))].id == @as(i32, static_local_rgb.rgb[@bitCast(@as(isize, @intCast(i)))]))) {
                z.*.rgb += 1;
            }
            q = stbi__get8(s);
            z.*.img_comp[@bitCast(@as(isize, @intCast(i)))].h = q >> @intCast(@as(i32, 4));
            if (!(z.*.img_comp[@bitCast(@as(isize, @intCast(i)))].h != 0) or (z.*.img_comp[@bitCast(@as(isize, @intCast(i)))].h > @as(i32, 4))) return stbi__err("bad H");
            z.*.img_comp[@bitCast(@as(isize, @intCast(i)))].v = q & @as(i32, 15);
            if (!(z.*.img_comp[@bitCast(@as(isize, @intCast(i)))].v != 0) or (z.*.img_comp[@bitCast(@as(isize, @intCast(i)))].v > @as(i32, 4))) return stbi__err("bad V");
            z.*.img_comp[@bitCast(@as(isize, @intCast(i)))].tq = stbi__get8(s);
            if (z.*.img_comp[@bitCast(@as(isize, @intCast(i)))].tq > @as(i32, 3)) return stbi__err("bad TQ");
        }
    }
    if (scan != STBI__SCAN_load) return 1;
    if (!(stbi__mad3sizes_valid(@bitCast(@as(u32, @truncate(s.*.img_x))), @bitCast(@as(u32, @truncate(s.*.img_y))), s.*.img_n, 0) != 0)) return stbi__err("too large");
    {
        i = 0;
        while (i < s.*.img_n) : (i += 1) {
            if (z.*.img_comp[@bitCast(@as(isize, @intCast(i)))].h > h_max) {
                h_max = z.*.img_comp[@bitCast(@as(isize, @intCast(i)))].h;
            }
            if (z.*.img_comp[@bitCast(@as(isize, @intCast(i)))].v > v_max) {
                v_max = z.*.img_comp[@bitCast(@as(isize, @intCast(i)))].v;
            }
        }
    }
    {
        i = 0;
        while (i < s.*.img_n) : (i += 1) {
            if (@rem(h_max, z.*.img_comp[@bitCast(@as(isize, @intCast(i)))].h) != @as(i32, 0)) return stbi__err("bad H");
            if (@rem(v_max, z.*.img_comp[@bitCast(@as(isize, @intCast(i)))].v) != @as(i32, 0)) return stbi__err("bad V");
        }
    }
    z.*.img_h_max = h_max;
    z.*.img_v_max = v_max;
    z.*.img_mcu_w = h_max * @as(i32, 8);
    z.*.img_mcu_h = v_max * @as(i32, 8);
    z.*.img_mcu_x = @bitCast(@as(u32, @truncate(((s.*.img_x +% @as(u32, @bitCast(@as(i32, z.*.img_mcu_w)))) -% @as(u32, 1)) / @as(u32, @bitCast(@as(i32, z.*.img_mcu_w))))));
    z.*.img_mcu_y = @bitCast(@as(u32, @truncate(((s.*.img_y +% @as(u32, @bitCast(@as(i32, z.*.img_mcu_h)))) -% @as(u32, 1)) / @as(u32, @bitCast(@as(i32, z.*.img_mcu_h))))));
    {
        i = 0;
        while (i < s.*.img_n) : (i += 1) {
            z.*.img_comp[@bitCast(@as(isize, @intCast(i)))].x = @bitCast(@as(u32, @truncate((((s.*.img_x *% @as(u32, @bitCast(@as(i32, z.*.img_comp[@bitCast(@as(isize, @intCast(i)))].h)))) +% @as(u32, @bitCast(@as(i32, h_max)))) -% @as(u32, 1)) / @as(u32, @bitCast(@as(i32, h_max))))));
            z.*.img_comp[@bitCast(@as(isize, @intCast(i)))].y = @bitCast(@as(u32, @truncate((((s.*.img_y *% @as(u32, @bitCast(@as(i32, z.*.img_comp[@bitCast(@as(isize, @intCast(i)))].v)))) +% @as(u32, @bitCast(@as(i32, v_max)))) -% @as(u32, 1)) / @as(u32, @bitCast(@as(i32, v_max))))));
            z.*.img_comp[@bitCast(@as(isize, @intCast(i)))].w2 = (z.*.img_mcu_x * z.*.img_comp[@bitCast(@as(isize, @intCast(i)))].h) * @as(i32, 8);
            z.*.img_comp[@bitCast(@as(isize, @intCast(i)))].h2 = (z.*.img_mcu_y * z.*.img_comp[@bitCast(@as(isize, @intCast(i)))].v) * @as(i32, 8);
            z.*.img_comp[@bitCast(@as(isize, @intCast(i)))].coeff = null;
            z.*.img_comp[@bitCast(@as(isize, @intCast(i)))].raw_coeff = null;
            z.*.img_comp[@bitCast(@as(isize, @intCast(i)))].linebuf = null;
            z.*.img_comp[@bitCast(@as(isize, @intCast(i)))].raw_data = stbi__malloc_mad2(z.*.img_comp[@bitCast(@as(isize, @intCast(i)))].w2, z.*.img_comp[@bitCast(@as(isize, @intCast(i)))].h2, 15);
            if (z.*.img_comp[@bitCast(@as(isize, @intCast(i)))].raw_data == @as(?*anyopaque, null)) return stbi__free_jpeg_components(z, i + @as(i32, 1), stbi__err("outofmem"));
            z.*.img_comp[@bitCast(@as(isize, @intCast(i)))].data = @ptrFromInt((@as(usize, @intCast(@intFromPtr(z.*.img_comp[@bitCast(@as(isize, @intCast(i)))].raw_data))) +% @as(usize, 15)) & @as(usize, @bitCast(@as(i64, ~@as(i32, 15)))));
            if (z.*.progressive != 0) {
                z.*.img_comp[@bitCast(@as(isize, @intCast(i)))].coeff_w = @divTrunc(z.*.img_comp[@bitCast(@as(isize, @intCast(i)))].w2, @as(i32, 8));
                z.*.img_comp[@bitCast(@as(isize, @intCast(i)))].coeff_h = @divTrunc(z.*.img_comp[@bitCast(@as(isize, @intCast(i)))].h2, @as(i32, 8));
                z.*.img_comp[@bitCast(@as(isize, @intCast(i)))].raw_coeff = stbi__malloc_mad3(z.*.img_comp[@bitCast(@as(isize, @intCast(i)))].w2, z.*.img_comp[@bitCast(@as(isize, @intCast(i)))].h2, @bitCast(@as(u32, @truncate(@sizeOf(i16)))), 15);
                if (z.*.img_comp[@bitCast(@as(isize, @intCast(i)))].raw_coeff == @as(?*anyopaque, null)) return stbi__free_jpeg_components(z, i + @as(i32, 1), stbi__err("outofmem"));
                z.*.img_comp[@bitCast(@as(isize, @intCast(i)))].coeff = @ptrFromInt((@as(usize, @intCast(@intFromPtr(z.*.img_comp[@bitCast(@as(isize, @intCast(i)))].raw_coeff))) +% @as(usize, 15)) & @as(usize, @bitCast(@as(i64, ~@as(i32, 15)))));
            }
        }
    }
    return 1;
}
fn stbi__decode_jpeg_header(z: [*c]stbi__jpeg, scan: i32) callconv(.c) i32 {
    var m: i32 = undefined;
    z.*.jfif = 0;
    z.*.app14_color_transform = -@as(i32, 1);
    z.*.marker = STBI__MARKER_none;
    m = stbi__get_marker(z);
    if (!(m == @as(i32, 216))) return stbi__err("no SOI");
    if (scan == STBI__SCAN_type) return 1;
    m = stbi__get_marker(z);
    while (!(((m == @as(i32, 192)) or (m == @as(i32, 193))) or (m == @as(i32, 194)))) {
        if (!(stbi__process_marker(z, m) != 0)) return 0;
        m = stbi__get_marker(z);
        while (m == STBI__MARKER_none) {
            if (stbi__at_eof(z.*.s) != 0) return stbi__err("no SOF");
            m = stbi__get_marker(z);
        }
    }
    z.*.progressive = @intFromBool(m == @as(i32, 194));
    if (!(stbi__process_frame_header(z, scan) != 0)) return 0;
    return 1;
}
fn stbi__skip_jpeg_junk_at_end(j: [*c]stbi__jpeg) callconv(.c) u8 {
    while (!(stbi__at_eof(j.*.s) != 0)) {
        var x: u8 = stbi__get8(j.*.s);
        while (@as(i32, x) == @as(i32, 255)) {
            if (stbi__at_eof(j.*.s) != 0) return STBI__MARKER_none;
            x = stbi__get8(j.*.s);
            if ((@as(i32, x) != @as(i32, 0)) and (@as(i32, x) != @as(i32, 255))) {
                return x;
            }
        }
    }
    return STBI__MARKER_none;
}
fn stbi__decode_jpeg_image(j: [*c]stbi__jpeg) callconv(.c) i32 {
    var m: i32 = undefined;
    {
        m = 0;
        while (m < @as(i32, 4)) : (m += 1) {
            j.*.img_comp[@bitCast(@as(isize, @intCast(m)))].raw_data = null;
            j.*.img_comp[@bitCast(@as(isize, @intCast(m)))].raw_coeff = null;
        }
    }
    j.*.restart_interval = 0;
    if (!(stbi__decode_jpeg_header(j, STBI__SCAN_load) != 0)) return 0;
    m = stbi__get_marker(j);
    while (!(m == @as(i32, 217))) {
        if (m == @as(i32, 218)) {
            if (!(stbi__process_scan_header(j) != 0)) return 0;
            if (!(stbi__parse_entropy_coded_data(j) != 0)) return 0;
            if (@as(i32, j.*.marker) == STBI__MARKER_none) {
                j.*.marker = stbi__skip_jpeg_junk_at_end(j);
            }
            m = stbi__get_marker(j);
            if ((m >= @as(i32, 208)) and (m <= @as(i32, 215))) {
                m = stbi__get_marker(j);
            }
        } else if (m == @as(i32, 220)) {
            const Ld: i32 = stbi__get16be(j.*.s);
            const NL: u32 = @bitCast(@as(i32, stbi__get16be(j.*.s)));
            if (Ld != @as(i32, 4)) return stbi__err("bad DNL len");
            if (NL != j.*.s.*.img_y) return stbi__err("bad DNL height");
            m = stbi__get_marker(j);
        } else {
            if (!(stbi__process_marker(j, m) != 0)) return 1;
            m = stbi__get_marker(j);
        }
    }
    if (j.*.progressive != 0) {
        stbi__jpeg_finish(j);
    }
    return 1;
}
const resample_row_func = ?*const fn (out: [*c]u8, in0: [*c]u8, in1: [*c]u8, w: i32, hs: i32) callconv(.c) [*c]u8;
fn resample_row_1(out: [*c]u8, in_near: [*c]u8, in_far: [*c]u8, w: i32, hs: i32) callconv(.c) [*c]u8 {
    _ = @sizeOf(@TypeOf(out));
    _ = @sizeOf(@TypeOf(in_far));
    _ = @sizeOf(@TypeOf(w));
    _ = @sizeOf(@TypeOf(hs));
    return in_near;
}
fn stbi__resample_row_v_2(arg_out: [*c]u8, in_near: [*c]u8, in_far: [*c]u8, w: i32, hs: i32) callconv(.c) [*c]u8 {
    _ = hs;
    const out: [*]u8 = @ptrCast(arg_out);
    const near: [*]const u8 = @ptrCast(in_near);
    const far: [*]const u8 = @ptrCast(in_far);
    const n: usize = @intCast(w);
    const V16 = @Vector(16, u16);
    var i: usize = 0;
    while (i + 16 <= n) : (i += 16) {
        const vn: V16 = near[i..][0..16].*;
        const vf: V16 = far[i..][0..16].*;
        const res = (vn * @as(V16, @splat(3)) + vf + @as(V16, @splat(2))) >> @splat(2);
        out[i..][0..16].* = @as(@Vector(16, u8), @intCast(res));
    }
    while (i < n) : (i += 1) {
        out[i] = @intCast((@as(u16, 3) * near[i] + far[i] + 2) >> 2);
    }
    return arg_out;
}
fn stbi__resample_row_h_2(arg_out: [*c]u8, in_near: [*c]u8, in_far: [*c]u8, w: i32, hs: i32) callconv(.c) [*c]u8 {
    _ = in_far;
    _ = hs;
    const out: [*]u8 = @ptrCast(arg_out);
    const input: [*]const u8 = @ptrCast(in_near);
    const n: usize = @intCast(w);
    if (n == 1) {
        out[0] = input[0];
        out[1] = input[0];
        return arg_out;
    }
    out[0] = input[0];
    out[1] = @intCast((@as(u16, input[0]) * 3 + input[1] + 2) >> 2);
    const V16 = @Vector(16, u16);
    const interleave = comptime blk: {
        var mask: [32]i32 = undefined;
        for (0..16) |k| {
            mask[k * 2] = @intCast(k);
            mask[k * 2 + 1] = ~@as(i32, @intCast(k));
        }
        break :blk mask;
    };
    var i: usize = 1;
    while (i + 16 < n) : (i += 16) {
        const cur: V16 = input[i..][0..16].*;
        const prev: V16 = input[i - 1 ..][0..16].*;
        const next: V16 = input[i + 1 ..][0..16].*;
        const base = cur * @as(V16, @splat(3)) + @as(V16, @splat(2));
        const even: @Vector(16, u8) = @intCast((base + prev) >> @splat(2));
        const odd: @Vector(16, u8) = @intCast((base + next) >> @splat(2));
        out[i * 2 ..][0..32].* = @shuffle(u8, even, odd, interleave);
    }
    while (i < n - 1) : (i += 1) {
        const base: u16 = @as(u16, 3) * input[i] + 2;
        out[i * 2] = @intCast((base + input[i - 1]) >> 2);
        out[i * 2 + 1] = @intCast((base + input[i + 1]) >> 2);
    }
    out[(n - 1) * 2] = @intCast((@as(u16, input[n - 2]) * 3 + input[n - 1] + 2) >> 2);
    out[(n - 1) * 2 + 1] = input[n - 1];
    return arg_out;
}
fn stbi__resample_row_hv_2(arg_out: [*c]u8, in_near: [*c]u8, in_far: [*c]u8, w: i32, hs: i32) callconv(.c) [*c]u8 {
    _ = hs;
    const out: [*]u8 = @ptrCast(arg_out);
    const near: [*]const u8 = @ptrCast(in_near);
    const far: [*]const u8 = @ptrCast(in_far);
    const n: usize = @intCast(w);
    if (n == 1) {
        const tmp: u8 = @intCast((@as(u16, 3) * near[0] + far[0] + 2) >> 2);
        out[0] = tmp;
        out[1] = tmp;
        return arg_out;
    }
    var t1: u16 = @as(u16, 3) * near[0] + far[0];
    out[0] = @intCast((t1 + 2) >> 2);
    const V16 = @Vector(16, u16);
    const interleave = comptime blk: {
        var mask: [32]i32 = undefined;
        for (0..16) |k| {
            mask[k * 2] = @intCast(k);
            mask[k * 2 + 1] = ~@as(i32, @intCast(k));
        }
        break :blk mask;
    };
    var i: usize = 1;
    while (i + 16 <= n) : (i += 16) {
        const n0: V16 = near[i - 1 ..][0..16].*;
        const f0: V16 = far[i - 1 ..][0..16].*;
        const n1: V16 = near[i..][0..16].*;
        const f1: V16 = far[i..][0..16].*;
        const vt0 = n0 * @as(V16, @splat(3)) + f0;
        const vt1 = n1 * @as(V16, @splat(3)) + f1;
        const left: @Vector(16, u8) = @intCast((vt0 * @as(V16, @splat(3)) + vt1 + @as(V16, @splat(8))) >> @splat(4));
        const right: @Vector(16, u8) = @intCast((vt1 * @as(V16, @splat(3)) + vt0 + @as(V16, @splat(8))) >> @splat(4));
        out[i * 2 - 1 ..][0..32].* = @shuffle(u8, left, right, interleave);
    }
    t1 = @as(u16, 3) * near[i - 1] + far[i - 1];
    while (i < n) : (i += 1) {
        const t0 = t1;
        t1 = @as(u16, 3) * near[i] + far[i];
        out[i * 2 - 1] = @intCast((3 * t0 + t1 + 8) >> 4);
        out[i * 2] = @intCast((3 * t1 + t0 + 8) >> 4);
    }
    out[n * 2 - 1] = @intCast((t1 + 2) >> 2);
    return arg_out;
}
fn stbi__resample_row_generic(arg_out: [*c]u8, in_near: [*c]u8, in_far: [*c]u8, w: i32, hs: i32) callconv(.c) [*c]u8 {
    var out = arg_out;
    var i: i32 = undefined;
    var j: i32 = undefined;
    _ = @sizeOf(@TypeOf(in_far));
    {
        i = 0;
        while (i < w) : (i += 1) {
            j = 0;
            while (j < hs) : (j += 1) {
                out[@bitCast(@as(isize, @intCast((i * hs) + j)))] = in_near[@bitCast(@as(isize, @intCast(i)))];
            }
        }
    }
    return out;
}
fn stbi__YCbCr_to_RGB_row(arg_out: [*c]u8, y: [*c]const u8, pcb: [*c]const u8, pcr: [*c]const u8, count: i32, step: i32) callconv(.c) void {
    var out: [*]u8 = @ptrCast(arg_out);
    const yp: [*]const u8 = @ptrCast(y);
    const cbp: [*]const u8 = @ptrCast(pcb);
    const crp: [*]const u8 = @ptrCast(pcr);
    const n: usize = @intCast(count);
    const st: usize = @intCast(step);
    const k_r_s: i32 = @as(i32, @intFromFloat((@as(f32, 1.402) * @as(f32, 4096.0)) + @as(f32, 0.5))) << 8;
    const k_g_cr_s: i32 = -(@as(i32, @intFromFloat((@as(f32, 0.71414) * @as(f32, 4096.0)) + @as(f32, 0.5))) << 8);
    const k_g_cb_s: i32 = -(@as(i32, @intFromFloat((@as(f32, 0.34414) * @as(f32, 4096.0)) + @as(f32, 0.5))) << 8);
    const k_b_s: i32 = @as(i32, @intFromFloat((@as(f32, 1.772) * @as(f32, 4096.0)) + @as(f32, 0.5))) << 8;
    const V8 = @Vector(8, i32);
    const U8 = @Vector(8, u32);
    const k_r: V8 = @splat(k_r_s);
    const k_g_cr: V8 = @splat(k_g_cr_s);
    const k_g_cb: V8 = @splat(k_g_cb_s);
    const k_b: V8 = @splat(k_b_s);
    const c128: V8 = @splat(128);
    const bias: V8 = @splat(@as(i32, 1) << 19);
    const zero: V8 = @splat(0);
    const maxv: V8 = @splat(255);
    const mask: U8 = @splat(0xffff0000);
    var i: usize = 0;
    while (i + 8 <= n) : (i += 8) {
        const vy: V8 = @as(@Vector(8, u8), yp[i..][0..8].*);
        const vcb: V8 = @as(V8, @as(@Vector(8, u8), cbp[i..][0..8].*)) -% c128;
        const vcr: V8 = @as(V8, @as(@Vector(8, u8), crp[i..][0..8].*)) -% c128;
        const y_fixed = (vy << @splat(20)) +% bias;
        const r = @min(@max((y_fixed +% vcr *% k_r) >> @splat(20), zero), maxv);
        const cb_term: V8 = @bitCast(@as(U8, @bitCast(vcb *% k_g_cb)) & mask);
        const g = @min(@max((y_fixed +% vcr *% k_g_cr +% cb_term) >> @splat(20), zero), maxv);
        const b = @min(@max((y_fixed +% vcb *% k_b) >> @splat(20), zero), maxv);
        const rb: [8]u8 = @as(@Vector(8, u8), @intCast(r));
        const gb: [8]u8 = @as(@Vector(8, u8), @intCast(g));
        const bb: [8]u8 = @as(@Vector(8, u8), @intCast(b));
        if (st == 3) {
            inline for (0..8) |k| {
                out[k * 3] = rb[k];
                out[k * 3 + 1] = gb[k];
                out[k * 3 + 2] = bb[k];
            }
            out += 24;
        } else {
            inline for (0..8) |k| {
                out[0] = rb[k];
                out[1] = gb[k];
                out[2] = bb[k];
                out[3] = 255;
                out += st;
            }
        }
    }
    while (i < n) : (i += 1) {
        const y_fixed: i32 = (@as(i32, yp[i]) << 20) +% (@as(i32, 1) << 19);
        const cr: i32 = @as(i32, crp[i]) - 128;
        const cb: i32 = @as(i32, cbp[i]) - 128;
        const r = std.math.clamp((y_fixed +% cr *% k_r_s) >> 20, 0, 255);
        const cb_term: i32 = @bitCast(@as(u32, @bitCast(cb *% k_g_cb_s)) & 0xffff0000);
        const g = std.math.clamp((y_fixed +% cr *% k_g_cr_s +% cb_term) >> 20, 0, 255);
        const b = std.math.clamp((y_fixed +% cb *% k_b_s) >> 20, 0, 255);
        out[0] = @intCast(r);
        out[1] = @intCast(g);
        out[2] = @intCast(b);
        if (st >= 4) out[3] = 255;
        out += st;
    }
}
fn stbi__setup_jpeg(j: [*c]stbi__jpeg) callconv(.c) void {
    j.*.idct_block_kernel = stbi__idct_block;
    j.*.YCbCr_to_RGB_kernel = stbi__YCbCr_to_RGB_row;
    j.*.resample_row_hv_2_kernel = stbi__resample_row_hv_2;
}
fn stbi__cleanup_jpeg(j: [*c]stbi__jpeg) callconv(.c) void {
    _ = stbi__free_jpeg_components(j, j.*.s.*.img_n, 0);
}
const stbi__resample = extern struct {
    resample: resample_row_func = null,
    line0: [*c]u8 = null,
    line1: [*c]u8 = null,
    hs: i32 = 0,
    vs: i32 = 0,
    w_lores: i32 = 0,
    ystep: i32 = 0,
    ypos: i32 = 0,
};
fn stbi__blinn_8x8(x: u8, y: u8) callconv(.c) u8 {
    const t: u32 = @bitCast(@as(i32, (@as(i32, x) * @as(i32, y)) + @as(i32, 128)));
    return @truncate((t +% (t >> @intCast(@as(u32, 8)))) >> @intCast(@as(u32, 8)));
}
fn load_jpeg_image(z: [*c]stbi__jpeg, out_x: [*c]i32, out_y: [*c]i32, comp: [*c]i32, req_comp: i32) callconv(.c) [*c]u8 {
    var n: i32 = undefined;
    var decode_n: i32 = undefined;
    var is_rgb: i32 = undefined;
    z.*.s.*.img_n = 0;
    if ((req_comp < @as(i32, 0)) or (req_comp > @as(i32, 4))) return @as([*c]u8, @ptrFromInt(@as(usize, @intCast(@intFromPtr(if (stbi__err("bad req_comp") != 0) @as(?*anyopaque, null) else @as(?*anyopaque, null))))));
    if (!(stbi__decode_jpeg_image(z) != 0)) {
        stbi__cleanup_jpeg(z);
        return null;
    }
    n = if (req_comp != 0) req_comp else if (z.*.s.*.img_n >= @as(i32, 3)) @as(i32, 3) else @as(i32, 1);
    is_rgb = @intFromBool((z.*.s.*.img_n == @as(i32, 3)) and ((z.*.rgb == @as(i32, 3)) or ((z.*.app14_color_transform == @as(i32, 0)) and !(z.*.jfif != 0))));
    if (((z.*.s.*.img_n == @as(i32, 3)) and (n < @as(i32, 3))) and !(is_rgb != 0)) {
        decode_n = 1;
    } else {
        decode_n = z.*.s.*.img_n;
    }
    if (decode_n <= @as(i32, 0)) {
        stbi__cleanup_jpeg(z);
        return null;
    }
    {
        var k: i32 = undefined;
        var i: u32 = undefined;
        var j: u32 = undefined;
        var output: [*c]u8 = undefined;
        var coutput: [4][*c]u8 = [4][*c]u8{
            null,
            null,
            null,
            null,
        };
        var res_comp: [4]stbi__resample = undefined;
        {
            k = 0;
            while (k < decode_n) : (k += 1) {
                const r: [*c]stbi__resample = &res_comp[@bitCast(@as(isize, @intCast(k)))];
                z.*.img_comp[@bitCast(@as(isize, @intCast(k)))].linebuf = @ptrCast(@alignCast(stbi__malloc(z.*.s.*.img_x +% @as(u32, 3))));
                if (!(z.*.img_comp[@bitCast(@as(isize, @intCast(k)))].linebuf != null)) {
                    stbi__cleanup_jpeg(z);
                    return @as([*c]u8, @ptrFromInt(@as(usize, @intCast(@intFromPtr(if (stbi__err("outofmem") != 0) @as(?*anyopaque, null) else @as(?*anyopaque, null))))));
                }
                r.*.hs = @divTrunc(z.*.img_h_max, z.*.img_comp[@bitCast(@as(isize, @intCast(k)))].h);
                r.*.vs = @divTrunc(z.*.img_v_max, z.*.img_comp[@bitCast(@as(isize, @intCast(k)))].v);
                r.*.ystep = r.*.vs >> @intCast(@as(i32, 1));
                r.*.w_lores = @bitCast(@as(u32, @truncate(((z.*.s.*.img_x +% @as(u32, @bitCast(@as(i32, r.*.hs)))) -% @as(u32, 1)) / @as(u32, @bitCast(@as(i32, r.*.hs))))));
                r.*.ypos = 0;
                r.*.line0 = blk: {
                    const tmp = z.*.img_comp[@bitCast(@as(isize, @intCast(k)))].data;
                    r.*.line1 = tmp;
                    break :blk tmp;
                };
                if ((r.*.hs == @as(i32, 1)) and (r.*.vs == @as(i32, 1))) {
                    r.*.resample = resample_row_1;
                } else if ((r.*.hs == @as(i32, 1)) and (r.*.vs == @as(i32, 2))) {
                    r.*.resample = stbi__resample_row_v_2;
                } else if ((r.*.hs == @as(i32, 2)) and (r.*.vs == @as(i32, 1))) {
                    r.*.resample = stbi__resample_row_h_2;
                } else if ((r.*.hs == @as(i32, 2)) and (r.*.vs == @as(i32, 2))) {
                    r.*.resample = z.*.resample_row_hv_2_kernel;
                } else {
                    r.*.resample = stbi__resample_row_generic;
                }
            }
        }
        output = @ptrCast(@alignCast(stbi__malloc_mad3(n, @bitCast(@as(u32, @truncate(z.*.s.*.img_x))), @bitCast(@as(u32, @truncate(z.*.s.*.img_y))), 1)));
        if (!(output != null)) {
            stbi__cleanup_jpeg(z);
            return @as([*c]u8, @ptrFromInt(@as(usize, @intCast(@intFromPtr(if (stbi__err("outofmem") != 0) @as(?*anyopaque, null) else @as(?*anyopaque, null))))));
        }
        {
            j = 0;
            while (j < z.*.s.*.img_y) : (j +%= 1) {
                var out: [*c]u8 = output + ((@as(u32, @bitCast(@as(i32, n))) *% z.*.s.*.img_x) *% j);
                {
                    k = 0;
                    while (k < decode_n) : (k += 1) {
                        const r: [*c]stbi__resample = &res_comp[@bitCast(@as(isize, @intCast(k)))];
                        const y_bot: i32 = @intFromBool(r.*.ystep >= (r.*.vs >> @intCast(@as(i32, 1))));
                        coutput[@bitCast(@as(isize, @intCast(k)))] = r.*.resample.?(z.*.img_comp[@bitCast(@as(isize, @intCast(k)))].linebuf, if (y_bot != 0) r.*.line1 else r.*.line0, if (y_bot != 0) r.*.line0 else r.*.line1, r.*.w_lores, r.*.hs);
                        if ((blk: {
                            const ref = &r.*.ystep;
                            ref.* += 1;
                            break :blk ref.*;
                        }) >= r.*.vs) {
                            r.*.ystep = 0;
                            r.*.line0 = r.*.line1;
                            if ((blk: {
                                const ref = &r.*.ypos;
                                ref.* += 1;
                                break :blk ref.*;
                            }) < z.*.img_comp[@bitCast(@as(isize, @intCast(k)))].y) {
                                r.*.line1 += @as(usize, @bitCast(@as(isize, @intCast(z.*.img_comp[@bitCast(@as(isize, @intCast(k)))].w2))));
                            }
                        }
                    }
                }
                if (n >= @as(i32, 3)) {
                    const y: [*c]u8 = coutput[@as(i32, 0)];
                    if (z.*.s.*.img_n == @as(i32, 3)) {
                        if (is_rgb != 0) {
                            {
                                i = 0;
                                while (i < z.*.s.*.img_x) : (i +%= 1) {
                                    out[@as(i32, 0)] = y[i];
                                    out[@as(i32, 1)] = coutput[@as(i32, 1)][i];
                                    out[@as(i32, 2)] = coutput[@as(i32, 2)][i];
                                    out[@as(i32, 3)] = 255;
                                    out += @as(usize, @bitCast(@as(isize, @intCast(n))));
                                }
                            }
                        } else {
                            z.*.YCbCr_to_RGB_kernel.?(out, y, coutput[@as(i32, 1)], coutput[@as(i32, 2)], @bitCast(@as(u32, @truncate(z.*.s.*.img_x))), n);
                        }
                    } else if (z.*.s.*.img_n == @as(i32, 4)) {
                        if (z.*.app14_color_transform == @as(i32, 0)) {
                            {
                                i = 0;
                                while (i < z.*.s.*.img_x) : (i +%= 1) {
                                    const m: u8 = coutput[@as(i32, 3)][i];
                                    out[@as(i32, 0)] = stbi__blinn_8x8(coutput[@as(i32, 0)][i], m);
                                    out[@as(i32, 1)] = stbi__blinn_8x8(coutput[@as(i32, 1)][i], m);
                                    out[@as(i32, 2)] = stbi__blinn_8x8(coutput[@as(i32, 2)][i], m);
                                    out[@as(i32, 3)] = 255;
                                    out += @as(usize, @bitCast(@as(isize, @intCast(n))));
                                }
                            }
                        } else if (z.*.app14_color_transform == @as(i32, 2)) {
                            z.*.YCbCr_to_RGB_kernel.?(out, y, coutput[@as(i32, 1)], coutput[@as(i32, 2)], @bitCast(@as(u32, @truncate(z.*.s.*.img_x))), n);
                            {
                                i = 0;
                                while (i < z.*.s.*.img_x) : (i +%= 1) {
                                    const m: u8 = coutput[@as(i32, 3)][i];
                                    out[@as(i32, 0)] = stbi__blinn_8x8(@bitCast(@as(i8, @truncate(@as(i32, 255) - @as(i32, out[@as(i32, 0)])))), m);
                                    out[@as(i32, 1)] = stbi__blinn_8x8(@bitCast(@as(i8, @truncate(@as(i32, 255) - @as(i32, out[@as(i32, 1)])))), m);
                                    out[@as(i32, 2)] = stbi__blinn_8x8(@bitCast(@as(i8, @truncate(@as(i32, 255) - @as(i32, out[@as(i32, 2)])))), m);
                                    out += @as(usize, @bitCast(@as(isize, @intCast(n))));
                                }
                            }
                        } else {
                            z.*.YCbCr_to_RGB_kernel.?(out, y, coutput[@as(i32, 1)], coutput[@as(i32, 2)], @bitCast(@as(u32, @truncate(z.*.s.*.img_x))), n);
                        }
                    } else {
                        i = 0;
                        while (i < z.*.s.*.img_x) : (i +%= 1) {
                            out[@as(i32, 0)] = blk: {
                                const tmp = blk_1: {
                                    const tmp_2 = y[i];
                                    out[@as(i32, 2)] = tmp_2;
                                    break :blk_1 tmp_2;
                                };
                                out[@as(i32, 1)] = tmp;
                                break :blk tmp;
                            };
                            out[@as(i32, 3)] = 255;
                            out += @as(usize, @bitCast(@as(isize, @intCast(n))));
                        }
                    }
                } else {
                    if (is_rgb != 0) {
                        if (n == @as(i32, 1)) {
                            {
                                i = 0;
                                while (i < z.*.s.*.img_x) : (i +%= 1) {
                                    (blk: {
                                        const ref = &out;
                                        const tmp = ref.*;
                                        ref.* += 1;
                                        break :blk tmp;
                                    }).* = stbi__compute_y(coutput[@as(i32, 0)][i], coutput[@as(i32, 1)][i], coutput[@as(i32, 2)][i]);
                                }
                            }
                        } else {
                            {
                                i = 0;
                                while (i < z.*.s.*.img_x) : ({
                                    i +%= 1;
                                    out += @as(usize, @bitCast(@as(isize, @intCast(@as(i32, 2)))));
                                }) {
                                    out[@as(i32, 0)] = stbi__compute_y(coutput[@as(i32, 0)][i], coutput[@as(i32, 1)][i], coutput[@as(i32, 2)][i]);
                                    out[@as(i32, 1)] = 255;
                                }
                            }
                        }
                    } else if ((z.*.s.*.img_n == @as(i32, 4)) and (z.*.app14_color_transform == @as(i32, 0))) {
                        {
                            i = 0;
                            while (i < z.*.s.*.img_x) : (i +%= 1) {
                                const m: u8 = coutput[@as(i32, 3)][i];
                                const r: u8 = stbi__blinn_8x8(coutput[@as(i32, 0)][i], m);
                                const g: u8 = stbi__blinn_8x8(coutput[@as(i32, 1)][i], m);
                                const b: u8 = stbi__blinn_8x8(coutput[@as(i32, 2)][i], m);
                                out[@as(i32, 0)] = stbi__compute_y(r, g, b);
                                out[@as(i32, 1)] = 255;
                                out += @as(usize, @bitCast(@as(isize, @intCast(n))));
                            }
                        }
                    } else if ((z.*.s.*.img_n == @as(i32, 4)) and (z.*.app14_color_transform == @as(i32, 2))) {
                        {
                            i = 0;
                            while (i < z.*.s.*.img_x) : (i +%= 1) {
                                out[@as(i32, 0)] = stbi__blinn_8x8(@bitCast(@as(i8, @truncate(@as(i32, 255) - @as(i32, coutput[@as(i32, 0)][i])))), coutput[@as(i32, 3)][i]);
                                out[@as(i32, 1)] = 255;
                                out += @as(usize, @bitCast(@as(isize, @intCast(n))));
                            }
                        }
                    } else {
                        const y: [*c]u8 = coutput[@as(i32, 0)];
                        if (n == @as(i32, 1)) {
                            {
                                i = 0;
                                while (i < z.*.s.*.img_x) : (i +%= 1) {
                                    out[i] = y[i];
                                }
                            }
                        } else {
                            i = 0;
                            while (i < z.*.s.*.img_x) : (i +%= 1) {
                                (blk: {
                                    const ref = &out;
                                    const tmp = ref.*;
                                    ref.* += 1;
                                    break :blk tmp;
                                }).* = y[i];
                                (blk: {
                                    const ref = &out;
                                    const tmp = ref.*;
                                    ref.* += 1;
                                    break :blk tmp;
                                }).* = 255;
                            }
                        }
                    }
                }
            }
        }
        stbi__cleanup_jpeg(z);
        out_x.* = @bitCast(@as(u32, @truncate(z.*.s.*.img_x)));
        out_y.* = @bitCast(@as(u32, @truncate(z.*.s.*.img_y)));
        if (comp != null) {
            comp.* = if (z.*.s.*.img_n >= @as(i32, 3)) @as(i32, 3) else @as(i32, 1);
        }
        return output;
    }
}
fn stbi__jpeg_load(s: [*c]stbi__context, x: [*c]i32, y: [*c]i32, comp: [*c]i32, req_comp: i32, ri: [*c]stbi__result_info) callconv(.c) ?*anyopaque {
    var result: [*c]u8 = undefined;
    const j: [*c]stbi__jpeg = @ptrCast(@alignCast(stbi__malloc(@sizeOf(stbi__jpeg))));
    if (!(j != null)) return @ptrCast(@alignCast(@as([*c]u8, @ptrFromInt(@as(usize, @intCast(@intFromPtr(if (stbi__err("outofmem") != 0) @as(?*anyopaque, null) else @as(?*anyopaque, null))))))));
    _ = memset(@ptrCast(@alignCast(j)), 0, @sizeOf(stbi__jpeg));
    _ = @sizeOf(@TypeOf(ri));
    j.*.s = s;
    stbi__setup_jpeg(j);
    result = load_jpeg_image(j, x, y, comp, req_comp);
    free(@ptrCast(@alignCast(j)));
    return @ptrCast(@alignCast(result));
}
fn stbi__jpeg_test(s: [*c]stbi__context) callconv(.c) i32 {
    var r: i32 = undefined;
    const j: [*c]stbi__jpeg = @ptrCast(@alignCast(stbi__malloc(@sizeOf(stbi__jpeg))));
    if (!(j != null)) return stbi__err("outofmem");
    _ = memset(@ptrCast(@alignCast(j)), 0, @sizeOf(stbi__jpeg));
    j.*.s = s;
    stbi__setup_jpeg(j);
    r = stbi__decode_jpeg_header(j, STBI__SCAN_type);
    stbi__rewind(s);
    free(@ptrCast(@alignCast(j)));
    return r;
}
const stbi__zhuffman = extern struct {
    fast: [512]u16 = std.mem.zeroes([512]u16),
    firstcode: [16]u16 = std.mem.zeroes([16]u16),
    maxcode: [17]i32 = std.mem.zeroes([17]i32),
    firstsymbol: [16]u16 = std.mem.zeroes([16]u16),
    size: [288]u8 = std.mem.zeroes([288]u8),
    value: [288]u16 = std.mem.zeroes([288]u16),
};
fn stbi__bitreverse16(arg_n: i32) callconv(.c) i32 {
    var n = arg_n;
    n = ((n & @as(i32, 43690)) >> @intCast(@as(i32, 1))) | ((n & @as(i32, 21845)) << @intCast(@as(i32, 1)));
    n = ((n & @as(i32, 52428)) >> @intCast(@as(i32, 2))) | ((n & @as(i32, 13107)) << @intCast(@as(i32, 2)));
    n = ((n & @as(i32, 61680)) >> @intCast(@as(i32, 4))) | ((n & @as(i32, 3855)) << @intCast(@as(i32, 4)));
    n = ((n & @as(i32, 65280)) >> @intCast(@as(i32, 8))) | ((n & @as(i32, 255)) << @intCast(@as(i32, 8)));
    return n;
}
fn stbi__bit_reverse(v: i32, bits: i32) callconv(.c) i32 {
    _ = !!(bits <= @as(i32, 16)) or ((blk: {
        _assert("bits <= 16", "tools\\dataset\\vendor\\stb_image.h", 4118);
        break :blk 0;
    }) != 0);
    return stbi__bitreverse16(v) >> @intCast(@as(i32, 16) - bits);
}
fn stbi__zbuild_huffman(z: [*c]stbi__zhuffman, sizelist: [*c]const u8, num: i32) callconv(.c) i32 {
    var i: i32 = undefined;
    var k: i32 = 0;
    var code: i32 = undefined;
    var next_code: [16]i32 = undefined;
    var sizes: [17]i32 = undefined;
    _ = memset(@ptrCast(@alignCast(@as([*c]i32, @ptrCast(@alignCast(&sizes))))), 0, @sizeOf(@TypeOf(sizes)));
    _ = memset(@ptrCast(@alignCast(@as([*c]u16, @ptrCast(@alignCast(&z.*.fast))))), 0, @sizeOf(@TypeOf(z.*.fast)));
    {
        i = 0;
        while (i < num) : (i += 1) {
            sizes[sizelist[@bitCast(@as(isize, @intCast(i)))]] += 1;
        }
    }
    sizes[@as(i32, 0)] = 0;
    {
        i = 1;
        while (i < @as(i32, 16)) : (i += 1) if (sizes[@bitCast(@as(isize, @intCast(i)))] > (@as(i32, 1) << @intCast(i))) return stbi__err("bad sizes");
    }
    code = 0;
    {
        i = 1;
        while (i < @as(i32, 16)) : (i += 1) {
            next_code[@bitCast(@as(isize, @intCast(i)))] = code;
            z.*.firstcode[@bitCast(@as(isize, @intCast(i)))] = @bitCast(@as(i16, @truncate(code)));
            z.*.firstsymbol[@bitCast(@as(isize, @intCast(i)))] = @bitCast(@as(i16, @truncate(k)));
            code = code + sizes[@bitCast(@as(isize, @intCast(i)))];
            if (sizes[@bitCast(@as(isize, @intCast(i)))] != 0) if ((code - @as(i32, 1)) >= (@as(i32, 1) << @intCast(i))) return stbi__err("bad codelengths");
            z.*.maxcode[@bitCast(@as(isize, @intCast(i)))] = code << @intCast(@as(i32, 16) - i);
            code <<= @intCast(@as(i32, 1));
            k += sizes[@bitCast(@as(isize, @intCast(i)))];
        }
    }
    z.*.maxcode[@as(i32, 16)] = 65536;
    {
        i = 0;
        while (i < num) : (i += 1) {
            const s: i32 = sizelist[@bitCast(@as(isize, @intCast(i)))];
            if (s != 0) {
                const c: i32 = (next_code[@bitCast(@as(isize, @intCast(s)))] - @as(i32, z.*.firstcode[@bitCast(@as(isize, @intCast(s)))])) + @as(i32, z.*.firstsymbol[@bitCast(@as(isize, @intCast(s)))]);
                const fastv: u16 = @bitCast(@as(i16, @truncate((s << @intCast(@as(i32, 9))) | i)));
                z.*.size[@bitCast(@as(isize, @intCast(c)))] = @bitCast(@as(i8, @truncate(s)));
                z.*.value[@bitCast(@as(isize, @intCast(c)))] = @bitCast(@as(i16, @truncate(i)));
                if (s <= STBI__ZFAST_BITS) {
                    var j: i32 = stbi__bit_reverse(next_code[@bitCast(@as(isize, @intCast(s)))], s);
                    while (j < (@as(i32, 1) << @intCast(STBI__ZFAST_BITS))) {
                        z.*.fast[@bitCast(@as(isize, @intCast(j)))] = fastv;
                        j += @as(i32, 1) << @intCast(s);
                    }
                }
                next_code[@bitCast(@as(isize, @intCast(s)))] += 1;
            }
        }
    }
    return 1;
}
const stbi__zbuf = extern struct {
    zbuffer: [*c]u8 = null,
    zbuffer_end: [*c]u8 = null,
    num_bits: i32 = 0,
    hit_zeof_once: i32 = 0,
    code_buffer: u32 = 0,
    zout: [*c]u8 = null,
    zout_start: [*c]u8 = null,
    zout_end: [*c]u8 = null,
    z_expandable: i32 = 0,
    z_length: stbi__zhuffman = std.mem.zeroes(stbi__zhuffman),
    z_distance: stbi__zhuffman = std.mem.zeroes(stbi__zhuffman),
};
fn stbi__zeof(z: [*c]stbi__zbuf) callconv(.c) i32 {
    return @intFromBool(z.*.zbuffer >= z.*.zbuffer_end);
}
fn stbi__zget8(z: [*c]stbi__zbuf) callconv(.c) u8 {
    return @bitCast(@as(i8, @truncate(if (stbi__zeof(z) != 0) @as(i32, 0) else @as(i32, (blk: {
        const ref = &z.*.zbuffer;
        const tmp = ref.*;
        ref.* += 1;
        break :blk tmp;
    }).*))));
}
fn stbi__fill_bits(z: [*c]stbi__zbuf) callconv(.c) void {
    while (true) {
        if (z.*.code_buffer >= (@as(u32, 1) << @intCast(@as(u32, @bitCast(@as(i32, z.*.num_bits)))))) {
            z.*.zbuffer = z.*.zbuffer_end;
            return;
        }
        z.*.code_buffer |= @as(u32, stbi__zget8(z)) << @intCast(@as(u32, @bitCast(@as(i32, z.*.num_bits))));
        z.*.num_bits += 8;
        if (!(z.*.num_bits <= @as(i32, 24))) break;
    }
}
fn stbi__zreceive(z: [*c]stbi__zbuf, n: i32) callconv(.c) u32 {
    var k: u32 = undefined;
    if (z.*.num_bits < n) {
        stbi__fill_bits(z);
    }
    k = z.*.code_buffer & @as(u32, @bitCast(@as(i32, (@as(i32, 1) << @intCast(n)) - @as(i32, 1))));
    z.*.code_buffer >>= @intCast(@as(u32, @bitCast(@as(i32, n))));
    z.*.num_bits -= n;
    return k;
}
fn stbi__zhuffman_decode_slowpath(a: [*c]stbi__zbuf, z: [*c]stbi__zhuffman) callconv(.c) i32 {
    var b: i32 = undefined;
    var s: i32 = undefined;
    var k: i32 = undefined;
    k = stbi__bit_reverse(@bitCast(@as(u32, @truncate(a.*.code_buffer))), 16);
    {
        s = STBI__ZFAST_BITS + @as(i32, 1);
        while (true) : (s += 1) if (k < z.*.maxcode[@bitCast(@as(isize, @intCast(s)))]) break;
    }
    if (s >= @as(i32, 16)) return -@as(i32, 1);
    b = ((k >> @intCast(@as(i32, 16) - s)) - @as(i32, z.*.firstcode[@bitCast(@as(isize, @intCast(s)))])) + @as(i32, z.*.firstsymbol[@bitCast(@as(isize, @intCast(s)))]);
    if (b >= STBI__ZNSYMS) return -@as(i32, 1);
    if (@as(i32, z.*.size[@bitCast(@as(isize, @intCast(b)))]) != s) return -@as(i32, 1);
    a.*.code_buffer >>= @intCast(@as(u32, @bitCast(@as(i32, s))));
    a.*.num_bits -= s;
    return z.*.value[@bitCast(@as(isize, @intCast(b)))];
}
fn stbi__zhuffman_decode(a: [*c]stbi__zbuf, z: [*c]stbi__zhuffman) callconv(.c) i32 {
    var b: i32 = undefined;
    var s: i32 = undefined;
    if (a.*.num_bits < @as(i32, 16)) {
        if (stbi__zeof(a) != 0) {
            if (!(a.*.hit_zeof_once != 0)) {
                a.*.hit_zeof_once = 1;
                a.*.num_bits += 16;
            } else {
                return -@as(i32, 1);
            }
        } else {
            stbi__fill_bits(a);
        }
    }
    b = z.*.fast[a.*.code_buffer & @as(u32, @bitCast(@as(i32, (@as(i32, 1) << @intCast(STBI__ZFAST_BITS)) - @as(i32, 1))))];
    if (b != 0) {
        s = b >> @intCast(@as(i32, 9));
        a.*.code_buffer >>= @intCast(@as(u32, @bitCast(@as(i32, s))));
        a.*.num_bits -= s;
        return b & @as(i32, 511);
    }
    return stbi__zhuffman_decode_slowpath(a, z);
}
fn stbi__zexpand(z: [*c]stbi__zbuf, zout: [*c]u8, n: i32) callconv(.c) i32 {
    var q: [*c]u8 = undefined;
    var cur: u32 = undefined;
    var limit: u32 = undefined;
    var old_limit: u32 = undefined;
    z.*.zout = zout;
    if (!(z.*.z_expandable != 0)) return stbi__err("output buffer limit");
    cur = @bitCast(@as(i32, @truncate(@divExact(@as(i64, @bitCast(@intFromPtr(z.*.zout) -% @intFromPtr(z.*.zout_start))), @sizeOf(u8)))));
    limit = blk: {
        const tmp = @as(u32, @bitCast(@as(i32, @truncate(@divExact(@as(i64, @bitCast(@intFromPtr(z.*.zout_end) -% @intFromPtr(z.*.zout_start))), @sizeOf(u8))))));
        old_limit = tmp;
        break :blk tmp;
    };
    if ((((@as(u32, __INT_MAX__) *% @as(u32, 2)) +% @as(u32, 1)) -% cur) < @as(u32, @bitCast(@as(i32, n)))) return stbi__err("outofmem");
    while ((cur +% @as(u32, @bitCast(@as(i32, n)))) > limit) {
        if (limit > (((@as(u32, __INT_MAX__) *% @as(u32, 2)) +% @as(u32, 1)) / @as(u32, 2))) return stbi__err("outofmem");
        limit *%= 2;
    }
    q = @ptrCast(@alignCast(realloc(@ptrCast(@alignCast(z.*.zout_start)), limit)));
    _ = @sizeOf(@TypeOf(old_limit));
    if (@as(?*anyopaque, @ptrCast(@alignCast(q))) == @as(?*anyopaque, null)) return stbi__err("outofmem");
    z.*.zout_start = q;
    z.*.zout = q + cur;
    z.*.zout_end = q + limit;
    return 1;
}
const stbi__zlength_base: [31]i32 = [31]i32{
    3,  4,  5,  6,  7,  8,  9,  10,  11,  13,  15,  17,  19,  23, 27, 31,
    35, 43, 51, 59, 67, 83, 99, 115, 131, 163, 195, 227, 258, 0,  0,
};
const stbi__zlength_extra: [31]i32 = [31]i32{
    0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2,
    3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0, 0, 0,
};
const stbi__zdist_base: [32]i32 = [32]i32{
    1,   2,   3,   4,   5,    7,    9,    13,   17,   25,   33,   49,    65,    97,    129, 193,
    257, 385, 513, 769, 1025, 1537, 2049, 3073, 4097, 6145, 8193, 12289, 16385, 24577, 0,   0,
};
const stbi__zdist_extra: [32]i32 = [30]i32{
    0, 0, 0, 0, 1, 1, 2,  2,  3,  3,  4,  4,  5,  5,  6, 6,
    7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13,
} ++ @as([2]i32, @splat(0));
fn stbi__parse_huffman_block(a: [*c]stbi__zbuf) callconv(.c) i32 {
    var zout: [*c]u8 = a.*.zout;
    while (true) {
        var z: i32 = stbi__zhuffman_decode(a, &a.*.z_length);
        if (z < @as(i32, 256)) {
            if (z < @as(i32, 0)) return stbi__err("bad huffman code");
            if (zout >= a.*.zout_end) {
                if (!(stbi__zexpand(a, zout, 1) != 0)) return 0;
                zout = a.*.zout;
            }
            (blk: {
                const ref = &zout;
                const tmp = ref.*;
                ref.* += 1;
                break :blk tmp;
            }).* = @bitCast(@as(i8, @truncate(z)));
        } else {
            var p: [*c]u8 = undefined;
            var len: i32 = undefined;
            var dist: i32 = undefined;
            if (z == @as(i32, 256)) {
                a.*.zout = zout;
                if ((a.*.hit_zeof_once != 0) and (a.*.num_bits < @as(i32, 16))) {
                    return stbi__err("unexpected end");
                }
                return 1;
            }
            if (z >= @as(i32, 286)) return stbi__err("bad huffman code");
            z -= 257;
            len = stbi__zlength_base[@bitCast(@as(isize, @intCast(z)))];
            if (stbi__zlength_extra[@bitCast(@as(isize, @intCast(z)))] != 0) {
                {
                    const ref = &len;
                    ref.* = @bitCast(@as(u32, @truncate(@as(u32, @bitCast(@as(i32, ref.*))) +% stbi__zreceive(a, stbi__zlength_extra[@bitCast(@as(isize, @intCast(z)))]))));
                }
            }
            z = stbi__zhuffman_decode(a, &a.*.z_distance);
            if ((z < @as(i32, 0)) or (z >= @as(i32, 30))) return stbi__err("bad huffman code");
            dist = stbi__zdist_base[@bitCast(@as(isize, @intCast(z)))];
            if (stbi__zdist_extra[@bitCast(@as(isize, @intCast(z)))] != 0) {
                {
                    const ref = &dist;
                    ref.* = @bitCast(@as(u32, @truncate(@as(u32, @bitCast(@as(i32, ref.*))) +% stbi__zreceive(a, stbi__zdist_extra[@bitCast(@as(isize, @intCast(z)))]))));
                }
            }
            if (@divExact(@as(i64, @bitCast(@intFromPtr(zout) -% @intFromPtr(a.*.zout_start))), @sizeOf(u8)) < @as(i64, dist)) return stbi__err("bad dist");
            if (@as(i64, len) > @divExact(@as(i64, @bitCast(@intFromPtr(a.*.zout_end) -% @intFromPtr(zout))), @sizeOf(u8))) {
                if (!(stbi__zexpand(a, zout, len) != 0)) return 0;
                zout = a.*.zout;
            }
            p = @ptrCast(@alignCast(zout - @as(usize, @bitCast(@as(isize, @intCast(dist))))));
            if (dist == @as(i32, 1)) {
                const v: u8 = p.*;
                if (len != 0) {
                    while (true) {
                        (blk: {
                            const ref = &zout;
                            const tmp = ref.*;
                            ref.* += 1;
                            break :blk tmp;
                        }).* = @truncate(v);
                        if (!((blk: {
                            const ref = &len;
                            ref.* -= 1;
                            break :blk ref.*;
                        }) != 0)) break;
                    }
                }
            } else {
                if (len != 0) {
                    while (true) {
                        (blk: {
                            const ref = &zout;
                            const tmp = ref.*;
                            ref.* += 1;
                            break :blk tmp;
                        }).* = @truncate((blk: {
                            const ref = &p;
                            const tmp = ref.*;
                            ref.* += 1;
                            break :blk tmp;
                        }).*);
                        if (!((blk: {
                            const ref = &len;
                            ref.* -= 1;
                            break :blk ref.*;
                        }) != 0)) break;
                    }
                }
            }
        }
    }
    return undefined;
}
fn stbi__compute_huffman_codes(a: [*c]stbi__zbuf) callconv(.c) i32 {
    const static_local_length_dezigzag = struct {
        const length_dezigzag: [19]u8 = [19]u8{
            16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2,
            14, 1,  15,
        };
    };
    var z_codelength: stbi__zhuffman = undefined;
    var lencodes: [455]u8 = undefined;
    var codelength_sizes: [19]u8 = undefined;
    var i: i32 = undefined;
    var n: i32 = undefined;
    const hlit: i32 = @bitCast(@as(u32, @truncate(stbi__zreceive(a, 5) +% @as(u32, 257))));
    const hdist: i32 = @bitCast(@as(u32, @truncate(stbi__zreceive(a, 5) +% @as(u32, 1))));
    const hclen: i32 = @bitCast(@as(u32, @truncate(stbi__zreceive(a, 4) +% @as(u32, 4))));
    const ntot: i32 = hlit + hdist;
    _ = memset(@ptrCast(@alignCast(@as([*c]u8, @ptrCast(@alignCast(&codelength_sizes))))), 0, @sizeOf(@TypeOf(codelength_sizes)));
    {
        i = 0;
        while (i < hclen) : (i += 1) {
            const s: i32 = @bitCast(@as(u32, @truncate(stbi__zreceive(a, 3))));
            codelength_sizes[static_local_length_dezigzag.length_dezigzag[@bitCast(@as(isize, @intCast(i)))]] = @bitCast(@as(i8, @truncate(s)));
        }
    }
    if (!(stbi__zbuild_huffman(&z_codelength, @ptrCast(@alignCast(&codelength_sizes)), 19) != 0)) return 0;
    n = 0;
    while (n < ntot) {
        var c: i32 = stbi__zhuffman_decode(a, &z_codelength);
        if ((c < @as(i32, 0)) or (c >= @as(i32, 19))) return stbi__err("bad codelengths");
        if (c < @as(i32, 16)) {
            lencodes[
                @bitCast(@as(isize, @intCast(blk: {
                    const ref = &n;
                    const tmp = ref.*;
                    ref.* += 1;
                    break :blk tmp;
                })))
            ] = @bitCast(@as(i8, @truncate(c)));
        } else {
            var fill: u8 = 0;
            if (c == @as(i32, 16)) {
                c = @bitCast(@as(u32, @truncate(stbi__zreceive(a, 2) +% @as(u32, 3))));
                if (n == @as(i32, 0)) return stbi__err("bad codelengths");
                fill = lencodes[@bitCast(@as(isize, @intCast(n - @as(i32, 1))))];
            } else if (c == @as(i32, 17)) {
                c = @bitCast(@as(u32, @truncate(stbi__zreceive(a, 3) +% @as(u32, 3))));
            } else if (c == @as(i32, 18)) {
                c = @bitCast(@as(u32, @truncate(stbi__zreceive(a, 7) +% @as(u32, 11))));
            } else {
                return stbi__err("bad codelengths");
            }
            if ((ntot - n) < c) return stbi__err("bad codelengths");
            _ = memset(@ptrCast(@alignCast(@as([*c]u8, @ptrCast(@alignCast(&lencodes))) + @as(usize, @bitCast(@as(isize, @intCast(n)))))), fill, @bitCast(@as(i64, c)));
            n += c;
        }
    }
    if (n != ntot) return stbi__err("bad codelengths");
    if (!(stbi__zbuild_huffman(&a.*.z_length, @ptrCast(@alignCast(&lencodes)), hlit) != 0)) return 0;
    if (!(stbi__zbuild_huffman(&a.*.z_distance, @as([*c]u8, @ptrCast(@alignCast(&lencodes))) + @as(usize, @bitCast(@as(isize, @intCast(hlit)))), hdist) != 0)) return 0;
    return 1;
}
fn stbi__parse_uncompressed_block(a: [*c]stbi__zbuf) callconv(.c) i32 {
    var header: [4]u8 = undefined;
    var len: i32 = undefined;
    var nlen: i32 = undefined;
    var k: i32 = undefined;
    if ((a.*.num_bits & @as(i32, 7)) != 0) {
        _ = stbi__zreceive(a, a.*.num_bits & @as(i32, 7));
    }
    k = 0;
    while (a.*.num_bits > @as(i32, 0)) {
        header[
            @bitCast(@as(isize, @intCast(blk: {
                const ref = &k;
                const tmp = ref.*;
                ref.* += 1;
                break :blk tmp;
            })))
        ] = @truncate(a.*.code_buffer & @as(u32, 255));
        a.*.code_buffer >>= @intCast(@as(u32, 8));
        a.*.num_bits -= 8;
    }
    if (a.*.num_bits < @as(i32, 0)) return stbi__err("zlib corrupt");
    while (k < @as(i32, 4)) {
        header[
            @bitCast(@as(isize, @intCast(blk: {
                const ref = &k;
                const tmp = ref.*;
                ref.* += 1;
                break :blk tmp;
            })))
        ] = stbi__zget8(a);
    }
    len = (@as(i32, header[@as(i32, 1)]) * @as(i32, 256)) + @as(i32, header[@as(i32, 0)]);
    nlen = (@as(i32, header[@as(i32, 3)]) * @as(i32, 256)) + @as(i32, header[@as(i32, 2)]);
    if (nlen != (len ^ @as(i32, 65535))) return stbi__err("zlib corrupt");
    if ((a.*.zbuffer + @as(usize, @bitCast(@as(isize, @intCast(len))))) > a.*.zbuffer_end) return stbi__err("read past buffer");
    if ((a.*.zout + @as(usize, @bitCast(@as(isize, @intCast(len))))) > a.*.zout_end) if (!(stbi__zexpand(a, a.*.zout, len) != 0)) return 0;
    _ = memcpy(@ptrCast(@alignCast(a.*.zout)), @ptrCast(@alignCast(a.*.zbuffer)), @bitCast(@as(i64, len)));
    a.*.zbuffer += @as(usize, @bitCast(@as(isize, @intCast(len))));
    a.*.zout += @as(usize, @bitCast(@as(isize, @intCast(len))));
    return 1;
}
fn stbi__parse_zlib_header(a: [*c]stbi__zbuf) callconv(.c) i32 {
    const cmf: i32 = stbi__zget8(a);
    const cm: i32 = cmf & @as(i32, 15);
    const flg: i32 = stbi__zget8(a);
    if (stbi__zeof(a) != 0) return stbi__err("bad zlib header");
    if (@rem((cmf * @as(i32, 256)) + flg, @as(i32, 31)) != @as(i32, 0)) return stbi__err("bad zlib header");
    if ((flg & @as(i32, 32)) != 0) return stbi__err("no preset dict");
    if (cm != @as(i32, 8)) return stbi__err("bad compression");
    return 1;
}
const stbi__zdefault_length: [288]u8 = [288]u8{
    8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
    8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
    8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
    8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
    8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
    8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
    8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
    8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
    8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
    9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9,
    9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9,
    9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9,
    9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9,
    9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9,
    9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9,
    9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9,
    7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
    7, 7, 7, 7, 7, 7, 7, 7, 8, 8, 8, 8, 8, 8, 8, 8,
};
const stbi__zdefault_distance: [32]u8 = [32]u8{
    5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
    5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
};
fn stbi__parse_zlib(a: [*c]stbi__zbuf, parse_header: i32) callconv(.c) i32 {
    var final: i32 = undefined;
    var @"type": i32 = undefined;
    if (parse_header != 0) if (!(stbi__parse_zlib_header(a) != 0)) return 0;
    a.*.num_bits = 0;
    a.*.code_buffer = 0;
    a.*.hit_zeof_once = 0;
    while (true) {
        final = @bitCast(@as(u32, @truncate(stbi__zreceive(a, 1))));
        @"type" = @bitCast(@as(u32, @truncate(stbi__zreceive(a, 2))));
        if (@"type" == @as(i32, 0)) {
            if (!(stbi__parse_uncompressed_block(a) != 0)) return 0;
        } else if (@"type" == @as(i32, 3)) {
            return 0;
        } else {
            if (@"type" == @as(i32, 1)) {
                if (!(stbi__zbuild_huffman(&a.*.z_length, @ptrCast(@alignCast(&stbi__zdefault_length)), STBI__ZNSYMS) != 0)) return 0;
                if (!(stbi__zbuild_huffman(&a.*.z_distance, @ptrCast(@alignCast(&stbi__zdefault_distance)), 32) != 0)) return 0;
            } else {
                if (!(stbi__compute_huffman_codes(a) != 0)) return 0;
            }
            if (!(stbi__parse_huffman_block(a) != 0)) return 0;
        }
        if (!!(final != 0)) break;
    }
    return 1;
}
fn stbi__do_zlib(a: [*c]stbi__zbuf, obuf: [*c]u8, olen: i32, exp_1: i32, parse_header: i32) callconv(.c) i32 {
    a.*.zout_start = obuf;
    a.*.zout = obuf;
    a.*.zout_end = obuf + @as(usize, @bitCast(@as(isize, @intCast(olen))));
    a.*.z_expandable = exp_1;
    return stbi__parse_zlib(a, parse_header);
}
fn stbi_zlib_decode_malloc_guesssize_headerflag(buffer: [*c]const u8, len: i32, initial_size: i32, outlen: [*c]i32, parse_header: i32) [*c]u8 {
    var a: stbi__zbuf = undefined;
    const p: [*c]u8 = @ptrCast(@alignCast(stbi__malloc(@bitCast(@as(i64, initial_size)))));
    if (@as(?*anyopaque, @ptrCast(@alignCast(p))) == @as(?*anyopaque, null)) return null;
    a.zbuffer = @ptrCast(@alignCast(@constCast(buffer)));
    a.zbuffer_end = @as([*c]u8, @ptrCast(@alignCast(@constCast(buffer)))) + @as(usize, @bitCast(@as(isize, @intCast(len))));
    if (stbi__do_zlib(&a, p, initial_size, 1, parse_header) != 0) {
        if (outlen != null) {
            outlen.* = @truncate(@divExact(@as(i64, @bitCast(@intFromPtr(a.zout) -% @intFromPtr(a.zout_start))), @sizeOf(u8)));
        }
        return a.zout_start;
    } else {
        free(@ptrCast(@alignCast(a.zout_start)));
        return null;
    }
}
const stbi__pngchunk = extern struct {
    length: u32 = 0,
    type: u32 = 0,
};
fn stbi__get_chunk_header(s: [*c]stbi__context) callconv(.c) stbi__pngchunk {
    var c: stbi__pngchunk = undefined;
    c.length = stbi__get32be(s);
    c.type = stbi__get32be(s);
    return c;
}
fn stbi__check_png_header(s: [*c]stbi__context) callconv(.c) i32 {
    const static_local_png_sig = struct {
        const png_sig: [8]u8 = [8]u8{ 137, 80, 78, 71, 13, 10, 26, 10 };
    };
    var i: i32 = undefined;
    {
        i = 0;
        while (i < @as(i32, 8)) : (i += 1) if (@as(i32, stbi__get8(s)) != @as(i32, static_local_png_sig.png_sig[@bitCast(@as(isize, @intCast(i)))])) return stbi__err("bad png sig");
    }
    return 1;
}
const stbi__png = extern struct {
    s: [*c]stbi__context = null,
    idata: [*c]u8 = null,
    expanded: [*c]u8 = null,
    out: [*c]u8 = null,
    depth: i32 = 0,
};
const STBI__F_none: i32 = 0;
const STBI__F_sub: i32 = 1;
const STBI__F_up: i32 = 2;
const STBI__F_avg: i32 = 3;
const STBI__F_paeth: i32 = 4;
const STBI__F_avg_first: i32 = 5;
var first_row_filter: [5]u8 = [5]u8{
    STBI__F_none,
    STBI__F_sub,
    STBI__F_none,
    STBI__F_avg_first,
    STBI__F_sub,
};
inline fn stbi__paeth(a: i32, b: i32, c: i32) i32 {
    const thresh: i32 = (c * @as(i32, 3)) - (a + b);
    const lo: i32 = if (a < b) a else b;
    const hi: i32 = if (a < b) b else a;
    const t0: i32 = if (hi <= thresh) lo else c;
    const t1: i32 = if (thresh <= lo) hi else t0;
    return t1;
}
const stbi__depth_scale_table: [9]u8 = [9]u8{
    0, 255, 85, 0, 17, 0, 0, 0, 1,
};
fn stbi__create_png_alpha_expand8(arg_dest: [*c]u8, src: [*c]u8, x: u32, img_n: i32) callconv(.c) void {
    var dest = arg_dest;
    var i: i32 = undefined;
    if (img_n == @as(i32, 1)) {
        {
            i = @bitCast(@as(u32, @truncate(x -% @as(u32, 1))));
            while (i >= @as(i32, 0)) : (i -= 1) {
                dest[@bitCast(@as(isize, @intCast((i * @as(i32, 2)) + @as(i32, 1))))] = 255;
                dest[@bitCast(@as(isize, @intCast((i * @as(i32, 2)) + @as(i32, 0))))] = src[@bitCast(@as(isize, @intCast(i)))];
            }
        }
    } else {
        _ = !!(img_n == @as(i32, 3)) or ((blk: {
            _assert("img_n == 3", "tools\\dataset\\vendor\\stb_image.h", 4685);
            break :blk 0;
        }) != 0);
        {
            i = @bitCast(@as(u32, @truncate(x -% @as(u32, 1))));
            while (i >= @as(i32, 0)) : (i -= 1) {
                dest[@bitCast(@as(isize, @intCast((i * @as(i32, 4)) + @as(i32, 3))))] = 255;
                dest[@bitCast(@as(isize, @intCast((i * @as(i32, 4)) + @as(i32, 2))))] = src[@bitCast(@as(isize, @intCast((i * @as(i32, 3)) + @as(i32, 2))))];
                dest[@bitCast(@as(isize, @intCast((i * @as(i32, 4)) + @as(i32, 1))))] = src[@bitCast(@as(isize, @intCast((i * @as(i32, 3)) + @as(i32, 1))))];
                dest[@bitCast(@as(isize, @intCast((i * @as(i32, 4)) + @as(i32, 0))))] = src[@bitCast(@as(isize, @intCast((i * @as(i32, 3)) + @as(i32, 0))))];
            }
        }
    }
}
fn stbi__create_png_image_raw(a: [*c]stbi__png, arg_raw: [*c]u8, raw_len: u32, out_n: i32, x: u32, y: u32, depth: i32, color: i32) callconv(.c) i32 {
    var raw = arg_raw;
    const bytes: i32 = if (depth == @as(i32, 16)) @as(i32, 2) else @as(i32, 1);
    const s: [*c]stbi__context = a.*.s;
    var i: u32 = undefined;
    var j: u32 = undefined;
    const stride: u32 = (x *% @as(u32, @bitCast(@as(i32, out_n)))) *% @as(u32, @bitCast(@as(i32, bytes)));
    var img_len: u32 = undefined;
    var img_width_bytes: u32 = undefined;
    var filter_buf: [*c]u8 = undefined;
    var all_ok: i32 = 1;
    var k: i32 = undefined;
    const img_n: i32 = s.*.img_n;
    const output_bytes: i32 = out_n * bytes;
    var filter_bytes: i32 = img_n * bytes;
    var width: i32 = @bitCast(@as(u32, @truncate(x)));
    _ = !!((out_n == s.*.img_n) or (out_n == (s.*.img_n + @as(i32, 1)))) or ((blk: {
        _assert("out_n == s->img_n || out_n == s->img_n+1", "tools\\dataset\\vendor\\stb_image.h", 4711);
        break :blk 0;
    }) != 0);
    a.*.out = @ptrCast(@alignCast(stbi__malloc_mad3(@bitCast(@as(u32, @truncate(x))), @bitCast(@as(u32, @truncate(y))), output_bytes, 0)));
    if (!(a.*.out != null)) return stbi__err("outofmem");
    if (!(stbi__mad3sizes_valid(img_n, @bitCast(@as(u32, @truncate(x))), depth, 7) != 0)) return stbi__err("too large");
    img_width_bytes = (((@as(u32, @bitCast(@as(i32, img_n))) *% x) *% @as(u32, @bitCast(@as(i32, depth)))) +% @as(u32, 7)) >> @intCast(@as(u32, 3));
    if (!(stbi__mad2sizes_valid(@bitCast(@as(u32, @truncate(img_width_bytes))), @bitCast(@as(u32, @truncate(y))), @bitCast(@as(u32, @truncate(img_width_bytes)))) != 0)) return stbi__err("too large");
    img_len = (img_width_bytes +% @as(u32, 1)) *% y;
    if (raw_len < img_len) return stbi__err("not enough pixels");
    filter_buf = @ptrCast(@alignCast(stbi__malloc_mad2(@bitCast(@as(u32, @truncate(img_width_bytes))), 2, 0)));
    if (!(filter_buf != null)) return stbi__err("outofmem");
    if (depth < @as(i32, 8)) {
        filter_bytes = 1;
        width = @bitCast(@as(u32, @truncate(img_width_bytes)));
    }
    {
        j = 0;
        while (j < y) : (j +%= 1) {
            var cur: [*c]u8 = filter_buf + ((j & @as(u32, 1)) *% img_width_bytes);
            const prior: [*c]u8 = filter_buf + ((~j & @as(u32, 1)) *% img_width_bytes);
            const dest: [*c]u8 = a.*.out + (stride *% j);
            const nk: i32 = width * filter_bytes;
            var filter: i32 = (blk: {
                const ref = &raw;
                const tmp = ref.*;
                ref.* += 1;
                break :blk tmp;
            }).*;
            if (filter > @as(i32, 4)) {
                all_ok = stbi__err("invalid filter");
                break;
            }
            if (j == @as(u32, 0)) {
                filter = first_row_filter[@bitCast(@as(isize, @intCast(filter)))];
            }
            blk: {
                switch (filter) {
                    STBI__F_none => {
                        _ = memcpy(@ptrCast(@alignCast(cur)), @ptrCast(@alignCast(raw)), @bitCast(@as(i64, nk)));
                        break :blk;
                    },
                    STBI__F_sub => {
                        _ = memcpy(@ptrCast(@alignCast(cur)), @ptrCast(@alignCast(raw)), @bitCast(@as(i64, filter_bytes)));
                        {
                            k = filter_bytes;
                            while (k < nk) : (k += 1) {
                                cur[@bitCast(@as(isize, @intCast(k)))] = @as(u8, @bitCast(@as(i8, @truncate((@as(i32, raw[@bitCast(@as(isize, @intCast(k)))]) + @as(i32, cur[@bitCast(@as(isize, @intCast(k - filter_bytes)))])) & @as(i32, 255)))));
                            }
                        }
                        break :blk;
                    },
                    STBI__F_up => {
                        {
                            const cur_p: [*]u8 = @ptrCast(cur);
                            const raw_p: [*]const u8 = @ptrCast(raw);
                            const pri_p: [*]const u8 = @ptrCast(prior);
                            const nk_u: usize = @intCast(nk);
                            var u: usize = 0;
                            while (u + 16 <= nk_u) : (u += 16) {
                                const rv: @Vector(16, u8) = raw_p[u..][0..16].*;
                                const pv: @Vector(16, u8) = pri_p[u..][0..16].*;
                                cur_p[u..][0..16].* = rv +% pv;
                            }
                            while (u < nk_u) : (u += 1) {
                                cur_p[u] = raw_p[u] +% pri_p[u];
                            }
                        }
                        break :blk;
                    },
                    STBI__F_avg => {
                        {
                            k = 0;
                            while (k < filter_bytes) : (k += 1) {
                                cur[@bitCast(@as(isize, @intCast(k)))] = @as(u8, @bitCast(@as(i8, @truncate((@as(i32, raw[@bitCast(@as(isize, @intCast(k)))]) + (@as(i32, prior[@bitCast(@as(isize, @intCast(k)))]) >> @intCast(@as(i32, 1)))) & @as(i32, 255)))));
                            }
                        }
                        {
                            k = filter_bytes;
                            while (k < nk) : (k += 1) {
                                cur[@bitCast(@as(isize, @intCast(k)))] = @as(u8, @bitCast(@as(i8, @truncate((@as(i32, raw[@bitCast(@as(isize, @intCast(k)))]) + ((@as(i32, prior[@bitCast(@as(isize, @intCast(k)))]) + @as(i32, cur[@bitCast(@as(isize, @intCast(k - filter_bytes)))])) >> @intCast(@as(i32, 1)))) & @as(i32, 255)))));
                            }
                        }
                        break :blk;
                    },
                    STBI__F_paeth => {
                        {
                            k = 0;
                            while (k < filter_bytes) : (k += 1) {
                                cur[@bitCast(@as(isize, @intCast(k)))] = @as(u8, @bitCast(@as(i8, @truncate((@as(i32, raw[@bitCast(@as(isize, @intCast(k)))]) + @as(i32, prior[@bitCast(@as(isize, @intCast(k)))])) & @as(i32, 255)))));
                            }
                        }
                        {
                            k = filter_bytes;
                            while (k < nk) : (k += 1) {
                                cur[@bitCast(@as(isize, @intCast(k)))] = @as(u8, @bitCast(@as(i8, @truncate((@as(i32, raw[@bitCast(@as(isize, @intCast(k)))]) + stbi__paeth(cur[@bitCast(@as(isize, @intCast(k - filter_bytes)))], prior[@bitCast(@as(isize, @intCast(k)))], prior[@bitCast(@as(isize, @intCast(k - filter_bytes)))])) & @as(i32, 255)))));
                            }
                        }
                        break :blk;
                    },
                    STBI__F_avg_first => {
                        _ = memcpy(@ptrCast(@alignCast(cur)), @ptrCast(@alignCast(raw)), @bitCast(@as(i64, filter_bytes)));
                        {
                            k = filter_bytes;
                            while (k < nk) : (k += 1) {
                                cur[@bitCast(@as(isize, @intCast(k)))] = @as(u8, @bitCast(@as(i8, @truncate((@as(i32, raw[@bitCast(@as(isize, @intCast(k)))]) + (@as(i32, cur[@bitCast(@as(isize, @intCast(k - filter_bytes)))]) >> @intCast(@as(i32, 1)))) & @as(i32, 255)))));
                            }
                        }
                        break :blk;
                    },
                    else => {},
                }
            }
            raw += @as(usize, @bitCast(@as(isize, @intCast(nk))));
            if (depth < @as(i32, 8)) {
                const scale: u8 = @bitCast(@as(i8, @truncate(if (color == @as(i32, 0)) @as(i32, stbi__depth_scale_table[@bitCast(@as(isize, @intCast(depth)))]) else @as(i32, 1))));
                var in: [*c]u8 = cur;
                var out: [*c]u8 = dest;
                var inb: u8 = 0;
                const nsmp: u32 = x *% @as(u32, @bitCast(@as(i32, img_n)));
                if (depth == @as(i32, 4)) {
                    {
                        i = 0;
                        while (i < nsmp) : (i +%= 1) {
                            if ((i & @as(u32, 1)) == @as(u32, 0)) {
                                inb = (blk: {
                                    const ref = &in;
                                    const tmp = ref.*;
                                    ref.* += 1;
                                    break :blk tmp;
                                }).*;
                            }
                            (blk: {
                                const ref = &out;
                                const tmp = ref.*;
                                ref.* += 1;
                                break :blk tmp;
                            }).* = @bitCast(@as(i8, @truncate(@as(i32, scale) * (@as(i32, inb) >> @intCast(@as(i32, 4))))));
                            {
                                const ref = &inb;
                                ref.* = @bitCast(@as(i8, @truncate(@as(i32, ref.*) << @intCast(@as(i32, 4)))));
                            }
                        }
                    }
                } else if (depth == @as(i32, 2)) {
                    {
                        i = 0;
                        while (i < nsmp) : (i +%= 1) {
                            if ((i & @as(u32, 3)) == @as(u32, 0)) {
                                inb = (blk: {
                                    const ref = &in;
                                    const tmp = ref.*;
                                    ref.* += 1;
                                    break :blk tmp;
                                }).*;
                            }
                            (blk: {
                                const ref = &out;
                                const tmp = ref.*;
                                ref.* += 1;
                                break :blk tmp;
                            }).* = @bitCast(@as(i8, @truncate(@as(i32, scale) * (@as(i32, inb) >> @intCast(@as(i32, 6))))));
                            {
                                const ref = &inb;
                                ref.* = @bitCast(@as(i8, @truncate(@as(i32, ref.*) << @intCast(@as(i32, 2)))));
                            }
                        }
                    }
                } else {
                    _ = !!(depth == @as(i32, 1)) or ((blk: {
                        _assert("depth == 1", "tools\\dataset\\vendor\\stb_image.h", 4811);
                        break :blk 0;
                    }) != 0);
                    {
                        i = 0;
                        while (i < nsmp) : (i +%= 1) {
                            if ((i & @as(u32, 7)) == @as(u32, 0)) {
                                inb = (blk: {
                                    const ref = &in;
                                    const tmp = ref.*;
                                    ref.* += 1;
                                    break :blk tmp;
                                }).*;
                            }
                            (blk: {
                                const ref = &out;
                                const tmp = ref.*;
                                ref.* += 1;
                                break :blk tmp;
                            }).* = @bitCast(@as(i8, @truncate(@as(i32, scale) * (@as(i32, inb) >> @intCast(@as(i32, 7))))));
                            {
                                const ref = &inb;
                                ref.* = @bitCast(@as(i8, @truncate(@as(i32, ref.*) << @intCast(@as(i32, 1)))));
                            }
                        }
                    }
                }
                if (img_n != out_n) {
                    stbi__create_png_alpha_expand8(dest, dest, x, img_n);
                }
            } else if (depth == @as(i32, 8)) {
                if (img_n == out_n) {
                    _ = memcpy(@ptrCast(@alignCast(dest)), @ptrCast(@alignCast(cur)), x *% @as(u32, @bitCast(@as(i32, img_n))));
                } else {
                    stbi__create_png_alpha_expand8(dest, cur, x, img_n);
                }
            } else if (depth == @as(i32, 16)) {
                var dest16: [*c]u16 = @ptrCast(@alignCast(dest));
                const nsmp: u32 = x *% @as(u32, @bitCast(@as(i32, img_n)));
                if (img_n == out_n) {
                    {
                        i = 0;
                        while (i < nsmp) : ({
                            i +%= 1;
                            dest16 += 1;
                            cur += @as(usize, @bitCast(@as(isize, @intCast(@as(i32, 2)))));
                        }) {
                            dest16.* = @bitCast(@as(i16, @truncate((@as(i32, cur[@as(i32, 0)]) << @intCast(@as(i32, 8))) | @as(i32, cur[@as(i32, 1)]))));
                        }
                    }
                } else {
                    _ = !!((img_n + @as(i32, 1)) == out_n) or ((blk: {
                        _assert("img_n+1 == out_n", "tools\\dataset\\vendor\\stb_image.h", 4836);
                        break :blk 0;
                    }) != 0);
                    if (img_n == @as(i32, 1)) {
                        {
                            i = 0;
                            while (i < x) : ({
                                i +%= 1;
                                dest16 += @as(usize, @bitCast(@as(isize, @intCast(@as(i32, 2)))));
                                cur += @as(usize, @bitCast(@as(isize, @intCast(@as(i32, 2)))));
                            }) {
                                dest16[@as(i32, 0)] = @bitCast(@as(i16, @truncate((@as(i32, cur[@as(i32, 0)]) << @intCast(@as(i32, 8))) | @as(i32, cur[@as(i32, 1)]))));
                                dest16[@as(i32, 1)] = 65535;
                            }
                        }
                    } else {
                        _ = !!(img_n == @as(i32, 3)) or ((blk: {
                            _assert("img_n == 3", "tools\\dataset\\vendor\\stb_image.h", 4843);
                            break :blk 0;
                        }) != 0);
                        {
                            i = 0;
                            while (i < x) : ({
                                i +%= 1;
                                dest16 += @as(usize, @bitCast(@as(isize, @intCast(@as(i32, 4)))));
                                cur += @as(usize, @bitCast(@as(isize, @intCast(@as(i32, 6)))));
                            }) {
                                dest16[@as(i32, 0)] = @bitCast(@as(i16, @truncate((@as(i32, cur[@as(i32, 0)]) << @intCast(@as(i32, 8))) | @as(i32, cur[@as(i32, 1)]))));
                                dest16[@as(i32, 1)] = @bitCast(@as(i16, @truncate((@as(i32, cur[@as(i32, 2)]) << @intCast(@as(i32, 8))) | @as(i32, cur[@as(i32, 3)]))));
                                dest16[@as(i32, 2)] = @bitCast(@as(i16, @truncate((@as(i32, cur[@as(i32, 4)]) << @intCast(@as(i32, 8))) | @as(i32, cur[@as(i32, 5)]))));
                                dest16[@as(i32, 3)] = 65535;
                            }
                        }
                    }
                }
            }
        }
    }
    free(@ptrCast(@alignCast(filter_buf)));
    if (!(all_ok != 0)) return 0;
    return 1;
}
fn stbi__create_png_image(a: [*c]stbi__png, arg_image_data: [*c]u8, arg_image_data_len: u32, out_n: i32, depth: i32, color: i32, interlaced: i32) callconv(.c) i32 {
    var image_data = arg_image_data;
    var image_data_len = arg_image_data_len;
    const bytes: i32 = if (depth == @as(i32, 16)) @as(i32, 2) else @as(i32, 1);
    const out_bytes: i32 = out_n * bytes;
    var final: [*c]u8 = undefined;
    var p: i32 = undefined;
    if (!(interlaced != 0)) return stbi__create_png_image_raw(a, image_data, image_data_len, out_n, a.*.s.*.img_x, a.*.s.*.img_y, depth, color);
    final = @ptrCast(@alignCast(stbi__malloc_mad3(@bitCast(@as(u32, @truncate(a.*.s.*.img_x))), @bitCast(@as(u32, @truncate(a.*.s.*.img_y))), out_bytes, 0)));
    if (!(final != null)) return stbi__err("outofmem");
    {
        p = 0;
        while (p < @as(i32, 7)) : (p += 1) {
            const xorig: [7]i32 = [7]i32{ 0, 4, 0, 2, 0, 1, 0 };
            const yorig: [7]i32 = [7]i32{ 0, 0, 4, 0, 2, 0, 1 };
            const xspc: [7]i32 = [7]i32{ 8, 8, 4, 4, 2, 2, 1 };
            const yspc: [7]i32 = [7]i32{ 8, 8, 8, 4, 4, 2, 2 };
            var i: i32 = undefined;
            var j: i32 = undefined;
            var x: i32 = undefined;
            var y: i32 = undefined;
            x = @bitCast(@as(u32, @truncate((((a.*.s.*.img_x -% @as(u32, @bitCast(@as(i32, xorig[@bitCast(@as(isize, @intCast(p)))])))) +% @as(u32, @bitCast(@as(i32, xspc[@bitCast(@as(isize, @intCast(p)))])))) -% @as(u32, 1)) / @as(u32, @bitCast(@as(i32, xspc[@bitCast(@as(isize, @intCast(p)))]))))));
            y = @bitCast(@as(u32, @truncate((((a.*.s.*.img_y -% @as(u32, @bitCast(@as(i32, yorig[@bitCast(@as(isize, @intCast(p)))])))) +% @as(u32, @bitCast(@as(i32, yspc[@bitCast(@as(isize, @intCast(p)))])))) -% @as(u32, 1)) / @as(u32, @bitCast(@as(i32, yspc[@bitCast(@as(isize, @intCast(p)))]))))));
            if ((x != 0) and (y != 0)) {
                const img_len: u32 = @bitCast(@as(i32, (((((a.*.s.*.img_n * x) * depth) + @as(i32, 7)) >> @intCast(@as(i32, 3))) + @as(i32, 1)) * y));
                if (!(stbi__create_png_image_raw(a, image_data, image_data_len, out_n, @bitCast(@as(i32, x)), @bitCast(@as(i32, y)), depth, color) != 0)) {
                    free(@ptrCast(@alignCast(final)));
                    return 0;
                }
                {
                    j = 0;
                    while (j < y) : (j += 1) {
                        {
                            i = 0;
                            while (i < x) : (i += 1) {
                                const out_y: i32 = (j * yspc[@bitCast(@as(isize, @intCast(p)))]) + yorig[@bitCast(@as(isize, @intCast(p)))];
                                const out_x: i32 = (i * xspc[@bitCast(@as(isize, @intCast(p)))]) + xorig[@bitCast(@as(isize, @intCast(p)))];
                                _ = memcpy(@ptrCast(@alignCast((final + ((@as(u32, @bitCast(@as(i32, out_y))) *% a.*.s.*.img_x) *% @as(u32, @bitCast(@as(i32, out_bytes))))) + @as(usize, @bitCast(@as(isize, @intCast(out_x * out_bytes)))))), @ptrCast(@alignCast(a.*.out + @as(usize, @bitCast(@as(isize, @intCast(((j * x) + i) * out_bytes)))))), @bitCast(@as(i64, out_bytes)));
                            }
                        }
                    }
                }
                free(@ptrCast(@alignCast(a.*.out)));
                image_data += img_len;
                image_data_len -%= img_len;
            }
        }
    }
    a.*.out = final;
    return 1;
}
fn stbi__compute_transparency(z: [*c]stbi__png, tc: [*c]u8, out_n: i32) callconv(.c) i32 {
    const s: [*c]stbi__context = z.*.s;
    var i: u32 = undefined;
    const pixel_count: u32 = s.*.img_x *% s.*.img_y;
    var p: [*c]u8 = z.*.out;
    _ = !!((out_n == @as(i32, 2)) or (out_n == @as(i32, 4))) or ((blk: {
        _assert("out_n == 2 || out_n == 4", "tools\\dataset\\vendor\\stb_image.h", 4914);
        break :blk 0;
    }) != 0);
    if (out_n == @as(i32, 2)) {
        {
            i = 0;
            while (i < pixel_count) : (i +%= 1) {
                p[@as(i32, 1)] = @bitCast(@as(i8, @truncate(if (@as(i32, p[@as(i32, 0)]) == @as(i32, tc[@as(i32, 0)])) @as(i32, 0) else @as(i32, 255))));
                p += @as(usize, @bitCast(@as(isize, @intCast(@as(i32, 2)))));
            }
        }
    } else {
        {
            i = 0;
            while (i < pixel_count) : (i +%= 1) {
                if (((@as(i32, p[@as(i32, 0)]) == @as(i32, tc[@as(i32, 0)])) and (@as(i32, p[@as(i32, 1)]) == @as(i32, tc[@as(i32, 1)]))) and (@as(i32, p[@as(i32, 2)]) == @as(i32, tc[@as(i32, 2)]))) {
                    p[@as(i32, 3)] = 0;
                }
                p += @as(usize, @bitCast(@as(isize, @intCast(@as(i32, 4)))));
            }
        }
    }
    return 1;
}
fn stbi__compute_transparency16(z: [*c]stbi__png, tc: [*c]u16, out_n: i32) callconv(.c) i32 {
    const s: [*c]stbi__context = z.*.s;
    var i: u32 = undefined;
    const pixel_count: u32 = s.*.img_x *% s.*.img_y;
    var p: [*c]u16 = @ptrCast(@alignCast(z.*.out));
    _ = !!((out_n == @as(i32, 2)) or (out_n == @as(i32, 4))) or ((blk: {
        _assert("out_n == 2 || out_n == 4", "tools\\dataset\\vendor\\stb_image.h", 4939);
        break :blk 0;
    }) != 0);
    if (out_n == @as(i32, 2)) {
        {
            i = 0;
            while (i < pixel_count) : (i +%= 1) {
                p[@as(i32, 1)] = @bitCast(@as(i16, @truncate(if (@as(i32, p[@as(i32, 0)]) == @as(i32, tc[@as(i32, 0)])) @as(i32, 0) else @as(i32, 65535))));
                p += @as(usize, @bitCast(@as(isize, @intCast(@as(i32, 2)))));
            }
        }
    } else {
        {
            i = 0;
            while (i < pixel_count) : (i +%= 1) {
                if (((@as(i32, p[@as(i32, 0)]) == @as(i32, tc[@as(i32, 0)])) and (@as(i32, p[@as(i32, 1)]) == @as(i32, tc[@as(i32, 1)]))) and (@as(i32, p[@as(i32, 2)]) == @as(i32, tc[@as(i32, 2)]))) {
                    p[@as(i32, 3)] = 0;
                }
                p += @as(usize, @bitCast(@as(isize, @intCast(@as(i32, 4)))));
            }
        }
    }
    return 1;
}
fn stbi__expand_png_palette(a: [*c]stbi__png, palette: [*c]u8, len: i32, pal_img_n: i32) callconv(.c) i32 {
    var i: u32 = undefined;
    const pixel_count: u32 = a.*.s.*.img_x *% a.*.s.*.img_y;
    var p: [*c]u8 = undefined;
    var temp_out: [*c]u8 = undefined;
    const orig: [*c]u8 = a.*.out;
    p = @ptrCast(@alignCast(stbi__malloc_mad2(@bitCast(@as(u32, @truncate(pixel_count))), pal_img_n, 0)));
    if (@as(?*anyopaque, @ptrCast(@alignCast(p))) == @as(?*anyopaque, null)) return stbi__err("outofmem");
    temp_out = p;
    if (pal_img_n == @as(i32, 3)) {
        {
            i = 0;
            while (i < pixel_count) : (i +%= 1) {
                const n: i32 = @as(i32, orig[i]) * @as(i32, 4);
                p[@as(i32, 0)] = palette[@bitCast(@as(isize, @intCast(n)))];
                p[@as(i32, 1)] = palette[@bitCast(@as(isize, @intCast(n + @as(i32, 1))))];
                p[@as(i32, 2)] = palette[@bitCast(@as(isize, @intCast(n + @as(i32, 2))))];
                p += @as(usize, @bitCast(@as(isize, @intCast(@as(i32, 3)))));
            }
        }
    } else {
        {
            i = 0;
            while (i < pixel_count) : (i +%= 1) {
                const n: i32 = @as(i32, orig[i]) * @as(i32, 4);
                p[@as(i32, 0)] = palette[@bitCast(@as(isize, @intCast(n)))];
                p[@as(i32, 1)] = palette[@bitCast(@as(isize, @intCast(n + @as(i32, 1))))];
                p[@as(i32, 2)] = palette[@bitCast(@as(isize, @intCast(n + @as(i32, 2))))];
                p[@as(i32, 3)] = palette[@bitCast(@as(isize, @intCast(n + @as(i32, 3))))];
                p += @as(usize, @bitCast(@as(isize, @intCast(@as(i32, 4)))));
            }
        }
    }
    free(@ptrCast(@alignCast(a.*.out)));
    a.*.out = temp_out;
    _ = @sizeOf(@TypeOf(len));
    return 1;
}
var stbi__unpremultiply_on_load_global: i32 = 0;
var stbi__de_iphone_flag_global: i32 = 0;
threadlocal var stbi__unpremultiply_on_load_local: i32 = 0;
threadlocal var stbi__unpremultiply_on_load_set: i32 = 0;
threadlocal var stbi__de_iphone_flag_local: i32 = 0;
threadlocal var stbi__de_iphone_flag_set: i32 = 0;
fn stbi__de_iphone(z: [*c]stbi__png) callconv(.c) void {
    const s: [*c]stbi__context = z.*.s;
    var i: u32 = undefined;
    const pixel_count: u32 = s.*.img_x *% s.*.img_y;
    var p: [*c]u8 = z.*.out;
    if (s.*.img_out_n == @as(i32, 3)) {
        {
            i = 0;
            while (i < pixel_count) : (i +%= 1) {
                const t: u8 = p[@as(i32, 0)];
                p[@as(i32, 0)] = p[@as(i32, 2)];
                p[@as(i32, 2)] = t;
                p += @as(usize, @bitCast(@as(isize, @intCast(@as(i32, 3)))));
            }
        }
    } else {
        _ = !!(s.*.img_out_n == @as(i32, 4)) or ((blk: {
            _assert("s->img_out_n == 4", "tools\\dataset\\vendor\\stb_image.h", 5047);
            break :blk 0;
        }) != 0);
        if ((if (stbi__unpremultiply_on_load_set != 0) stbi__unpremultiply_on_load_local else stbi__unpremultiply_on_load_global) != 0) {
            {
                i = 0;
                while (i < pixel_count) : (i +%= 1) {
                    const a: u8 = p[@as(i32, 3)];
                    const t: u8 = p[@as(i32, 0)];
                    if (@as(i32, a) != 0) {
                        const half: u8 = @bitCast(@as(i8, @truncate(@divTrunc(@as(i32, a), @as(i32, 2)))));
                        p[@as(i32, 0)] = @bitCast(@as(i8, @truncate(@divTrunc((@as(i32, p[@as(i32, 2)]) * @as(i32, 255)) + @as(i32, half), @as(i32, a)))));
                        p[@as(i32, 1)] = @bitCast(@as(i8, @truncate(@divTrunc((@as(i32, p[@as(i32, 1)]) * @as(i32, 255)) + @as(i32, half), @as(i32, a)))));
                        p[@as(i32, 2)] = @bitCast(@as(i8, @truncate(@divTrunc((@as(i32, t) * @as(i32, 255)) + @as(i32, half), @as(i32, a)))));
                    } else {
                        p[@as(i32, 0)] = p[@as(i32, 2)];
                        p[@as(i32, 2)] = t;
                    }
                    p += @as(usize, @bitCast(@as(isize, @intCast(@as(i32, 4)))));
                }
            }
        } else {
            {
                i = 0;
                while (i < pixel_count) : (i +%= 1) {
                    const t: u8 = p[@as(i32, 0)];
                    p[@as(i32, 0)] = p[@as(i32, 2)];
                    p[@as(i32, 2)] = t;
                    p += @as(usize, @bitCast(@as(isize, @intCast(@as(i32, 4)))));
                }
            }
        }
    }
}
fn stbi__parse_png_file(z: [*c]stbi__png, scan: i32, req_comp: i32) callconv(.c) i32 {
    var palette: [1024]u8 = undefined;
    var pal_img_n: u8 = 0;
    var has_trans: u8 = 0;
    var tc: [3]u8 = [1]u8{0} ++ @as([2]u8, @splat(0));
    var tc16: [3]u16 = undefined;
    var ioff: u32 = 0;
    var idata_limit: u32 = 0;
    var i: u32 = undefined;
    var pal_len: u32 = 0;
    var first: i32 = 1;
    var k: i32 = undefined;
    var interlace: i32 = 0;
    var color: i32 = 0;
    var is_iphone: i32 = 0;
    const s: [*c]stbi__context = z.*.s;
    z.*.expanded = null;
    z.*.idata = null;
    z.*.out = null;
    if (!(stbi__check_png_header(s) != 0)) return 0;
    if (scan == STBI__SCAN_type) return 1;
    while (true) {
        const c: stbi__pngchunk = stbi__get_chunk_header(s);
        blk: {
            switch (c.type) {
                (((@as(u32, @bitCast(@as(i32, @as(i32, 'C')))) << @intCast(@as(u32, 24))) +% (@as(u32, @bitCast(@as(i32, @as(i32, 'g')))) << @intCast(@as(u32, 16)))) +% (@as(u32, @bitCast(@as(i32, @as(i32, 'B')))) << @intCast(@as(u32, 8)))) +% @as(u32, @bitCast(@as(i32, @as(i32, 'I')))) => {
                    is_iphone = 1;
                    stbi__skip(s, @bitCast(@as(u32, @truncate(c.length))));
                    break :blk;
                },
                (((@as(u32, @bitCast(@as(i32, @as(i32, 'I')))) << @intCast(@as(u32, 24))) +% (@as(u32, @bitCast(@as(i32, @as(i32, 'H')))) << @intCast(@as(u32, 16)))) +% (@as(u32, @bitCast(@as(i32, @as(i32, 'D')))) << @intCast(@as(u32, 8)))) +% @as(u32, @bitCast(@as(i32, @as(i32, 'R')))) => {
                    {
                        var comp: i32 = undefined;
                        var filter: i32 = undefined;
                        if (!(first != 0)) return stbi__err("multiple IHDR");
                        first = 0;
                        if (c.length != @as(u32, 13)) return stbi__err("bad IHDR len");
                        s.*.img_x = stbi__get32be(s);
                        s.*.img_y = stbi__get32be(s);
                        if (s.*.img_y > @as(u32, @bitCast(@as(i32, @as(i32, 1) << @intCast(@as(i32, 24)))))) return stbi__err("too large");
                        if (s.*.img_x > @as(u32, @bitCast(@as(i32, @as(i32, 1) << @intCast(@as(i32, 24)))))) return stbi__err("too large");
                        z.*.depth = stbi__get8(s);
                        if (((((z.*.depth != @as(i32, 1)) and (z.*.depth != @as(i32, 2))) and (z.*.depth != @as(i32, 4))) and (z.*.depth != @as(i32, 8))) and (z.*.depth != @as(i32, 16))) return stbi__err("1/2/4/8/16-bit only");
                        color = stbi__get8(s);
                        if (color > @as(i32, 6)) return stbi__err("bad ctype");
                        if ((color == @as(i32, 3)) and (z.*.depth == @as(i32, 16))) return stbi__err("bad ctype");
                        if (color == @as(i32, 3)) {
                            pal_img_n = 3;
                        } else if ((color & @as(i32, 1)) != 0) return stbi__err("bad ctype");
                        comp = stbi__get8(s);
                        if (comp != 0) return stbi__err("bad comp method");
                        filter = stbi__get8(s);
                        if (filter != 0) return stbi__err("bad filter method");
                        interlace = stbi__get8(s);
                        if (interlace > @as(i32, 1)) return stbi__err("bad interlace method");
                        if (!(s.*.img_x != 0) or !(s.*.img_y != 0)) return stbi__err("0-pixel image");
                        if (!(@as(i32, pal_img_n) != 0)) {
                            s.*.img_n = (if ((color & @as(i32, 2)) != 0) @as(i32, 3) else @as(i32, 1)) + (if ((color & @as(i32, 4)) != 0) @as(i32, 1) else @as(i32, 0));
                            if (((@as(u32, @bitCast(@as(i32, @as(i32, 1) << @intCast(@as(i32, 30))))) / s.*.img_x) / @as(u32, @bitCast(@as(i32, s.*.img_n)))) < s.*.img_y) return stbi__err("too large");
                        } else {
                            s.*.img_n = 1;
                            if (((@as(u32, @bitCast(@as(i32, @as(i32, 1) << @intCast(@as(i32, 30))))) / s.*.img_x) / @as(u32, 4)) < s.*.img_y) return stbi__err("too large");
                        }
                        break :blk;
                    }
                },
                (((@as(u32, @bitCast(@as(i32, @as(i32, 'P')))) << @intCast(@as(u32, 24))) +% (@as(u32, @bitCast(@as(i32, @as(i32, 'L')))) << @intCast(@as(u32, 16)))) +% (@as(u32, @bitCast(@as(i32, @as(i32, 'T')))) << @intCast(@as(u32, 8)))) +% @as(u32, @bitCast(@as(i32, @as(i32, 'E')))) => {
                    {
                        if (first != 0) return stbi__err("first not IHDR");
                        if (c.length > @as(u32, @bitCast(@as(i32, @as(i32, 256) * @as(i32, 3))))) return stbi__err("invalid PLTE");
                        pal_len = c.length / @as(u32, 3);
                        if ((pal_len *% @as(u32, 3)) != c.length) return stbi__err("invalid PLTE");
                        {
                            i = 0;
                            while (i < pal_len) : (i +%= 1) {
                                palette[(i *% @as(u32, 4)) +% @as(u32, 0)] = stbi__get8(s);
                                palette[(i *% @as(u32, 4)) +% @as(u32, 1)] = stbi__get8(s);
                                palette[(i *% @as(u32, 4)) +% @as(u32, 2)] = stbi__get8(s);
                                palette[(i *% @as(u32, 4)) +% @as(u32, 3)] = 255;
                            }
                        }
                        break :blk;
                    }
                },
                (((@as(u32, @bitCast(@as(i32, @as(i32, 't')))) << @intCast(@as(u32, 24))) +% (@as(u32, @bitCast(@as(i32, @as(i32, 'R')))) << @intCast(@as(u32, 16)))) +% (@as(u32, @bitCast(@as(i32, @as(i32, 'N')))) << @intCast(@as(u32, 8)))) +% @as(u32, @bitCast(@as(i32, @as(i32, 'S')))) => {
                    {
                        if (first != 0) return stbi__err("first not IHDR");
                        if (z.*.idata != null) return stbi__err("tRNS after IDAT");
                        if (@as(i32, pal_img_n) != 0) {
                            if (scan == STBI__SCAN_header) {
                                s.*.img_n = 4;
                                return 1;
                            }
                            if (pal_len == @as(u32, 0)) return stbi__err("tRNS before PLTE");
                            if (c.length > pal_len) return stbi__err("bad tRNS len");
                            pal_img_n = 4;
                            {
                                i = 0;
                                while (i < c.length) : (i +%= 1) {
                                    palette[(i *% @as(u32, 4)) +% @as(u32, 3)] = stbi__get8(s);
                                }
                            }
                        } else {
                            if (!((s.*.img_n & @as(i32, 1)) != 0)) return stbi__err("tRNS with alpha");
                            if (c.length != (@as(u32, @bitCast(@as(i32, s.*.img_n))) *% @as(u32, 2))) return stbi__err("bad tRNS len");
                            has_trans = 1;
                            if (scan == STBI__SCAN_header) {
                                s.*.img_n += 1;
                                return 1;
                            }
                            if (z.*.depth == @as(i32, 16)) {
                                {
                                    k = 0;
                                    while ((k < s.*.img_n) and (k < @as(i32, 3))) : (k += 1) {
                                        tc16[@bitCast(@as(isize, @intCast(k)))] = @bitCast(@as(i16, @truncate(stbi__get16be(s))));
                                    }
                                }
                            } else {
                                {
                                    k = 0;
                                    while ((k < s.*.img_n) and (k < @as(i32, 3))) : (k += 1) {
                                        tc[@bitCast(@as(isize, @intCast(k)))] = @bitCast(@as(i8, @truncate(@as(i32, @as(u8, @bitCast(@as(i8, @truncate(stbi__get16be(s) & @as(i32, 255)))))) * @as(i32, stbi__depth_scale_table[@bitCast(@as(isize, @intCast(z.*.depth)))]))));
                                    }
                                }
                            }
                        }
                        break :blk;
                    }
                },
                (((@as(u32, @bitCast(@as(i32, @as(i32, 'I')))) << @intCast(@as(u32, 24))) +% (@as(u32, @bitCast(@as(i32, @as(i32, 'D')))) << @intCast(@as(u32, 16)))) +% (@as(u32, @bitCast(@as(i32, @as(i32, 'A')))) << @intCast(@as(u32, 8)))) +% @as(u32, @bitCast(@as(i32, @as(i32, 'T')))) => {
                    {
                        if (first != 0) return stbi__err("first not IHDR");
                        if ((@as(i32, pal_img_n) != 0) and !(pal_len != 0)) return stbi__err("no PLTE");
                        if (scan == STBI__SCAN_header) {
                            if (@as(i32, pal_img_n) != 0) {
                                s.*.img_n = pal_img_n;
                            }
                            return 1;
                        }
                        if (c.length > (@as(u32, 1) << @intCast(@as(u32, 30)))) return stbi__err("IDAT size limit");
                        if (@as(i32, @bitCast(@as(u32, @truncate(ioff +% c.length)))) < @as(i32, @bitCast(@as(u32, @truncate(ioff))))) return 0;
                        if ((ioff +% c.length) > idata_limit) {
                            const idata_limit_old: u32 = idata_limit;
                            var p: [*c]u8 = undefined;
                            if (idata_limit == @as(u32, 0)) {
                                idata_limit = if (c.length > @as(u32, 4096)) c.length else @as(u32, 4096);
                            }
                            while ((ioff +% c.length) > idata_limit) {
                                idata_limit *%= 2;
                            }
                            _ = @sizeOf(@TypeOf(idata_limit_old));
                            p = @ptrCast(@alignCast(realloc(@ptrCast(@alignCast(z.*.idata)), idata_limit)));
                            if (@as(?*anyopaque, @ptrCast(@alignCast(p))) == @as(?*anyopaque, null)) return stbi__err("outofmem");
                            z.*.idata = p;
                        }
                        if (!(stbi__getn(s, z.*.idata + ioff, @bitCast(@as(u32, @truncate(c.length)))) != 0)) return stbi__err("outofdata");
                        ioff +%= c.length;
                        break :blk;
                    }
                },
                (((@as(u32, @bitCast(@as(i32, @as(i32, 'I')))) << @intCast(@as(u32, 24))) +% (@as(u32, @bitCast(@as(i32, @as(i32, 'E')))) << @intCast(@as(u32, 16)))) +% (@as(u32, @bitCast(@as(i32, @as(i32, 'N')))) << @intCast(@as(u32, 8)))) +% @as(u32, @bitCast(@as(i32, @as(i32, 'D')))) => {
                    {
                        var raw_len: u32 = undefined;
                        var bpl: u32 = undefined;
                        if (first != 0) return stbi__err("first not IHDR");
                        if (scan != STBI__SCAN_load) return 1;
                        if (@as(?*anyopaque, @ptrCast(@alignCast(z.*.idata))) == @as(?*anyopaque, null)) return stbi__err("no IDAT");
                        bpl = ((s.*.img_x *% @as(u32, @bitCast(@as(i32, z.*.depth)))) +% @as(u32, 7)) / @as(u32, 8);
                        raw_len = ((bpl *% s.*.img_y) *% @as(u32, @bitCast(@as(i32, s.*.img_n)))) +% s.*.img_y;
                        z.*.expanded = @ptrCast(@alignCast(stbi_zlib_decode_malloc_guesssize_headerflag(@ptrCast(@alignCast(z.*.idata)), @bitCast(@as(u32, @truncate(ioff))), @bitCast(@as(u32, @truncate(raw_len))), @ptrCast(@alignCast(&raw_len)), @intFromBool(!(is_iphone != 0)))));
                        if (@as(?*anyopaque, @ptrCast(@alignCast(z.*.expanded))) == @as(?*anyopaque, null)) return 0;
                        free(@ptrCast(@alignCast(z.*.idata)));
                        z.*.idata = null;
                        if ((((req_comp == (s.*.img_n + @as(i32, 1))) and (req_comp != @as(i32, 3))) and !(@as(i32, pal_img_n) != 0)) or (@as(i32, has_trans) != 0)) {
                            s.*.img_out_n = s.*.img_n + @as(i32, 1);
                        } else {
                            s.*.img_out_n = s.*.img_n;
                        }
                        if (!(stbi__create_png_image(z, z.*.expanded, raw_len, s.*.img_out_n, z.*.depth, color, interlace) != 0)) return 0;
                        if (@as(i32, has_trans) != 0) {
                            if (z.*.depth == @as(i32, 16)) {
                                if (!(stbi__compute_transparency16(z, @ptrCast(@alignCast(&tc16)), s.*.img_out_n) != 0)) return 0;
                            } else {
                                if (!(stbi__compute_transparency(z, @ptrCast(@alignCast(&tc)), s.*.img_out_n) != 0)) return 0;
                            }
                        }
                        if (((is_iphone != 0) and ((if (stbi__de_iphone_flag_set != 0) stbi__de_iphone_flag_local else stbi__de_iphone_flag_global) != 0)) and (s.*.img_out_n > @as(i32, 2))) {
                            stbi__de_iphone(z);
                        }
                        if (@as(i32, pal_img_n) != 0) {
                            s.*.img_n = pal_img_n;
                            s.*.img_out_n = pal_img_n;
                            if (req_comp >= @as(i32, 3)) {
                                s.*.img_out_n = req_comp;
                            }
                            if (!(stbi__expand_png_palette(z, @ptrCast(@alignCast(&palette)), @bitCast(@as(u32, @truncate(pal_len))), s.*.img_out_n) != 0)) return 0;
                        } else if (@as(i32, has_trans) != 0) {
                            s.*.img_n += 1;
                        }
                        free(@ptrCast(@alignCast(z.*.expanded)));
                        z.*.expanded = null;
                        _ = stbi__get32be(s);
                        return 1;
                    }
                },
                else => {
                    if (first != 0) return stbi__err("first not IHDR");
                    if ((c.type & @as(u32, @bitCast(@as(i32, @as(i32, 1) << @intCast(@as(i32, 29)))))) == @as(u32, 0)) {
                        const static_local_invalid_chunk = struct {
                            var invalid_chunk: [24:0]u8 = "XXXX PNG chunk not known".*;
                        };
                        static_local_invalid_chunk.invalid_chunk[@as(i32, 0)] = @truncate(@as(u8, @truncate((c.type >> @intCast(@as(u32, 24))) & @as(u32, 255))));
                        static_local_invalid_chunk.invalid_chunk[@as(i32, 1)] = @truncate(@as(u8, @truncate((c.type >> @intCast(@as(u32, 16))) & @as(u32, 255))));
                        static_local_invalid_chunk.invalid_chunk[@as(i32, 2)] = @truncate(@as(u8, @truncate((c.type >> @intCast(@as(u32, 8))) & @as(u32, 255))));
                        static_local_invalid_chunk.invalid_chunk[@as(i32, 3)] = @truncate(@as(u8, @truncate((c.type >> @intCast(@as(u32, 0))) & @as(u32, 255))));
                        return stbi__err(@ptrCast(@alignCast(&static_local_invalid_chunk.invalid_chunk)));
                    }
                    stbi__skip(s, @bitCast(@as(u32, @truncate(c.length))));
                    break :blk;
                },
            }
        }
        _ = stbi__get32be(s);
    }
    return undefined;
}
fn stbi__do_png(p: [*c]stbi__png, x: [*c]i32, y: [*c]i32, n: [*c]i32, req_comp: i32, ri: [*c]stbi__result_info) callconv(.c) ?*anyopaque {
    var result: ?*anyopaque = null;
    if ((req_comp < @as(i32, 0)) or (req_comp > @as(i32, 4))) return @ptrCast(@alignCast(@as([*c]u8, @ptrFromInt(@as(usize, @intCast(@intFromPtr(if (stbi__err("bad req_comp") != 0) @as(?*anyopaque, null) else @as(?*anyopaque, null))))))));
    if (stbi__parse_png_file(p, STBI__SCAN_load, req_comp) != 0) {
        if (p.*.depth <= @as(i32, 8)) {
            ri.*.bits_per_channel = 8;
        } else if (p.*.depth == @as(i32, 16)) {
            ri.*.bits_per_channel = 16;
        } else return @ptrCast(@alignCast(@as([*c]u8, @ptrFromInt(@as(usize, @intCast(@intFromPtr(if (stbi__err("bad bits_per_channel") != 0) @as(?*anyopaque, null) else @as(?*anyopaque, null))))))));
        result = @ptrCast(@alignCast(p.*.out));
        p.*.out = null;
        if ((req_comp != 0) and (req_comp != p.*.s.*.img_out_n)) {
            if (ri.*.bits_per_channel == @as(i32, 8)) {
                result = @ptrCast(@alignCast(stbi__convert_format(@ptrCast(@alignCast(result)), p.*.s.*.img_out_n, req_comp, p.*.s.*.img_x, p.*.s.*.img_y)));
            } else {
                result = @ptrCast(@alignCast(stbi__convert_format16(@ptrCast(@alignCast(result)), p.*.s.*.img_out_n, req_comp, p.*.s.*.img_x, p.*.s.*.img_y)));
            }
            p.*.s.*.img_out_n = req_comp;
            if (result == @as(?*anyopaque, null)) return result;
        }
        x.* = @bitCast(@as(u32, @truncate(p.*.s.*.img_x)));
        y.* = @bitCast(@as(u32, @truncate(p.*.s.*.img_y)));
        if (n != null) {
            n.* = p.*.s.*.img_n;
        }
    }
    free(@ptrCast(@alignCast(p.*.out)));
    p.*.out = null;
    free(@ptrCast(@alignCast(p.*.expanded)));
    p.*.expanded = null;
    free(@ptrCast(@alignCast(p.*.idata)));
    p.*.idata = null;
    return result;
}
fn stbi__png_load(s: [*c]stbi__context, x: [*c]i32, y: [*c]i32, comp: [*c]i32, req_comp: i32, ri: [*c]stbi__result_info) callconv(.c) ?*anyopaque {
    var p: stbi__png = undefined;
    p.s = s;
    return stbi__do_png(&p, x, y, comp, req_comp, ri);
}
fn stbi__png_test(s: [*c]stbi__context) callconv(.c) i32 {
    var r: i32 = undefined;
    r = stbi__check_png_header(s);
    stbi__rewind(s);
    return r;
}
fn stbi__bmp_test_raw(s: [*c]stbi__context) callconv(.c) i32 {
    var r: i32 = undefined;
    var sz: i32 = undefined;
    if (@as(i32, stbi__get8(s)) != @as(i32, 'B')) return 0;
    if (@as(i32, stbi__get8(s)) != @as(i32, 'M')) return 0;
    _ = stbi__get32le(s);
    _ = stbi__get16le(s);
    _ = stbi__get16le(s);
    _ = stbi__get32le(s);
    sz = @bitCast(@as(u32, @truncate(stbi__get32le(s))));
    r = @intFromBool(((((sz == @as(i32, 12)) or (sz == @as(i32, 40))) or (sz == @as(i32, 56))) or (sz == @as(i32, 108))) or (sz == @as(i32, 124)));
    return r;
}
fn stbi__bmp_test(s: [*c]stbi__context) callconv(.c) i32 {
    const r: i32 = stbi__bmp_test_raw(s);
    stbi__rewind(s);
    return r;
}
fn stbi__high_bit(arg_z: u32) callconv(.c) i32 {
    var z = arg_z;
    var n: i32 = 0;
    if (z == @as(u32, 0)) return -@as(i32, 1);
    if (z >= @as(u32, 65536)) {
        n += 16;
        z >>= @intCast(@as(u32, 16));
    }
    if (z >= @as(u32, 256)) {
        n += 8;
        z >>= @intCast(@as(u32, 8));
    }
    if (z >= @as(u32, 16)) {
        n += 4;
        z >>= @intCast(@as(u32, 4));
    }
    if (z >= @as(u32, 4)) {
        n += 2;
        z >>= @intCast(@as(u32, 2));
    }
    if (z >= @as(u32, 2)) {
        n += 1;
    }
    return n;
}
fn stbi__bitcount(arg_a: u32) callconv(.c) i32 {
    var a = arg_a;
    a = (a & @as(u32, 1431655765)) +% ((a >> @intCast(@as(u32, 1))) & @as(u32, 1431655765));
    a = (a & @as(u32, 858993459)) +% ((a >> @intCast(@as(u32, 2))) & @as(u32, 858993459));
    a = (a +% (a >> @intCast(@as(u32, 4)))) & @as(u32, 252645135);
    a = a +% (a >> @intCast(@as(u32, 8)));
    a = a +% (a >> @intCast(@as(u32, 16)));
    return @bitCast(@as(u32, @truncate(a & @as(u32, 255))));
}
fn stbi__shiftsigned(arg_v: u32, shift: i32, bits: i32) callconv(.c) i32 {
    var v = arg_v;
    const static_local_mul_table = struct {
        var mul_table: [9]u32 = [9]u32{
            0, 255, 85, 73, 17, 33, 65, 129, 1,
        };
    };
    const static_local_shift_table = struct {
        var shift_table: [9]u32 = [9]u32{
            0, 0, 0, 1, 0, 2, 4, 6, 0,
        };
    };
    if (shift < @as(i32, 0)) {
        v <<= @intCast(@as(u32, @bitCast(@as(i32, -shift))));
    } else {
        v >>= @intCast(@as(u32, @bitCast(@as(i32, shift))));
    }
    _ = !!(v < @as(u32, 256)) or ((blk: {
        _assert("v < 256", "tools\\dataset\\vendor\\stb_image.h", 5409);
        break :blk 0;
    }) != 0);
    v >>= @intCast(@as(u32, @bitCast(@as(i32, @as(i32, 8) - bits))));
    _ = !!((bits >= @as(i32, 0)) and (bits <= @as(i32, 8))) or ((blk: {
        _assert("bits >= 0 && bits <= 8", "tools\\dataset\\vendor\\stb_image.h", 5411);
        break :blk 0;
    }) != 0);
    return @bitCast(@as(u32, @truncate(@as(u32, @bitCast(@as(i32, @as(i32, @bitCast(@as(u32, @truncate(v *% static_local_mul_table.mul_table[@bitCast(@as(isize, @intCast(bits)))]))))))) >> @intCast(static_local_shift_table.shift_table[@bitCast(@as(isize, @intCast(bits)))]))));
}
const stbi__bmp_data = extern struct {
    bpp: i32 = 0,
    offset: i32 = 0,
    hsz: i32 = 0,
    mr: u32 = 0,
    mg: u32 = 0,
    mb: u32 = 0,
    ma: u32 = 0,
    all_a: u32 = 0,
    extra_read: i32 = 0,
};
fn stbi__bmp_set_mask_defaults(info: [*c]stbi__bmp_data, compress: i32) callconv(.c) i32 {
    if (compress == @as(i32, 3)) return 1;
    if (compress == @as(i32, 0)) {
        if (info.*.bpp == @as(i32, 16)) {
            info.*.mr = @as(u32, 31) << @intCast(@as(u32, 10));
            info.*.mg = @as(u32, 31) << @intCast(@as(u32, 5));
            info.*.mb = @as(u32, 31) << @intCast(@as(u32, 0));
        } else if (info.*.bpp == @as(i32, 32)) {
            info.*.mr = @as(u32, 255) << @intCast(@as(u32, 16));
            info.*.mg = @as(u32, 255) << @intCast(@as(u32, 8));
            info.*.mb = @as(u32, 255) << @intCast(@as(u32, 0));
            info.*.ma = @as(u32, 255) << @intCast(@as(u32, 24));
            info.*.all_a = 0;
        } else {
            info.*.mr = blk: {
                const tmp = blk_1: {
                    const tmp_2 = blk_2: {
                        const tmp_3 = @as(u32, 0);
                        info.*.ma = tmp_3;
                        break :blk_2 tmp_3;
                    };
                    info.*.mb = tmp_2;
                    break :blk_1 tmp_2;
                };
                info.*.mg = tmp;
                break :blk tmp;
            };
        }
        return 1;
    }
    return 0;
}
fn stbi__bmp_parse_header(s: [*c]stbi__context, info: [*c]stbi__bmp_data) callconv(.c) ?*anyopaque {
    var hsz: i32 = undefined;
    if ((@as(i32, stbi__get8(s)) != @as(i32, 'B')) or (@as(i32, stbi__get8(s)) != @as(i32, 'M'))) return @ptrCast(@alignCast(@as([*c]u8, @ptrFromInt(@as(usize, @intCast(@intFromPtr(if (stbi__err("not BMP") != 0) @as(?*anyopaque, null) else @as(?*anyopaque, null))))))));
    _ = stbi__get32le(s);
    _ = stbi__get16le(s);
    _ = stbi__get16le(s);
    info.*.offset = @bitCast(@as(u32, @truncate(stbi__get32le(s))));
    info.*.hsz = blk: {
        const tmp = @as(i32, @bitCast(@as(u32, @truncate(stbi__get32le(s)))));
        hsz = tmp;
        break :blk tmp;
    };
    info.*.mr = blk: {
        const tmp = blk_1: {
            const tmp_2 = blk_2: {
                const tmp_3 = @as(u32, 0);
                info.*.ma = tmp_3;
                break :blk_2 tmp_3;
            };
            info.*.mb = tmp_2;
            break :blk_1 tmp_2;
        };
        info.*.mg = tmp;
        break :blk tmp;
    };
    info.*.extra_read = 14;
    if (info.*.offset < @as(i32, 0)) return @ptrCast(@alignCast(@as([*c]u8, @ptrFromInt(@as(usize, @intCast(@intFromPtr(if (stbi__err("bad BMP") != 0) @as(?*anyopaque, null) else @as(?*anyopaque, null))))))));
    if (((((hsz != @as(i32, 12)) and (hsz != @as(i32, 40))) and (hsz != @as(i32, 56))) and (hsz != @as(i32, 108))) and (hsz != @as(i32, 124))) return @ptrCast(@alignCast(@as([*c]u8, @ptrFromInt(@as(usize, @intCast(@intFromPtr(if (stbi__err("unknown BMP") != 0) @as(?*anyopaque, null) else @as(?*anyopaque, null))))))));
    if (hsz == @as(i32, 12)) {
        s.*.img_x = @bitCast(@as(i32, stbi__get16le(s)));
        s.*.img_y = @bitCast(@as(i32, stbi__get16le(s)));
    } else {
        s.*.img_x = stbi__get32le(s);
        s.*.img_y = stbi__get32le(s);
    }
    if (stbi__get16le(s) != @as(i32, 1)) return @ptrCast(@alignCast(@as([*c]u8, @ptrFromInt(@as(usize, @intCast(@intFromPtr(if (stbi__err("bad BMP") != 0) @as(?*anyopaque, null) else @as(?*anyopaque, null))))))));
    info.*.bpp = stbi__get16le(s);
    if (hsz != @as(i32, 12)) {
        const compress: i32 = @bitCast(@as(u32, @truncate(stbi__get32le(s))));
        if ((compress == @as(i32, 1)) or (compress == @as(i32, 2))) return @ptrCast(@alignCast(@as([*c]u8, @ptrFromInt(@as(usize, @intCast(@intFromPtr(if (stbi__err("BMP RLE") != 0) @as(?*anyopaque, null) else @as(?*anyopaque, null))))))));
        if (compress >= @as(i32, 4)) return @ptrCast(@alignCast(@as([*c]u8, @ptrFromInt(@as(usize, @intCast(@intFromPtr(if (stbi__err("BMP JPEG/PNG") != 0) @as(?*anyopaque, null) else @as(?*anyopaque, null))))))));
        if (((compress == @as(i32, 3)) and (info.*.bpp != @as(i32, 16))) and (info.*.bpp != @as(i32, 32))) return @ptrCast(@alignCast(@as([*c]u8, @ptrFromInt(@as(usize, @intCast(@intFromPtr(if (stbi__err("bad BMP") != 0) @as(?*anyopaque, null) else @as(?*anyopaque, null))))))));
        _ = stbi__get32le(s);
        _ = stbi__get32le(s);
        _ = stbi__get32le(s);
        _ = stbi__get32le(s);
        _ = stbi__get32le(s);
        if ((hsz == @as(i32, 40)) or (hsz == @as(i32, 56))) {
            if (hsz == @as(i32, 56)) {
                _ = stbi__get32le(s);
                _ = stbi__get32le(s);
                _ = stbi__get32le(s);
                _ = stbi__get32le(s);
            }
            if ((info.*.bpp == @as(i32, 16)) or (info.*.bpp == @as(i32, 32))) {
                if (compress == @as(i32, 0)) {
                    _ = stbi__bmp_set_mask_defaults(info, compress);
                } else if (compress == @as(i32, 3)) {
                    info.*.mr = stbi__get32le(s);
                    info.*.mg = stbi__get32le(s);
                    info.*.mb = stbi__get32le(s);
                    info.*.extra_read += 12;
                    if ((info.*.mr == info.*.mg) and (info.*.mg == info.*.mb)) {
                        return @ptrCast(@alignCast(@as([*c]u8, @ptrFromInt(@as(usize, @intCast(@intFromPtr(if (stbi__err("bad BMP") != 0) @as(?*anyopaque, null) else @as(?*anyopaque, null))))))));
                    }
                } else return @ptrCast(@alignCast(@as([*c]u8, @ptrFromInt(@as(usize, @intCast(@intFromPtr(if (stbi__err("bad BMP") != 0) @as(?*anyopaque, null) else @as(?*anyopaque, null))))))));
            }
        } else {
            var i: i32 = undefined;
            if ((hsz != @as(i32, 108)) and (hsz != @as(i32, 124))) return @ptrCast(@alignCast(@as([*c]u8, @ptrFromInt(@as(usize, @intCast(@intFromPtr(if (stbi__err("bad BMP") != 0) @as(?*anyopaque, null) else @as(?*anyopaque, null))))))));
            info.*.mr = stbi__get32le(s);
            info.*.mg = stbi__get32le(s);
            info.*.mb = stbi__get32le(s);
            info.*.ma = stbi__get32le(s);
            if (compress != @as(i32, 3)) {
                _ = stbi__bmp_set_mask_defaults(info, compress);
            }
            _ = stbi__get32le(s);
            {
                i = 0;
                while (i < @as(i32, 12)) : (i += 1) {
                    _ = stbi__get32le(s);
                }
            }
            if (hsz == @as(i32, 124)) {
                _ = stbi__get32le(s);
                _ = stbi__get32le(s);
                _ = stbi__get32le(s);
                _ = stbi__get32le(s);
            }
        }
    }
    return @ptrFromInt(@as(usize, @intCast(@as(i32, 1))));
}
fn stbi__bmp_load(s: [*c]stbi__context, x: [*c]i32, y: [*c]i32, comp: [*c]i32, req_comp: i32, ri: [*c]stbi__result_info) callconv(.c) ?*anyopaque {
    var out: [*c]u8 = undefined;
    var mr: u32 = 0;
    var mg: u32 = 0;
    var mb: u32 = 0;
    var ma: u32 = 0;
    var all_a: u32 = undefined;
    var pal: [256][4]u8 = undefined;
    var psize: i32 = 0;
    var i: i32 = undefined;
    var j: i32 = undefined;
    var width: i32 = undefined;
    var flip_vertically: i32 = undefined;
    var pad: i32 = undefined;
    var target: i32 = undefined;
    var info: stbi__bmp_data = undefined;
    _ = @sizeOf(@TypeOf(ri));
    info.all_a = 255;
    if (stbi__bmp_parse_header(s, &info) == @as(?*anyopaque, null)) return null;
    flip_vertically = @intFromBool(@as(i32, @bitCast(@as(u32, @truncate(s.*.img_y)))) > @as(i32, 0));
    s.*.img_y = @bitCast(@as(i32, abs(@bitCast(@as(u32, @truncate(s.*.img_y))))));
    if (s.*.img_y > @as(u32, @bitCast(@as(i32, @as(i32, 1) << @intCast(@as(i32, 24)))))) return @ptrCast(@alignCast(@as([*c]u8, @ptrFromInt(@as(usize, @intCast(@intFromPtr(if (stbi__err("too large") != 0) @as(?*anyopaque, null) else @as(?*anyopaque, null))))))));
    if (s.*.img_x > @as(u32, @bitCast(@as(i32, @as(i32, 1) << @intCast(@as(i32, 24)))))) return @ptrCast(@alignCast(@as([*c]u8, @ptrFromInt(@as(usize, @intCast(@intFromPtr(if (stbi__err("too large") != 0) @as(?*anyopaque, null) else @as(?*anyopaque, null))))))));
    mr = info.mr;
    mg = info.mg;
    mb = info.mb;
    ma = info.ma;
    all_a = info.all_a;
    if (info.hsz == @as(i32, 12)) {
        if (info.bpp < @as(i32, 24)) {
            psize = @divTrunc((info.offset - info.extra_read) - @as(i32, 24), @as(i32, 3));
        }
    } else {
        if (info.bpp < @as(i32, 16)) {
            psize = ((info.offset - info.extra_read) - info.hsz) >> @intCast(@as(i32, 2));
        }
    }
    if (psize == @as(i32, 0)) {
        const bytes_read_so_far: i32 = s.*.callback_already_read + @as(i32, @truncate(@divExact(@as(i64, @bitCast(@intFromPtr(s.*.img_buffer) -% @intFromPtr(s.*.img_buffer_original))), @sizeOf(u8))));
        const header_limit: i32 = 1024;
        const extra_data_limit: i32 = @as(i32, 256) * @as(i32, 4);
        if ((bytes_read_so_far <= @as(i32, 0)) or (bytes_read_so_far > header_limit)) {
            return @ptrCast(@alignCast(@as([*c]u8, @ptrFromInt(@as(usize, @intCast(@intFromPtr(if (stbi__err("bad header") != 0) @as(?*anyopaque, null) else @as(?*anyopaque, null))))))));
        }
        if ((info.offset < bytes_read_so_far) or ((info.offset - bytes_read_so_far) > extra_data_limit)) {
            return @ptrCast(@alignCast(@as([*c]u8, @ptrFromInt(@as(usize, @intCast(@intFromPtr(if (stbi__err("bad offset") != 0) @as(?*anyopaque, null) else @as(?*anyopaque, null))))))));
        } else {
            stbi__skip(s, info.offset - bytes_read_so_far);
        }
    }
    if ((info.bpp == @as(i32, 24)) and (ma == @as(u32, 4278190080))) {
        s.*.img_n = 3;
    } else {
        s.*.img_n = if (ma != 0) @as(i32, 4) else @as(i32, 3);
    }
    if ((req_comp != 0) and (req_comp >= @as(i32, 3))) {
        target = req_comp;
    } else {
        target = s.*.img_n;
    }
    if (!(stbi__mad3sizes_valid(target, @bitCast(@as(u32, @truncate(s.*.img_x))), @bitCast(@as(u32, @truncate(s.*.img_y))), 0) != 0)) return @ptrCast(@alignCast(@as([*c]u8, @ptrFromInt(@as(usize, @intCast(@intFromPtr(if (stbi__err("too large") != 0) @as(?*anyopaque, null) else @as(?*anyopaque, null))))))));
    out = @ptrCast(@alignCast(stbi__malloc_mad3(target, @bitCast(@as(u32, @truncate(s.*.img_x))), @bitCast(@as(u32, @truncate(s.*.img_y))), 0)));
    if (!(out != null)) return @ptrCast(@alignCast(@as([*c]u8, @ptrFromInt(@as(usize, @intCast(@intFromPtr(if (stbi__err("outofmem") != 0) @as(?*anyopaque, null) else @as(?*anyopaque, null))))))));
    if (info.bpp < @as(i32, 16)) {
        var z: i32 = 0;
        if ((psize == @as(i32, 0)) or (psize > @as(i32, 256))) {
            free(@ptrCast(@alignCast(out)));
            return @ptrCast(@alignCast(@as([*c]u8, @ptrFromInt(@as(usize, @intCast(@intFromPtr(if (stbi__err("invalid") != 0) @as(?*anyopaque, null) else @as(?*anyopaque, null))))))));
        }
        {
            i = 0;
            while (i < psize) : (i += 1) {
                pal[@bitCast(@as(isize, @intCast(i)))][@as(i32, 2)] = stbi__get8(s);
                pal[@bitCast(@as(isize, @intCast(i)))][@as(i32, 1)] = stbi__get8(s);
                pal[@bitCast(@as(isize, @intCast(i)))][@as(i32, 0)] = stbi__get8(s);
                if (info.hsz != @as(i32, 12)) {
                    _ = stbi__get8(s);
                }
                pal[@bitCast(@as(isize, @intCast(i)))][@as(i32, 3)] = 255;
            }
        }
        stbi__skip(s, ((info.offset - info.extra_read) - info.hsz) - (psize * (if (info.hsz == @as(i32, 12)) @as(i32, 3) else @as(i32, 4))));
        if (info.bpp == @as(i32, 1)) {
            width = @bitCast(@as(u32, @truncate((s.*.img_x +% @as(u32, 7)) >> @intCast(@as(u32, 3)))));
        } else if (info.bpp == @as(i32, 4)) {
            width = @bitCast(@as(u32, @truncate((s.*.img_x +% @as(u32, 1)) >> @intCast(@as(u32, 1)))));
        } else if (info.bpp == @as(i32, 8)) {
            width = @bitCast(@as(u32, @truncate(s.*.img_x)));
        } else {
            free(@ptrCast(@alignCast(out)));
            return @ptrCast(@alignCast(@as([*c]u8, @ptrFromInt(@as(usize, @intCast(@intFromPtr(if (stbi__err("bad bpp") != 0) @as(?*anyopaque, null) else @as(?*anyopaque, null))))))));
        }
        pad = -width & @as(i32, 3);
        if (info.bpp == @as(i32, 1)) {
            {
                j = 0;
                while (j < @as(i32, @bitCast(@as(u32, @truncate(s.*.img_y))))) : (j += 1) {
                    var bit_offset: i32 = 7;
                    var v: i32 = stbi__get8(s);
                    {
                        i = 0;
                        while (i < @as(i32, @bitCast(@as(u32, @truncate(s.*.img_x))))) : (i += 1) {
                            const color: i32 = (v >> @intCast(bit_offset)) & @as(i32, 1);
                            out[
                                @bitCast(@as(isize, @intCast(blk: {
                                    const ref = &z;
                                    const tmp = ref.*;
                                    ref.* += 1;
                                    break :blk tmp;
                                })))
                            ] = pal[@bitCast(@as(isize, @intCast(color)))][@as(i32, 0)];
                            out[
                                @bitCast(@as(isize, @intCast(blk: {
                                    const ref = &z;
                                    const tmp = ref.*;
                                    ref.* += 1;
                                    break :blk tmp;
                                })))
                            ] = pal[@bitCast(@as(isize, @intCast(color)))][@as(i32, 1)];
                            out[
                                @bitCast(@as(isize, @intCast(blk: {
                                    const ref = &z;
                                    const tmp = ref.*;
                                    ref.* += 1;
                                    break :blk tmp;
                                })))
                            ] = pal[@bitCast(@as(isize, @intCast(color)))][@as(i32, 2)];
                            if (target == @as(i32, 4)) {
                                out[
                                    @bitCast(@as(isize, @intCast(blk: {
                                        const ref = &z;
                                        const tmp = ref.*;
                                        ref.* += 1;
                                        break :blk tmp;
                                    })))
                                ] = 255;
                            }
                            if ((i + @as(i32, 1)) == @as(i32, @bitCast(@as(u32, @truncate(s.*.img_x))))) break;
                            if ((blk: {
                                const ref = &bit_offset;
                                ref.* -= 1;
                                break :blk ref.*;
                            }) < @as(i32, 0)) {
                                bit_offset = 7;
                                v = stbi__get8(s);
                            }
                        }
                    }
                    stbi__skip(s, pad);
                }
            }
        } else {
            {
                j = 0;
                while (j < @as(i32, @bitCast(@as(u32, @truncate(s.*.img_y))))) : (j += 1) {
                    {
                        i = 0;
                        while (i < @as(i32, @bitCast(@as(u32, @truncate(s.*.img_x))))) : (i += 2) {
                            var v: i32 = stbi__get8(s);
                            var v2: i32 = 0;
                            if (info.bpp == @as(i32, 4)) {
                                v2 = v & @as(i32, 15);
                                v >>= @intCast(@as(i32, 4));
                            }
                            out[
                                @bitCast(@as(isize, @intCast(blk: {
                                    const ref = &z;
                                    const tmp = ref.*;
                                    ref.* += 1;
                                    break :blk tmp;
                                })))
                            ] = pal[@bitCast(@as(isize, @intCast(v)))][@as(i32, 0)];
                            out[
                                @bitCast(@as(isize, @intCast(blk: {
                                    const ref = &z;
                                    const tmp = ref.*;
                                    ref.* += 1;
                                    break :blk tmp;
                                })))
                            ] = pal[@bitCast(@as(isize, @intCast(v)))][@as(i32, 1)];
                            out[
                                @bitCast(@as(isize, @intCast(blk: {
                                    const ref = &z;
                                    const tmp = ref.*;
                                    ref.* += 1;
                                    break :blk tmp;
                                })))
                            ] = pal[@bitCast(@as(isize, @intCast(v)))][@as(i32, 2)];
                            if (target == @as(i32, 4)) {
                                out[
                                    @bitCast(@as(isize, @intCast(blk: {
                                        const ref = &z;
                                        const tmp = ref.*;
                                        ref.* += 1;
                                        break :blk tmp;
                                    })))
                                ] = 255;
                            }
                            if ((i + @as(i32, 1)) == @as(i32, @bitCast(@as(u32, @truncate(s.*.img_x))))) break;
                            v = if (info.bpp == @as(i32, 8)) @as(i32, stbi__get8(s)) else v2;
                            out[
                                @bitCast(@as(isize, @intCast(blk: {
                                    const ref = &z;
                                    const tmp = ref.*;
                                    ref.* += 1;
                                    break :blk tmp;
                                })))
                            ] = pal[@bitCast(@as(isize, @intCast(v)))][@as(i32, 0)];
                            out[
                                @bitCast(@as(isize, @intCast(blk: {
                                    const ref = &z;
                                    const tmp = ref.*;
                                    ref.* += 1;
                                    break :blk tmp;
                                })))
                            ] = pal[@bitCast(@as(isize, @intCast(v)))][@as(i32, 1)];
                            out[
                                @bitCast(@as(isize, @intCast(blk: {
                                    const ref = &z;
                                    const tmp = ref.*;
                                    ref.* += 1;
                                    break :blk tmp;
                                })))
                            ] = pal[@bitCast(@as(isize, @intCast(v)))][@as(i32, 2)];
                            if (target == @as(i32, 4)) {
                                out[
                                    @bitCast(@as(isize, @intCast(blk: {
                                        const ref = &z;
                                        const tmp = ref.*;
                                        ref.* += 1;
                                        break :blk tmp;
                                    })))
                                ] = 255;
                            }
                        }
                    }
                    stbi__skip(s, pad);
                }
            }
        }
    } else {
        var rshift: i32 = 0;
        var gshift: i32 = 0;
        var bshift: i32 = 0;
        var ashift: i32 = 0;
        var rcount: i32 = 0;
        var gcount: i32 = 0;
        var bcount: i32 = 0;
        var acount: i32 = 0;
        var z: i32 = 0;
        var easy: i32 = 0;
        stbi__skip(s, (info.offset - info.extra_read) - info.hsz);
        if (info.bpp == @as(i32, 24)) {
            width = @bitCast(@as(u32, @truncate(@as(u32, 3) *% s.*.img_x)));
        } else if (info.bpp == @as(i32, 16)) {
            width = @bitCast(@as(u32, @truncate(@as(u32, 2) *% s.*.img_x)));
        } else {
            width = 0;
        }
        pad = -width & @as(i32, 3);
        if (info.bpp == @as(i32, 24)) {
            easy = 1;
        } else if (info.bpp == @as(i32, 32)) {
            if ((((mb == @as(u32, 255)) and (mg == @as(u32, 65280))) and (mr == @as(u32, 16711680))) and (ma == @as(u32, 4278190080))) {
                easy = 2;
            }
        }
        if (!(easy != 0)) {
            if ((!(mr != 0) or !(mg != 0)) or !(mb != 0)) {
                free(@ptrCast(@alignCast(out)));
                return @ptrCast(@alignCast(@as([*c]u8, @ptrFromInt(@as(usize, @intCast(@intFromPtr(if (stbi__err("bad masks") != 0) @as(?*anyopaque, null) else @as(?*anyopaque, null))))))));
            }
            rshift = stbi__high_bit(mr) - @as(i32, 7);
            rcount = stbi__bitcount(mr);
            gshift = stbi__high_bit(mg) - @as(i32, 7);
            gcount = stbi__bitcount(mg);
            bshift = stbi__high_bit(mb) - @as(i32, 7);
            bcount = stbi__bitcount(mb);
            ashift = stbi__high_bit(ma) - @as(i32, 7);
            acount = stbi__bitcount(ma);
            if ((((rcount > @as(i32, 8)) or (gcount > @as(i32, 8))) or (bcount > @as(i32, 8))) or (acount > @as(i32, 8))) {
                free(@ptrCast(@alignCast(out)));
                return @ptrCast(@alignCast(@as([*c]u8, @ptrFromInt(@as(usize, @intCast(@intFromPtr(if (stbi__err("bad masks") != 0) @as(?*anyopaque, null) else @as(?*anyopaque, null))))))));
            }
        }
        {
            j = 0;
            while (j < @as(i32, @bitCast(@as(u32, @truncate(s.*.img_y))))) : (j += 1) {
                if (easy != 0) {
                    {
                        i = 0;
                        while (i < @as(i32, @bitCast(@as(u32, @truncate(s.*.img_x))))) : (i += 1) {
                            var a: u8 = undefined;
                            out[@bitCast(@as(isize, @intCast(z + @as(i32, 2))))] = stbi__get8(s);
                            out[@bitCast(@as(isize, @intCast(z + @as(i32, 1))))] = stbi__get8(s);
                            out[@bitCast(@as(isize, @intCast(z + @as(i32, 0))))] = stbi__get8(s);
                            z += 3;
                            a = @bitCast(@as(i8, @truncate(if (easy == @as(i32, 2)) @as(i32, stbi__get8(s)) else @as(i32, 255))));
                            all_a |= @bitCast(@as(i32, @as(i32, a)));
                            if (target == @as(i32, 4)) {
                                out[
                                    @bitCast(@as(isize, @intCast(blk: {
                                        const ref = &z;
                                        const tmp = ref.*;
                                        ref.* += 1;
                                        break :blk tmp;
                                    })))
                                ] = a;
                            }
                        }
                    }
                } else {
                    const bpp: i32 = info.bpp;
                    {
                        i = 0;
                        while (i < @as(i32, @bitCast(@as(u32, @truncate(s.*.img_x))))) : (i += 1) {
                            const v: u32 = if (bpp == @as(i32, 16)) @as(u32, @bitCast(@as(i32, stbi__get16le(s)))) else stbi__get32le(s);
                            var a: u32 = undefined;
                            out[
                                @bitCast(@as(isize, @intCast(blk: {
                                    const ref = &z;
                                    const tmp = ref.*;
                                    ref.* += 1;
                                    break :blk tmp;
                                })))
                            ] = @as(u8, @bitCast(@as(i8, @truncate(stbi__shiftsigned(v & mr, rshift, rcount) & @as(i32, 255)))));
                            out[
                                @bitCast(@as(isize, @intCast(blk: {
                                    const ref = &z;
                                    const tmp = ref.*;
                                    ref.* += 1;
                                    break :blk tmp;
                                })))
                            ] = @as(u8, @bitCast(@as(i8, @truncate(stbi__shiftsigned(v & mg, gshift, gcount) & @as(i32, 255)))));
                            out[
                                @bitCast(@as(isize, @intCast(blk: {
                                    const ref = &z;
                                    const tmp = ref.*;
                                    ref.* += 1;
                                    break :blk tmp;
                                })))
                            ] = @as(u8, @bitCast(@as(i8, @truncate(stbi__shiftsigned(v & mb, bshift, bcount) & @as(i32, 255)))));
                            a = @bitCast(@as(i32, if (ma != 0) stbi__shiftsigned(v & ma, ashift, acount) else @as(i32, 255)));
                            all_a |= a;
                            if (target == @as(i32, 4)) {
                                out[
                                    @bitCast(@as(isize, @intCast(blk: {
                                        const ref = &z;
                                        const tmp = ref.*;
                                        ref.* += 1;
                                        break :blk tmp;
                                    })))
                                ] = @as(u8, @truncate(a & @as(u32, 255)));
                            }
                        }
                    }
                }
                stbi__skip(s, pad);
            }
        }
    }
    if ((target == @as(i32, 4)) and (all_a == @as(u32, 0))) {
        i = @bitCast(@as(u32, @truncate(((@as(u32, 4) *% s.*.img_x) *% s.*.img_y) -% @as(u32, 1))));
        while (i >= @as(i32, 0)) : (i -= 4) {
            out[@bitCast(@as(isize, @intCast(i)))] = 255;
        }
    }
    if (flip_vertically != 0) {
        var t: u8 = undefined;
        {
            j = 0;
            while (j < (@as(i32, @bitCast(@as(u32, @truncate(s.*.img_y)))) >> @intCast(@as(i32, 1)))) : (j += 1) {
                var p1: [*c]u8 = out + ((@as(u32, @bitCast(@as(i32, j))) *% s.*.img_x) *% @as(u32, @bitCast(@as(i32, target))));
                var p2: [*c]u8 = out + ((((s.*.img_y -% @as(u32, 1)) -% @as(u32, @bitCast(@as(i32, j)))) *% s.*.img_x) *% @as(u32, @bitCast(@as(i32, target))));
                {
                    i = 0;
                    while (i < (@as(i32, @bitCast(@as(u32, @truncate(s.*.img_x)))) * target)) : (i += 1) {
                        t = p1[@bitCast(@as(isize, @intCast(i)))];
                        p1[@bitCast(@as(isize, @intCast(i)))] = p2[@bitCast(@as(isize, @intCast(i)))];
                        p2[@bitCast(@as(isize, @intCast(i)))] = t;
                    }
                }
            }
        }
    }
    if ((req_comp != 0) and (req_comp != target)) {
        out = stbi__convert_format(out, target, req_comp, s.*.img_x, s.*.img_y);
        if (@as(?*anyopaque, @ptrCast(@alignCast(out))) == @as(?*anyopaque, null)) return @ptrCast(@alignCast(out));
    }
    x.* = @bitCast(@as(u32, @truncate(s.*.img_x)));
    y.* = @bitCast(@as(u32, @truncate(s.*.img_y)));
    if (comp != null) {
        comp.* = s.*.img_n;
    }
    return @ptrCast(@alignCast(out));
}
const stbi_write_func = fn (context: ?*anyopaque, data: ?*anyopaque, size: i32) callconv(.c) void;
var stbi_write_png_compression_level: i32 = 8;
var stbi_write_force_png_filter: i32 = -@as(i32, 1);
var stbi__flip_vertically_on_write: i32 = 0;
const stbi__write_context = extern struct {
    func: ?*const stbi_write_func = null,
    context: ?*anyopaque = null,
    buffer: [64]u8 = std.mem.zeroes([64]u8),
    buf_used: i32 = 0,
};
fn stbi__start_write_callbacks(s: [*c]stbi__write_context, c: ?*const stbi_write_func, context: ?*anyopaque) callconv(.c) void {
    s.*.func = c;
    s.*.context = context;
}
fn stbiw__putc(s: [*c]stbi__write_context, arg_c: u8) callconv(.c) void {
    var c = arg_c;
    s.*.func.?(s.*.context, @ptrCast(@alignCast(&c)), 1);
}
// Use Zig's compressor so allocation failures unwind without C stretchy-buffer macros.
fn stbi_zlib_compress(data: [*c]u8, data_len: i32, out_len: [*c]i32, _: i32) [*c]u8 {
    var output: std.Io.Writer.Allocating = std.Io.Writer.Allocating.initCapacity(allocator, 1024) catch return null;
    defer output.deinit();
    const window = allocator.alloc(u8, std.compress.flate.max_window_len) catch return null;
    defer allocator.free(window);
    const compressor = allocator.create(std.compress.flate.Compress) catch return null;
    defer allocator.destroy(compressor);
    compressor.* = std.compress.flate.Compress.init(&output.writer, window, .zlib, std.compress.flate.Compress.Options.default) catch return null;
    compressor.writer.writeAll(data[0..@intCast(data_len)]) catch return null;
    compressor.finish() catch return null;
    const encoded = output.written();
    out_len.* = std.math.cast(i32, encoded.len) orelse return null;
    const result: [*c]u8 = @ptrCast(malloc(encoded.len) orelse return null);
    @memcpy(result[0..encoded.len], encoded);
    return result;
}
fn stbiw__crc32(buffer: [*c]u8, len: i32) callconv(.c) u32 {
    const static_local_crc_table = struct {
        var crc_table: [256]u32 = [256]u32{
            0,          1996959894, 3993919788, 2567524794, 124634137,  1886057615, 3915621685, 2657392035, 249268274,  2044508324, 3772115230, 2547177864, 162941995,  2125561021, 3887607047, 2428444049,
            498536548,  1789927666, 4089016648, 2227061214, 450548861,  1843258603, 4107580753, 2211677639, 325883990,  1684777152, 4251122042, 2321926636, 335633487,  1661365465, 4195302755, 2366115317,
            997073096,  1281953886, 3579855332, 2724688242, 1006888145, 1258607687, 3524101629, 2768942443, 901097722,  1119000684, 3686517206, 2898065728, 853044451,  1172266101, 3705015759, 2882616665,
            651767980,  1373503546, 3369554304, 3218104598, 565507253,  1454621731, 3485111705, 3099436303, 671266974,  1594198024, 3322730930, 2970347812, 795835527,  1483230225, 3244367275, 3060149565,
            1994146192, 31158534,   2563907772, 4023717930, 1907459465, 112637215,  2680153253, 3904427059, 2013776290, 251722036,  2517215374, 3775830040, 2137656763, 141376813,  2439277719, 3865271297,
            1802195444, 476864866,  2238001368, 4066508878, 1812370925, 453092731,  2181625025, 4111451223, 1706088902, 314042704,  2344532202, 4240017532, 1658658271, 366619977,  2362670323, 4224994405,
            1303535960, 984961486,  2747007092, 3569037538, 1256170817, 1037604311, 2765210733, 3554079995, 1131014506, 879679996,  2909243462, 3663771856, 1141124467, 855842277,  2852801631, 3708648649,
            1342533948, 654459306,  3188396048, 3373015174, 1466479909, 544179635,  3110523913, 3462522015, 1591671054, 702138776,  2966460450, 3352799412, 1504918807, 783551873,  3082640443, 3233442989,
            3988292384, 2596254646, 62317068,   1957810842, 3939845945, 2647816111, 81470997,   1943803523, 3814918930, 2489596804, 225274430,  2053790376, 3826175755, 2466906013, 167816743,  2097651377,
            4027552580, 2265490386, 503444072,  1762050814, 4150417245, 2154129355, 426522225,  1852507879, 4275313526, 2312317920, 282753626,  1742555852, 4189708143, 2394877945, 397917763,  1622183637,
            3604390888, 2714866558, 953729732,  1340076626, 3518719985, 2797360999, 1068828381, 1219638859, 3624741850, 2936675148, 906185462,  1090812512, 3747672003, 2825379669, 829329135,  1181335161,
            3412177804, 3160834842, 628085408,  1382605366, 3423369109, 3138078467, 570562233,  1426400815, 3317316542, 2998733608, 733239954,  1555261956, 3268935591, 3050360625, 752459403,  1541320221,
            2607071920, 3965973030, 1969922972, 40735498,   2617837225, 3943577151, 1913087877, 83908371,   2512341634, 3803740692, 2075208622, 213261112,  2463272603, 3855990285, 2094854071, 198958881,
            2262029012, 4057260610, 1759359992, 534414190,  2176718541, 4139329115, 1873836001, 414664567,  2282248934, 4279200368, 1711684554, 285281116,  2405801727, 4167216745, 1634467795, 376229701,
            2685067896, 3608007406, 1308918612, 956543938,  2808555105, 3495958263, 1231636301, 1047427035, 2932959818, 3654703836, 1088359270, 936918000,  2847714899, 3736837829, 1202900863, 817233897,
            3183342108, 3401237130, 1404277552, 615818150,  3134207493, 3453421203, 1423857449, 601450431,  3009837614, 3294710456, 1567103746, 711928724,  3020668471, 3272380065, 1510334235, 755167117,
        };
    };
    var crc: u32 = ~@as(u32, 0);
    var i: i32 = undefined;
    {
        i = 0;
        while (i < len) : (i += 1) {
            crc = (crc >> @intCast(@as(u32, 8))) ^ static_local_crc_table.crc_table[@as(u32, @bitCast(@as(i32, @as(i32, buffer[@bitCast(@as(isize, @intCast(i)))])))) ^ (crc & @as(u32, 255))];
        }
    }
    return ~crc;
}
fn stbiw__wpcrc(data: [*c][*c]u8, len: i32) callconv(.c) void {
    const crc: u32 = stbiw__crc32((data.* - @as(usize, @bitCast(@as(isize, @intCast(len))))) - @as(usize, @bitCast(@as(isize, @intCast(@as(i32, 4))))), len + @as(i32, 4));
    data.*[@as(i32, 0)] = @truncate((crc >> @intCast(@as(u32, 24))) & @as(u32, 255));
    data.*[@as(i32, 1)] = @truncate((crc >> @intCast(@as(u32, 16))) & @as(u32, 255));
    data.*[@as(i32, 2)] = @truncate((crc >> @intCast(@as(u32, 8))) & @as(u32, 255));
    data.*[@as(i32, 3)] = @truncate(crc & @as(u32, 255));
    data.* += @as(usize, @bitCast(@as(isize, @intCast(@as(i32, 4)))));
}
fn stbiw__paeth(a: i32, b: i32, c: i32) callconv(.c) u8 {
    const p: i32 = (a + b) - c;
    const pa: i32 = abs(p - a);
    const pb: i32 = abs(p - b);
    const pc: i32 = abs(p - c);
    if ((pa <= pb) and (pa <= pc)) return @bitCast(@as(i8, @truncate(a & @as(i32, 255))));
    if (pb <= pc) return @bitCast(@as(i8, @truncate(b & @as(i32, 255))));
    return @bitCast(@as(i8, @truncate(c & @as(i32, 255))));
}
fn stbiw__encode_png_line(pixels: [*c]u8, stride_bytes: i32, width: i32, height: i32, y: i32, n: i32, filter_type: i32, arg_line_buffer: [*c]i8) callconv(.c) void {
    var line_buffer = arg_line_buffer;
    const static_local_mapping = struct {
        var mapping: [5]i32 = [5]i32{ 0, 1, 2, 3, 4 };
    };
    const static_local_firstmap = struct {
        var firstmap: [5]i32 = [5]i32{ 0, 1, 0, 5, 6 };
    };
    const mymap: [*c]i32 = if (y != @as(i32, 0)) @as([*c]i32, @ptrCast(@alignCast(&static_local_mapping.mapping))) else @as([*c]i32, @ptrCast(@alignCast(&static_local_firstmap.firstmap)));
    var i: i32 = undefined;
    const @"type": i32 = mymap[@bitCast(@as(isize, @intCast(filter_type)))];
    const z: [*c]u8 = pixels + @as(usize, @bitCast(@as(isize, @intCast(stride_bytes * (if (stbi__flip_vertically_on_write != 0) (height - @as(i32, 1)) - y else y)))));
    const signed_stride: i32 = if (stbi__flip_vertically_on_write != 0) -stride_bytes else stride_bytes;
    if (@"type" == @as(i32, 0)) {
        _ = memcpy(@ptrCast(@alignCast(line_buffer)), @ptrCast(@alignCast(z)), @bitCast(@as(i64, width * n)));
        return;
    }
    {
        i = 0;
        while (i < n) : (i += 1) {
            blk: {
                switch (@"type") {
                    @as(i32, 1) => {
                        line_buffer[@bitCast(@as(isize, @intCast(i)))] = @bitCast(@as(u8, @truncate(z[@bitCast(@as(isize, @intCast(i)))])));
                        break :blk;
                    },
                    @as(i32, 2) => {
                        line_buffer[@bitCast(@as(isize, @intCast(i)))] = @truncate(@as(i32, z[@bitCast(@as(isize, @intCast(i)))]) - @as(i32, z[@bitCast(@as(isize, @intCast(i - signed_stride)))]));
                        break :blk;
                    },
                    @as(i32, 3) => {
                        line_buffer[@bitCast(@as(isize, @intCast(i)))] = @truncate(@as(i32, z[@bitCast(@as(isize, @intCast(i)))]) - (@as(i32, z[@bitCast(@as(isize, @intCast(i - signed_stride)))]) >> @intCast(@as(i32, 1))));
                        break :blk;
                    },
                    @as(i32, 4) => {
                        line_buffer[@bitCast(@as(isize, @intCast(i)))] = @truncate(@as(i32, z[@bitCast(@as(isize, @intCast(i)))]) - @as(i32, stbiw__paeth(0, z[@bitCast(@as(isize, @intCast(i - signed_stride)))], 0)));
                        break :blk;
                    },
                    @as(i32, 5) => {
                        line_buffer[@bitCast(@as(isize, @intCast(i)))] = @bitCast(@as(u8, @truncate(z[@bitCast(@as(isize, @intCast(i)))])));
                        break :blk;
                    },
                    @as(i32, 6) => {
                        line_buffer[@bitCast(@as(isize, @intCast(i)))] = @bitCast(@as(u8, @truncate(z[@bitCast(@as(isize, @intCast(i)))])));
                        break :blk;
                    },
                    else => {},
                }
            }
        }
    }
    blk: {
        switch (@"type") {
            @as(i32, 1) => {
                {
                    i = n;
                    while (i < (width * n)) : (i += 1) {
                        line_buffer[@bitCast(@as(isize, @intCast(i)))] = @truncate(@as(i32, z[@bitCast(@as(isize, @intCast(i)))]) - @as(i32, z[@bitCast(@as(isize, @intCast(i - n)))]));
                    }
                }
                break :blk;
            },
            @as(i32, 2) => {
                {
                    i = n;
                    while (i < (width * n)) : (i += 1) {
                        line_buffer[@bitCast(@as(isize, @intCast(i)))] = @truncate(@as(i32, z[@bitCast(@as(isize, @intCast(i)))]) - @as(i32, z[@bitCast(@as(isize, @intCast(i - signed_stride)))]));
                    }
                }
                break :blk;
            },
            @as(i32, 3) => {
                {
                    i = n;
                    while (i < (width * n)) : (i += 1) {
                        line_buffer[@bitCast(@as(isize, @intCast(i)))] = @truncate(@as(i32, z[@bitCast(@as(isize, @intCast(i)))]) - ((@as(i32, z[@bitCast(@as(isize, @intCast(i - n)))]) + @as(i32, z[@bitCast(@as(isize, @intCast(i - signed_stride)))])) >> @intCast(@as(i32, 1))));
                    }
                }
                break :blk;
            },
            @as(i32, 4) => {
                {
                    i = n;
                    while (i < (width * n)) : (i += 1) {
                        line_buffer[@bitCast(@as(isize, @intCast(i)))] = @truncate(@as(i32, z[@bitCast(@as(isize, @intCast(i)))]) - @as(i32, stbiw__paeth(z[@bitCast(@as(isize, @intCast(i - n)))], z[@bitCast(@as(isize, @intCast(i - signed_stride)))], z[@bitCast(@as(isize, @intCast((i - signed_stride) - n)))])));
                    }
                }
                break :blk;
            },
            @as(i32, 5) => {
                {
                    i = n;
                    while (i < (width * n)) : (i += 1) {
                        line_buffer[@bitCast(@as(isize, @intCast(i)))] = @truncate(@as(i32, z[@bitCast(@as(isize, @intCast(i)))]) - (@as(i32, z[@bitCast(@as(isize, @intCast(i - n)))]) >> @intCast(@as(i32, 1))));
                    }
                }
                break :blk;
            },
            @as(i32, 6) => {
                {
                    i = n;
                    while (i < (width * n)) : (i += 1) {
                        line_buffer[@bitCast(@as(isize, @intCast(i)))] = @truncate(@as(i32, z[@bitCast(@as(isize, @intCast(i)))]) - @as(i32, stbiw__paeth(z[@bitCast(@as(isize, @intCast(i - n)))], 0, 0)));
                    }
                }
                break :blk;
            },
            else => {},
        }
    }
}
fn stbi_write_png_to_mem(pixels: [*c]const u8, arg_stride_bytes: i32, x: i32, y: i32, n: i32, out_len: [*c]i32) [*c]u8 {
    var stride_bytes = arg_stride_bytes;
    var force_filter: i32 = stbi_write_force_png_filter;
    const ctype: [5]i32 = [5]i32{
        -@as(i32, 1),
        0,
        4,
        2,
        6,
    };
    var sig: [8]u8 = [8]u8{ 137, 80, 78, 71, 13, 10, 26, 10 };
    var out: [*c]u8 = undefined;
    var o: [*c]u8 = undefined;
    var filt: [*c]u8 = undefined;
    var zlib: [*c]u8 = undefined;
    var line_buffer: [*c]i8 = undefined;
    var j: i32 = undefined;
    var zlen: i32 = undefined;
    if (stride_bytes == @as(i32, 0)) {
        stride_bytes = x * n;
    }
    if (force_filter >= @as(i32, 5)) {
        force_filter = -@as(i32, 1);
    }
    filt = @ptrCast(@alignCast(malloc(@bitCast(@as(i64, ((x * n) + @as(i32, 1)) * y)))));
    if (!(filt != null)) return null;
    line_buffer = @ptrCast(@alignCast(malloc(@bitCast(@as(i64, x * n)))));
    if (!(line_buffer != null)) {
        free(@ptrCast(@alignCast(filt)));
        return null;
    }
    {
        j = 0;
        while (j < y) : (j += 1) {
            var filter_type: i32 = undefined;
            if (force_filter > -@as(i32, 1)) {
                filter_type = force_filter;
                stbiw__encode_png_line(@ptrCast(@alignCast(@constCast(pixels))), stride_bytes, x, y, j, n, force_filter, line_buffer);
            } else {
                var best_filter: i32 = 0;
                var best_filter_val: i32 = 2147483647;
                var est: i32 = undefined;
                var i: i32 = undefined;
                {
                    filter_type = 0;
                    while (filter_type < @as(i32, 5)) : (filter_type += 1) {
                        stbiw__encode_png_line(@ptrCast(@alignCast(@constCast(pixels))), stride_bytes, x, y, j, n, filter_type, line_buffer);
                        est = 0;
                        {
                            i = 0;
                            while (i < (x * n)) : (i += 1) {
                                est += abs(line_buffer[@bitCast(@as(isize, @intCast(i)))]);
                            }
                        }
                        if (est < best_filter_val) {
                            best_filter_val = est;
                            best_filter = filter_type;
                        }
                    }
                }
                if (filter_type != best_filter) {
                    stbiw__encode_png_line(@ptrCast(@alignCast(@constCast(pixels))), stride_bytes, x, y, j, n, best_filter, line_buffer);
                    filter_type = best_filter;
                }
            }
            filt[@bitCast(@as(isize, @intCast(j * ((x * n) + @as(i32, 1)))))] = @bitCast(@as(i8, @truncate(filter_type)));
            _ = memmove(@ptrCast(@alignCast((filt + @as(usize, @bitCast(@as(isize, @intCast(j * ((x * n) + @as(i32, 1))))))) + @as(usize, @bitCast(@as(isize, @intCast(@as(i32, 1))))))), @ptrCast(@alignCast(line_buffer)), @bitCast(@as(i64, x * n)));
        }
    }
    free(@ptrCast(@alignCast(line_buffer)));
    zlib = stbi_zlib_compress(filt, y * ((x * n) + @as(i32, 1)), &zlen, stbi_write_png_compression_level);
    free(@ptrCast(@alignCast(filt)));
    if (!(zlib != null)) return null;
    defer free(@ptrCast(zlib));
    const total = std.math.add(i32, zlen, 57) catch return null;
    out = @ptrCast(malloc(@intCast(total)) orelse return null);
    out_len.* = total;
    o = out;
    _ = memmove(@ptrCast(@alignCast(o)), @ptrCast(@alignCast(@as([*c]u8, @ptrCast(@alignCast(&sig))))), 8);
    o += @as(usize, @bitCast(@as(isize, @intCast(@as(i32, 8)))));
    o[@as(i32, 0)] = @bitCast(@as(i8, @truncate((@as(i32, 13) >> @intCast(@as(i32, 24))) & @as(i32, 255))));
    o[@as(i32, 1)] = @bitCast(@as(i8, @truncate((@as(i32, 13) >> @intCast(@as(i32, 16))) & @as(i32, 255))));
    o[@as(i32, 2)] = @bitCast(@as(i8, @truncate((@as(i32, 13) >> @intCast(@as(i32, 8))) & @as(i32, 255))));
    o[@as(i32, 3)] = @bitCast(@as(i8, @truncate(@as(i32, 13) & @as(i32, 255))));
    o += @as(usize, @bitCast(@as(isize, @intCast(@as(i32, 4)))));
    o[@as(i32, 0)] = @bitCast(@as(i8, @truncate(@as(i32, "IHDR"[@as(i32, 0)]) & @as(i32, 255))));
    o[@as(i32, 1)] = @bitCast(@as(i8, @truncate(@as(i32, "IHDR"[@as(i32, 1)]) & @as(i32, 255))));
    o[@as(i32, 2)] = @bitCast(@as(i8, @truncate(@as(i32, "IHDR"[@as(i32, 2)]) & @as(i32, 255))));
    o[@as(i32, 3)] = @bitCast(@as(i8, @truncate(@as(i32, "IHDR"[@as(i32, 3)]) & @as(i32, 255))));
    o += @as(usize, @bitCast(@as(isize, @intCast(@as(i32, 4)))));
    o[@as(i32, 0)] = @bitCast(@as(i8, @truncate((x >> @intCast(@as(i32, 24))) & @as(i32, 255))));
    o[@as(i32, 1)] = @bitCast(@as(i8, @truncate((x >> @intCast(@as(i32, 16))) & @as(i32, 255))));
    o[@as(i32, 2)] = @bitCast(@as(i8, @truncate((x >> @intCast(@as(i32, 8))) & @as(i32, 255))));
    o[@as(i32, 3)] = @bitCast(@as(i8, @truncate(x & @as(i32, 255))));
    o += @as(usize, @bitCast(@as(isize, @intCast(@as(i32, 4)))));
    o[@as(i32, 0)] = @bitCast(@as(i8, @truncate((y >> @intCast(@as(i32, 24))) & @as(i32, 255))));
    o[@as(i32, 1)] = @bitCast(@as(i8, @truncate((y >> @intCast(@as(i32, 16))) & @as(i32, 255))));
    o[@as(i32, 2)] = @bitCast(@as(i8, @truncate((y >> @intCast(@as(i32, 8))) & @as(i32, 255))));
    o[@as(i32, 3)] = @bitCast(@as(i8, @truncate(y & @as(i32, 255))));
    o += @as(usize, @bitCast(@as(isize, @intCast(@as(i32, 4)))));
    (blk: {
        const ref = &o;
        const tmp = ref.*;
        ref.* += 1;
        break :blk tmp;
    }).* = 8;
    (blk: {
        const ref = &o;
        const tmp = ref.*;
        ref.* += 1;
        break :blk tmp;
    }).* = @bitCast(@as(i8, @truncate(ctype[@bitCast(@as(isize, @intCast(n)))] & @as(i32, 255))));
    (blk: {
        const ref = &o;
        const tmp = ref.*;
        ref.* += 1;
        break :blk tmp;
    }).* = 0;
    (blk: {
        const ref = &o;
        const tmp = ref.*;
        ref.* += 1;
        break :blk tmp;
    }).* = 0;
    (blk: {
        const ref = &o;
        const tmp = ref.*;
        ref.* += 1;
        break :blk tmp;
    }).* = 0;
    stbiw__wpcrc(&o, 13);
    o[@as(i32, 0)] = @bitCast(@as(i8, @truncate((zlen >> @intCast(@as(i32, 24))) & @as(i32, 255))));
    o[@as(i32, 1)] = @bitCast(@as(i8, @truncate((zlen >> @intCast(@as(i32, 16))) & @as(i32, 255))));
    o[@as(i32, 2)] = @bitCast(@as(i8, @truncate((zlen >> @intCast(@as(i32, 8))) & @as(i32, 255))));
    o[@as(i32, 3)] = @bitCast(@as(i8, @truncate(zlen & @as(i32, 255))));
    o += @as(usize, @bitCast(@as(isize, @intCast(@as(i32, 4)))));
    o[@as(i32, 0)] = @bitCast(@as(i8, @truncate(@as(i32, "IDAT"[@as(i32, 0)]) & @as(i32, 255))));
    o[@as(i32, 1)] = @bitCast(@as(i8, @truncate(@as(i32, "IDAT"[@as(i32, 1)]) & @as(i32, 255))));
    o[@as(i32, 2)] = @bitCast(@as(i8, @truncate(@as(i32, "IDAT"[@as(i32, 2)]) & @as(i32, 255))));
    o[@as(i32, 3)] = @bitCast(@as(i8, @truncate(@as(i32, "IDAT"[@as(i32, 3)]) & @as(i32, 255))));
    o += @as(usize, @bitCast(@as(isize, @intCast(@as(i32, 4)))));
    _ = memmove(@ptrCast(@alignCast(o)), @ptrCast(@alignCast(zlib)), @bitCast(@as(i64, zlen)));
    o += @as(usize, @bitCast(@as(isize, @intCast(zlen))));
    stbiw__wpcrc(&o, zlen);
    o[@as(i32, 0)] = @bitCast(@as(i8, @truncate((@as(i32, 0) >> @intCast(@as(i32, 24))) & @as(i32, 255))));
    o[@as(i32, 1)] = @bitCast(@as(i8, @truncate((@as(i32, 0) >> @intCast(@as(i32, 16))) & @as(i32, 255))));
    o[@as(i32, 2)] = @bitCast(@as(i8, @truncate((@as(i32, 0) >> @intCast(@as(i32, 8))) & @as(i32, 255))));
    o[@as(i32, 3)] = @bitCast(@as(i8, @truncate(@as(i32, 0) & @as(i32, 255))));
    o += @as(usize, @bitCast(@as(isize, @intCast(@as(i32, 4)))));
    o[@as(i32, 0)] = @bitCast(@as(i8, @truncate(@as(i32, "IEND"[@as(i32, 0)]) & @as(i32, 255))));
    o[@as(i32, 1)] = @bitCast(@as(i8, @truncate(@as(i32, "IEND"[@as(i32, 1)]) & @as(i32, 255))));
    o[@as(i32, 2)] = @bitCast(@as(i8, @truncate(@as(i32, "IEND"[@as(i32, 2)]) & @as(i32, 255))));
    o[@as(i32, 3)] = @bitCast(@as(i8, @truncate(@as(i32, "IEND"[@as(i32, 3)]) & @as(i32, 255))));
    o += @as(usize, @bitCast(@as(isize, @intCast(@as(i32, 4)))));
    stbiw__wpcrc(&o, 0);
    _ = !!(o == (out + @as(usize, @bitCast(@as(isize, @intCast(out_len.*)))))) or ((blk: {
        _assert("o == out + *out_len", "tools\\dataset\\vendor\\stb_image_write.h", 1209);
        break :blk 0;
    }) != 0);
    return out;
}
pub fn stbi_write_png_to_func(func: ?*const stbi_write_func, context: ?*anyopaque, x: i32, y: i32, comp: i32, data: ?*const anyopaque, stride_bytes: i32) i32 {
    var len: i32 = undefined;
    const png: [*c]u8 = stbi_write_png_to_mem(@ptrCast(@alignCast(data)), stride_bytes, x, y, comp, &len);
    if (@as(?*anyopaque, @ptrCast(@alignCast(png))) == @as(?*anyopaque, null)) return 0;
    func.?(context, @ptrCast(@alignCast(png)), len);
    free(@ptrCast(@alignCast(png)));
    return 1;
}
const stbiw__jpg_ZigZag: [64]u8 = [64]u8{
    0,  1,  5,  6,  14, 15, 27, 28, 2,  4,  7,  13, 16, 26, 29, 42,
    3,  8,  12, 17, 25, 30, 41, 43, 9,  11, 18, 24, 31, 40, 44, 53,
    10, 19, 23, 32, 39, 45, 52, 54, 20, 22, 33, 38, 46, 51, 55, 60,
    21, 34, 37, 47, 50, 56, 59, 61, 35, 36, 48, 49, 57, 58, 62, 63,
};
fn stbiw__jpg_writeBits(s: [*c]stbi__write_context, bitBufP: [*c]i32, bitCntP: [*c]i32, bs: [*c]const u16) callconv(.c) void {
    var bitBuf: i32 = bitBufP.*;
    var bitCnt: i32 = bitCntP.*;
    bitCnt += bs[@as(i32, 1)];
    bitBuf |= @as(i32, bs[@as(i32, 0)]) << @intCast(@as(i32, 24) - bitCnt);
    while (bitCnt >= @as(i32, 8)) {
        const c: u8 = @bitCast(@as(i8, @truncate((bitBuf >> @intCast(@as(i32, 16))) & @as(i32, 255))));
        stbiw__putc(s, c);
        if (@as(i32, c) == @as(i32, 255)) {
            stbiw__putc(s, 0);
        }
        bitBuf <<= @intCast(@as(i32, 8));
        bitCnt -= 8;
    }
    bitBufP.* = bitBuf;
    bitCntP.* = bitCnt;
}
fn stbiw__jpg_DCT(d0p: [*c]f32, d1p: [*c]f32, d2p: [*c]f32, d3p: [*c]f32, d4p: [*c]f32, d5p: [*c]f32, d6p: [*c]f32, d7p: [*c]f32) callconv(.c) void {
    var d0: f32 = d0p.*;
    const d1: f32 = d1p.*;
    var d2: f32 = d2p.*;
    const d3: f32 = d3p.*;
    var d4: f32 = d4p.*;
    const d5: f32 = d5p.*;
    var d6: f32 = d6p.*;
    const d7: f32 = d7p.*;
    var z1: f32 = undefined;
    var z2: f32 = undefined;
    var z3: f32 = undefined;
    var z4: f32 = undefined;
    var z5: f32 = undefined;
    var z11: f32 = undefined;
    var z13: f32 = undefined;
    const tmp0: f32 = d0 + d7;
    const tmp7: f32 = d0 - d7;
    const tmp1: f32 = d1 + d6;
    const tmp6: f32 = d1 - d6;
    const tmp2: f32 = d2 + d5;
    const tmp5: f32 = d2 - d5;
    const tmp3: f32 = d3 + d4;
    const tmp4: f32 = d3 - d4;
    var tmp10: f32 = tmp0 + tmp3;
    const tmp13: f32 = tmp0 - tmp3;
    var tmp11: f32 = tmp1 + tmp2;
    var tmp12: f32 = tmp1 - tmp2;
    d0 = tmp10 + tmp11;
    d4 = tmp10 - tmp11;
    z1 = (tmp12 + tmp13) * @as(f32, 0.707107);
    d2 = tmp13 + z1;
    d6 = tmp13 - z1;
    tmp10 = tmp4 + tmp5;
    tmp11 = tmp5 + tmp6;
    tmp12 = tmp6 + tmp7;
    z5 = (tmp10 - tmp12) * @as(f32, 0.382683);
    z2 = (tmp10 * @as(f32, 0.541196)) + z5;
    z4 = (tmp12 * @as(f32, 1.306563)) + z5;
    z3 = tmp11 * @as(f32, 0.707107);
    z11 = tmp7 + z3;
    z13 = tmp7 - z3;
    d5p.* = z13 + z2;
    d3p.* = z13 - z2;
    d1p.* = z11 + z4;
    d7p.* = z11 - z4;
    d0p.* = d0;
    d2p.* = d2;
    d4p.* = d4;
    d6p.* = d6;
}
fn stbiw__jpg_calcBits(arg_val: i32, arg_bits: [*c]u16) callconv(.c) void {
    var val = arg_val;
    var bits = arg_bits;
    var tmp1: i32 = if (val < @as(i32, 0)) -val else val;
    val = if (val < @as(i32, 0)) val - @as(i32, 1) else val;
    bits[@as(i32, 1)] = 1;
    while ((blk: {
        const ref = &tmp1;
        ref.* >>= @intCast(@as(i32, 1));
        break :blk ref.*;
    }) != 0) {
        bits[@as(i32, 1)] +%= 1;
    }
    bits[@as(i32, 0)] = @bitCast(@as(i16, @truncate(val & ((@as(i32, 1) << @intCast(@as(i32, bits[@as(i32, 1)]))) - @as(i32, 1)))));
}
fn stbiw__jpg_processDU(s: [*c]stbi__write_context, bitBuf: [*c]i32, bitCnt: [*c]i32, arg_CDU: [*c]f32, du_stride: i32, fdtbl: [*c]f32, DC: i32, arg_HTDC: [*c][2]u16, arg_HTAC: [*c][2]u16) callconv(.c) i32 {
    var CDU = arg_CDU;
    var HTDC = arg_HTDC;
    var HTAC = arg_HTAC;
    const EOB: [2]u16 = [2]u16{
        HTAC[@as(i32, 0)][@as(i32, 0)],
        HTAC[@as(i32, 0)][@as(i32, 1)],
    };
    const M16zeroes: [2]u16 = [2]u16{
        HTAC[@as(i32, 240)][@as(i32, 0)],
        HTAC[@as(i32, 240)][@as(i32, 1)],
    };
    var dataOff: i32 = undefined;
    var i: i32 = undefined;
    var j: i32 = undefined;
    var n: i32 = undefined;
    var diff: i32 = undefined;
    var end0pos: i32 = undefined;
    var x: i32 = undefined;
    var y: i32 = undefined;
    var DU: [64]i32 = undefined;
    {
        dataOff = 0;
        n = du_stride * @as(i32, 8);
        while (dataOff < n) : (dataOff += du_stride) {
            stbiw__jpg_DCT(&CDU[@bitCast(@as(isize, @intCast(dataOff)))], &CDU[@bitCast(@as(isize, @intCast(dataOff + @as(i32, 1))))], &CDU[@bitCast(@as(isize, @intCast(dataOff + @as(i32, 2))))], &CDU[@bitCast(@as(isize, @intCast(dataOff + @as(i32, 3))))], &CDU[@bitCast(@as(isize, @intCast(dataOff + @as(i32, 4))))], &CDU[@bitCast(@as(isize, @intCast(dataOff + @as(i32, 5))))], &CDU[@bitCast(@as(isize, @intCast(dataOff + @as(i32, 6))))], &CDU[@bitCast(@as(isize, @intCast(dataOff + @as(i32, 7))))]);
        }
    }
    {
        dataOff = 0;
        while (dataOff < @as(i32, 8)) : (dataOff += 1) {
            stbiw__jpg_DCT(&CDU[@bitCast(@as(isize, @intCast(dataOff)))], &CDU[@bitCast(@as(isize, @intCast(dataOff + du_stride)))], &CDU[@bitCast(@as(isize, @intCast(dataOff + (du_stride * @as(i32, 2)))))], &CDU[@bitCast(@as(isize, @intCast(dataOff + (du_stride * @as(i32, 3)))))], &CDU[@bitCast(@as(isize, @intCast(dataOff + (du_stride * @as(i32, 4)))))], &CDU[@bitCast(@as(isize, @intCast(dataOff + (du_stride * @as(i32, 5)))))], &CDU[@bitCast(@as(isize, @intCast(dataOff + (du_stride * @as(i32, 6)))))], &CDU[@bitCast(@as(isize, @intCast(dataOff + (du_stride * @as(i32, 7)))))]);
        }
    }
    {
        y = 0;
        j = 0;
        while (y < @as(i32, 8)) : (y += 1) {
            {
                x = 0;
                while (x < @as(i32, 8)) : ({
                    x += 1;
                    j += 1;
                }) {
                    var v: f32 = undefined;
                    i = (y * du_stride) + x;
                    v = CDU[@bitCast(@as(isize, @intCast(i)))] * fdtbl[@bitCast(@as(isize, @intCast(j)))];
                    DU[stbiw__jpg_ZigZag[@bitCast(@as(isize, @intCast(j)))]] = @intFromFloat(if (v < @as(f32, @floatFromInt(@as(i32, 0)))) v - @as(f32, 0.5) else v + @as(f32, 0.5));
                }
            }
        }
    }
    diff = DU[@as(i32, 0)] - DC;
    if (diff == @as(i32, 0)) {
        stbiw__jpg_writeBits(s, bitBuf, bitCnt, @ptrCast(@alignCast(&HTDC[@as(i32, 0)])));
    } else {
        var bits: [2]u16 = undefined;
        stbiw__jpg_calcBits(diff, @ptrCast(@alignCast(&bits)));
        stbiw__jpg_writeBits(s, bitBuf, bitCnt, @ptrCast(@alignCast(&HTDC[bits[@as(i32, 1)]])));
        stbiw__jpg_writeBits(s, bitBuf, bitCnt, @ptrCast(@alignCast(&bits)));
    }
    end0pos = 63;
    while ((end0pos > @as(i32, 0)) and (DU[@bitCast(@as(isize, @intCast(end0pos)))] == @as(i32, 0))) : (end0pos -= 1) {}
    if (end0pos == @as(i32, 0)) {
        stbiw__jpg_writeBits(s, bitBuf, bitCnt, @ptrCast(@alignCast(&EOB)));
        return DU[@as(i32, 0)];
    }
    {
        i = 1;
        while (i <= end0pos) : (i += 1) {
            const startpos: i32 = i;
            var nrzeroes: i32 = undefined;
            var bits: [2]u16 = undefined;
            while ((DU[@bitCast(@as(isize, @intCast(i)))] == @as(i32, 0)) and (i <= end0pos)) : (i += 1) {}
            nrzeroes = i - startpos;
            if (nrzeroes >= @as(i32, 16)) {
                const lng: i32 = nrzeroes >> @intCast(@as(i32, 4));
                var nrmarker: i32 = undefined;
                {
                    nrmarker = 1;
                    while (nrmarker <= lng) : (nrmarker += 1) {
                        stbiw__jpg_writeBits(s, bitBuf, bitCnt, @ptrCast(@alignCast(&M16zeroes)));
                    }
                }
                nrzeroes &= 15;
            }
            stbiw__jpg_calcBits(DU[@bitCast(@as(isize, @intCast(i)))], @ptrCast(@alignCast(&bits)));
            stbiw__jpg_writeBits(s, bitBuf, bitCnt, @ptrCast(@alignCast(&HTAC[@bitCast(@as(isize, @intCast((nrzeroes << @intCast(@as(i32, 4))) + @as(i32, bits[@as(i32, 1)]))))])));
            stbiw__jpg_writeBits(s, bitBuf, bitCnt, @ptrCast(@alignCast(&bits)));
        }
    }
    if (end0pos != @as(i32, 63)) {
        stbiw__jpg_writeBits(s, bitBuf, bitCnt, @ptrCast(@alignCast(&EOB)));
    }
    return DU[@as(i32, 0)];
}
fn stbi_write_jpg_core(s: [*c]stbi__write_context, width: i32, height: i32, comp: i32, data: ?*const anyopaque, arg_quality: i32) callconv(.c) i32 {
    var quality = arg_quality;
    const static_local_std_dc_luminance_nrcodes = struct {
        const std_dc_luminance_nrcodes: [17]u8 = [17]u8{
            0, 0, 1, 5, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0,
            0,
        };
    };
    const static_local_std_dc_luminance_values = struct {
        const std_dc_luminance_values: [12]u8 = [12]u8{
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
        };
    };
    const static_local_std_ac_luminance_nrcodes = struct {
        const std_ac_luminance_nrcodes: [17]u8 = [17]u8{
            0,   0, 2, 1, 3, 3, 2, 4, 3, 5, 5, 4, 4, 0, 0, 1,
            125,
        };
    };
    const static_local_std_ac_luminance_values = struct {
        const std_ac_luminance_values: [162]u8 = [162]u8{
            1,   2,   3,   0,   4,   17,  5,   18,  33,  49,  65,  6,   19,  81,  97,  7,
            34,  113, 20,  50,  129, 145, 161, 8,   35,  66,  177, 193, 21,  82,  209, 240,
            36,  51,  98,  114, 130, 9,   10,  22,  23,  24,  25,  26,  37,  38,  39,  40,
            41,  42,  52,  53,  54,  55,  56,  57,  58,  67,  68,  69,  70,  71,  72,  73,
            74,  83,  84,  85,  86,  87,  88,  89,  90,  99,  100, 101, 102, 103, 104, 105,
            106, 115, 116, 117, 118, 119, 120, 121, 122, 131, 132, 133, 134, 135, 136, 137,
            138, 146, 147, 148, 149, 150, 151, 152, 153, 154, 162, 163, 164, 165, 166, 167,
            168, 169, 170, 178, 179, 180, 181, 182, 183, 184, 185, 186, 194, 195, 196, 197,
            198, 199, 200, 201, 202, 210, 211, 212, 213, 214, 215, 216, 217, 218, 225, 226,
            227, 228, 229, 230, 231, 232, 233, 234, 241, 242, 243, 244, 245, 246, 247, 248,
            249, 250,
        };
    };
    const static_local_std_dc_chrominance_nrcodes = struct {
        const std_dc_chrominance_nrcodes: [17]u8 = [17]u8{
            0, 0, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0,
            0,
        };
    };
    const static_local_std_dc_chrominance_values = struct {
        const std_dc_chrominance_values: [12]u8 = [12]u8{
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
        };
    };
    const static_local_std_ac_chrominance_nrcodes = struct {
        const std_ac_chrominance_nrcodes: [17]u8 = [17]u8{
            0,   0, 2, 1, 2, 4, 4, 3, 4, 7, 5, 4, 4, 0, 1, 2,
            119,
        };
    };
    const static_local_std_ac_chrominance_values = struct {
        const std_ac_chrominance_values: [162]u8 = [162]u8{
            0,   1,   2,   3,   17,  4,   5,   33,  49,  6,   18,  65,  81,  7,   97,  113,
            19,  34,  50,  129, 8,   20,  66,  145, 161, 177, 193, 9,   35,  51,  82,  240,
            21,  98,  114, 209, 10,  22,  36,  52,  225, 37,  241, 23,  24,  25,  26,  38,
            39,  40,  41,  42,  53,  54,  55,  56,  57,  58,  67,  68,  69,  70,  71,  72,
            73,  74,  83,  84,  85,  86,  87,  88,  89,  90,  99,  100, 101, 102, 103, 104,
            105, 106, 115, 116, 117, 118, 119, 120, 121, 122, 130, 131, 132, 133, 134, 135,
            136, 137, 138, 146, 147, 148, 149, 150, 151, 152, 153, 154, 162, 163, 164, 165,
            166, 167, 168, 169, 170, 178, 179, 180, 181, 182, 183, 184, 185, 186, 194, 195,
            196, 197, 198, 199, 200, 201, 202, 210, 211, 212, 213, 214, 215, 216, 217, 218,
            226, 227, 228, 229, 230, 231, 232, 233, 234, 242, 243, 244, 245, 246, 247, 248,
            249, 250,
        };
    };
    const static_local_YDC_HT = struct {
        var YDC_HT: [256][2]u16 = [12][2]u16{
            [2]u16{ 0, 2 },
            [2]u16{ 2, 3 },
            [2]u16{ 3, 3 },
            [2]u16{ 4, 3 },
            [2]u16{ 5, 3 },
            [2]u16{ 6, 3 },
            [2]u16{ 14, 4 },
            [2]u16{ 30, 5 },
            [2]u16{ 62, 6 },
            [2]u16{ 126, 7 },
            [2]u16{ 254, 8 },
            [2]u16{ 510, 9 },
        } ++ @as([244][2]u16, @splat(std.mem.zeroes([2]u16)));
    };
    const static_local_UVDC_HT = struct {
        var UVDC_HT: [256][2]u16 = [12][2]u16{
            [2]u16{ 0, 2 },
            [2]u16{ 1, 2 },
            [2]u16{ 2, 2 },
            [2]u16{ 6, 3 },
            [2]u16{ 14, 4 },
            [2]u16{ 30, 5 },
            [2]u16{ 62, 6 },
            [2]u16{ 126, 7 },
            [2]u16{ 254, 8 },
            [2]u16{ 510, 9 },
            [2]u16{ 1022, 10 },
            [2]u16{ 2046, 11 },
        } ++ @as([244][2]u16, @splat(std.mem.zeroes([2]u16)));
    };
    const static_local_YAC_HT = struct {
        var YAC_HT: [256][2]u16 = [256][2]u16{
            [2]u16{ 10, 4 },
            [2]u16{ 0, 2 },
            [2]u16{ 1, 2 },
            [2]u16{ 4, 3 },
            [2]u16{ 11, 4 },
            [2]u16{ 26, 5 },
            [2]u16{ 120, 7 },
            [2]u16{ 248, 8 },
            [2]u16{ 1014, 10 },
            [2]u16{ 65410, 16 },
            [2]u16{ 65411, 16 },
            [2]u16{ 0, 0 },
            [2]u16{ 0, 0 },
            [2]u16{ 0, 0 },
            [2]u16{ 0, 0 },
            [2]u16{ 0, 0 },
            [2]u16{ 0, 0 },
            [2]u16{ 12, 4 },
            [2]u16{ 27, 5 },
            [2]u16{ 121, 7 },
            [2]u16{ 502, 9 },
            [2]u16{ 2038, 11 },
            [2]u16{ 65412, 16 },
            [2]u16{ 65413, 16 },
            [2]u16{ 65414, 16 },
            [2]u16{ 65415, 16 },
            [2]u16{ 65416, 16 },
            [2]u16{ 0, 0 },
            [2]u16{ 0, 0 },
            [2]u16{ 0, 0 },
            [2]u16{ 0, 0 },
            [2]u16{ 0, 0 },
            [2]u16{ 0, 0 },
            [2]u16{ 28, 5 },
            [2]u16{ 249, 8 },
            [2]u16{ 1015, 10 },
            [2]u16{ 4084, 12 },
            [2]u16{ 65417, 16 },
            [2]u16{ 65418, 16 },
            [2]u16{ 65419, 16 },
            [2]u16{ 65420, 16 },
            [2]u16{ 65421, 16 },
            [2]u16{ 65422, 16 },
            [2]u16{ 0, 0 },
            [2]u16{ 0, 0 },
            [2]u16{ 0, 0 },
            [2]u16{ 0, 0 },
            [2]u16{ 0, 0 },
            [2]u16{ 0, 0 },
            [2]u16{ 58, 6 },
            [2]u16{ 503, 9 },
            [2]u16{ 4085, 12 },
            [2]u16{ 65423, 16 },
            [2]u16{ 65424, 16 },
            [2]u16{ 65425, 16 },
            [2]u16{ 65426, 16 },
            [2]u16{ 65427, 16 },
            [2]u16{ 65428, 16 },
            [2]u16{ 65429, 16 },
            [2]u16{ 0, 0 },
            [2]u16{ 0, 0 },
            [2]u16{ 0, 0 },
            [2]u16{ 0, 0 },
            [2]u16{ 0, 0 },
            [2]u16{ 0, 0 },
            [2]u16{ 59, 6 },
            [2]u16{ 1016, 10 },
            [2]u16{ 65430, 16 },
            [2]u16{ 65431, 16 },
            [2]u16{ 65432, 16 },
            [2]u16{ 65433, 16 },
            [2]u16{ 65434, 16 },
            [2]u16{ 65435, 16 },
            [2]u16{ 65436, 16 },
            [2]u16{ 65437, 16 },
            [2]u16{ 0, 0 },
            [2]u16{ 0, 0 },
            [2]u16{ 0, 0 },
            [2]u16{ 0, 0 },
            [2]u16{ 0, 0 },
            [2]u16{ 0, 0 },
            [2]u16{ 122, 7 },
            [2]u16{ 2039, 11 },
            [2]u16{ 65438, 16 },
            [2]u16{ 65439, 16 },
            [2]u16{ 65440, 16 },
            [2]u16{ 65441, 16 },
            [2]u16{ 65442, 16 },
            [2]u16{ 65443, 16 },
            [2]u16{ 65444, 16 },
            [2]u16{ 65445, 16 },
            [2]u16{ 0, 0 },
            [2]u16{ 0, 0 },
            [2]u16{ 0, 0 },
            [2]u16{ 0, 0 },
            [2]u16{ 0, 0 },
            [2]u16{ 0, 0 },
            [2]u16{ 123, 7 },
            [2]u16{ 4086, 12 },
            [2]u16{ 65446, 16 },
            [2]u16{ 65447, 16 },
            [2]u16{ 65448, 16 },
            [2]u16{ 65449, 16 },
            [2]u16{ 65450, 16 },
            [2]u16{ 65451, 16 },
            [2]u16{ 65452, 16 },
            [2]u16{ 65453, 16 },
            [2]u16{ 0, 0 },
            [2]u16{ 0, 0 },
            [2]u16{ 0, 0 },
            [2]u16{ 0, 0 },
            [2]u16{ 0, 0 },
            [2]u16{ 0, 0 },
            [2]u16{ 250, 8 },
            [2]u16{ 4087, 12 },
            [2]u16{ 65454, 16 },
            [2]u16{ 65455, 16 },
            [2]u16{ 65456, 16 },
            [2]u16{ 65457, 16 },
            [2]u16{ 65458, 16 },
            [2]u16{ 65459, 16 },
            [2]u16{ 65460, 16 },
            [2]u16{ 65461, 16 },
            [2]u16{ 0, 0 },
            [2]u16{ 0, 0 },
            [2]u16{ 0, 0 },
            [2]u16{ 0, 0 },
            [2]u16{ 0, 0 },
            [2]u16{ 0, 0 },
            [2]u16{ 504, 9 },
            [2]u16{ 32704, 15 },
            [2]u16{ 65462, 16 },
            [2]u16{ 65463, 16 },
            [2]u16{ 65464, 16 },
            [2]u16{ 65465, 16 },
            [2]u16{ 65466, 16 },
            [2]u16{ 65467, 16 },
            [2]u16{ 65468, 16 },
            [2]u16{ 65469, 16 },
            [2]u16{ 0, 0 },
            [2]u16{ 0, 0 },
            [2]u16{ 0, 0 },
            [2]u16{ 0, 0 },
            [2]u16{ 0, 0 },
            [2]u16{ 0, 0 },
            [2]u16{ 505, 9 },
            [2]u16{ 65470, 16 },
            [2]u16{ 65471, 16 },
            [2]u16{ 65472, 16 },
            [2]u16{ 65473, 16 },
            [2]u16{ 65474, 16 },
            [2]u16{ 65475, 16 },
            [2]u16{ 65476, 16 },
            [2]u16{ 65477, 16 },
            [2]u16{ 65478, 16 },
            [2]u16{ 0, 0 },
            [2]u16{ 0, 0 },
            [2]u16{ 0, 0 },
            [2]u16{ 0, 0 },
            [2]u16{ 0, 0 },
            [2]u16{ 0, 0 },
            [2]u16{ 506, 9 },
            [2]u16{ 65479, 16 },
            [2]u16{ 65480, 16 },
            [2]u16{ 65481, 16 },
            [2]u16{ 65482, 16 },
            [2]u16{ 65483, 16 },
            [2]u16{ 65484, 16 },
            [2]u16{ 65485, 16 },
            [2]u16{ 65486, 16 },
            [2]u16{ 65487, 16 },
            [2]u16{ 0, 0 },
            [2]u16{ 0, 0 },
            [2]u16{ 0, 0 },
            [2]u16{ 0, 0 },
            [2]u16{ 0, 0 },
            [2]u16{ 0, 0 },
            [2]u16{ 1017, 10 },
            [2]u16{ 65488, 16 },
            [2]u16{ 65489, 16 },
            [2]u16{ 65490, 16 },
            [2]u16{ 65491, 16 },
            [2]u16{ 65492, 16 },
            [2]u16{ 65493, 16 },
            [2]u16{ 65494, 16 },
            [2]u16{ 65495, 16 },
            [2]u16{ 65496, 16 },
            [2]u16{ 0, 0 },
            [2]u16{ 0, 0 },
            [2]u16{ 0, 0 },
            [2]u16{ 0, 0 },
            [2]u16{ 0, 0 },
            [2]u16{ 0, 0 },
            [2]u16{ 1018, 10 },
            [2]u16{ 65497, 16 },
            [2]u16{ 65498, 16 },
            [2]u16{ 65499, 16 },
            [2]u16{ 65500, 16 },
            [2]u16{ 65501, 16 },
            [2]u16{ 65502, 16 },
            [2]u16{ 65503, 16 },
            [2]u16{ 65504, 16 },
            [2]u16{ 65505, 16 },
            [2]u16{ 0, 0 },
            [2]u16{ 0, 0 },
            [2]u16{ 0, 0 },
            [2]u16{ 0, 0 },
            [2]u16{ 0, 0 },
            [2]u16{ 0, 0 },
            [2]u16{ 2040, 11 },
            [2]u16{ 65506, 16 },
            [2]u16{ 65507, 16 },
            [2]u16{ 65508, 16 },
            [2]u16{ 65509, 16 },
            [2]u16{ 65510, 16 },
            [2]u16{ 65511, 16 },
            [2]u16{ 65512, 16 },
            [2]u16{ 65513, 16 },
            [2]u16{ 65514, 16 },
            [2]u16{ 0, 0 },
            [2]u16{ 0, 0 },
            [2]u16{ 0, 0 },
            [2]u16{ 0, 0 },
            [2]u16{ 0, 0 },
            [2]u16{ 0, 0 },
            [2]u16{ 65515, 16 },
            [2]u16{ 65516, 16 },
            [2]u16{ 65517, 16 },
            [2]u16{ 65518, 16 },
            [2]u16{ 65519, 16 },
            [2]u16{ 65520, 16 },
            [2]u16{ 65521, 16 },
            [2]u16{ 65522, 16 },
            [2]u16{ 65523, 16 },
            [2]u16{ 65524, 16 },
            [2]u16{ 0, 0 },
            [2]u16{ 0, 0 },
            [2]u16{ 0, 0 },
            [2]u16{ 0, 0 },
            [2]u16{ 0, 0 },
            [2]u16{ 2041, 11 },
            [2]u16{ 65525, 16 },
            [2]u16{ 65526, 16 },
            [2]u16{ 65527, 16 },
            [2]u16{ 65528, 16 },
            [2]u16{ 65529, 16 },
            [2]u16{ 65530, 16 },
            [2]u16{ 65531, 16 },
            [2]u16{ 65532, 16 },
            [2]u16{ 65533, 16 },
            [2]u16{ 65534, 16 },
            [2]u16{ 0, 0 },
            [2]u16{ 0, 0 },
            [2]u16{ 0, 0 },
            [2]u16{ 0, 0 },
            [2]u16{ 0, 0 },
        };
    };
    const static_local_UVAC_HT = struct {
        var UVAC_HT: [256][2]u16 = [256][2]u16{
            [2]u16{ 0, 2 },
            [2]u16{ 1, 2 },
            [2]u16{ 4, 3 },
            [2]u16{ 10, 4 },
            [2]u16{ 24, 5 },
            [2]u16{ 25, 5 },
            [2]u16{ 56, 6 },
            [2]u16{ 120, 7 },
            [2]u16{ 500, 9 },
            [2]u16{ 1014, 10 },
            [2]u16{ 4084, 12 },
            [2]u16{ 0, 0 },
            [2]u16{ 0, 0 },
            [2]u16{ 0, 0 },
            [2]u16{ 0, 0 },
            [2]u16{ 0, 0 },
            [2]u16{ 0, 0 },
            [2]u16{ 11, 4 },
            [2]u16{ 57, 6 },
            [2]u16{ 246, 8 },
            [2]u16{ 501, 9 },
            [2]u16{ 2038, 11 },
            [2]u16{ 4085, 12 },
            [2]u16{ 65416, 16 },
            [2]u16{ 65417, 16 },
            [2]u16{ 65418, 16 },
            [2]u16{ 65419, 16 },
            [2]u16{ 0, 0 },
            [2]u16{ 0, 0 },
            [2]u16{ 0, 0 },
            [2]u16{ 0, 0 },
            [2]u16{ 0, 0 },
            [2]u16{ 0, 0 },
            [2]u16{ 26, 5 },
            [2]u16{ 247, 8 },
            [2]u16{ 1015, 10 },
            [2]u16{ 4086, 12 },
            [2]u16{ 32706, 15 },
            [2]u16{ 65420, 16 },
            [2]u16{ 65421, 16 },
            [2]u16{ 65422, 16 },
            [2]u16{ 65423, 16 },
            [2]u16{ 65424, 16 },
            [2]u16{ 0, 0 },
            [2]u16{ 0, 0 },
            [2]u16{ 0, 0 },
            [2]u16{ 0, 0 },
            [2]u16{ 0, 0 },
            [2]u16{ 0, 0 },
            [2]u16{ 27, 5 },
            [2]u16{ 248, 8 },
            [2]u16{ 1016, 10 },
            [2]u16{ 4087, 12 },
            [2]u16{ 65425, 16 },
            [2]u16{ 65426, 16 },
            [2]u16{ 65427, 16 },
            [2]u16{ 65428, 16 },
            [2]u16{ 65429, 16 },
            [2]u16{ 65430, 16 },
            [2]u16{ 0, 0 },
            [2]u16{ 0, 0 },
            [2]u16{ 0, 0 },
            [2]u16{ 0, 0 },
            [2]u16{ 0, 0 },
            [2]u16{ 0, 0 },
            [2]u16{ 58, 6 },
            [2]u16{ 502, 9 },
            [2]u16{ 65431, 16 },
            [2]u16{ 65432, 16 },
            [2]u16{ 65433, 16 },
            [2]u16{ 65434, 16 },
            [2]u16{ 65435, 16 },
            [2]u16{ 65436, 16 },
            [2]u16{ 65437, 16 },
            [2]u16{ 65438, 16 },
            [2]u16{ 0, 0 },
            [2]u16{ 0, 0 },
            [2]u16{ 0, 0 },
            [2]u16{ 0, 0 },
            [2]u16{ 0, 0 },
            [2]u16{ 0, 0 },
            [2]u16{ 59, 6 },
            [2]u16{ 1017, 10 },
            [2]u16{ 65439, 16 },
            [2]u16{ 65440, 16 },
            [2]u16{ 65441, 16 },
            [2]u16{ 65442, 16 },
            [2]u16{ 65443, 16 },
            [2]u16{ 65444, 16 },
            [2]u16{ 65445, 16 },
            [2]u16{ 65446, 16 },
            [2]u16{ 0, 0 },
            [2]u16{ 0, 0 },
            [2]u16{ 0, 0 },
            [2]u16{ 0, 0 },
            [2]u16{ 0, 0 },
            [2]u16{ 0, 0 },
            [2]u16{ 121, 7 },
            [2]u16{ 2039, 11 },
            [2]u16{ 65447, 16 },
            [2]u16{ 65448, 16 },
            [2]u16{ 65449, 16 },
            [2]u16{ 65450, 16 },
            [2]u16{ 65451, 16 },
            [2]u16{ 65452, 16 },
            [2]u16{ 65453, 16 },
            [2]u16{ 65454, 16 },
            [2]u16{ 0, 0 },
            [2]u16{ 0, 0 },
            [2]u16{ 0, 0 },
            [2]u16{ 0, 0 },
            [2]u16{ 0, 0 },
            [2]u16{ 0, 0 },
            [2]u16{ 122, 7 },
            [2]u16{ 2040, 11 },
            [2]u16{ 65455, 16 },
            [2]u16{ 65456, 16 },
            [2]u16{ 65457, 16 },
            [2]u16{ 65458, 16 },
            [2]u16{ 65459, 16 },
            [2]u16{ 65460, 16 },
            [2]u16{ 65461, 16 },
            [2]u16{ 65462, 16 },
            [2]u16{ 0, 0 },
            [2]u16{ 0, 0 },
            [2]u16{ 0, 0 },
            [2]u16{ 0, 0 },
            [2]u16{ 0, 0 },
            [2]u16{ 0, 0 },
            [2]u16{ 249, 8 },
            [2]u16{ 65463, 16 },
            [2]u16{ 65464, 16 },
            [2]u16{ 65465, 16 },
            [2]u16{ 65466, 16 },
            [2]u16{ 65467, 16 },
            [2]u16{ 65468, 16 },
            [2]u16{ 65469, 16 },
            [2]u16{ 65470, 16 },
            [2]u16{ 65471, 16 },
            [2]u16{ 0, 0 },
            [2]u16{ 0, 0 },
            [2]u16{ 0, 0 },
            [2]u16{ 0, 0 },
            [2]u16{ 0, 0 },
            [2]u16{ 0, 0 },
            [2]u16{ 503, 9 },
            [2]u16{ 65472, 16 },
            [2]u16{ 65473, 16 },
            [2]u16{ 65474, 16 },
            [2]u16{ 65475, 16 },
            [2]u16{ 65476, 16 },
            [2]u16{ 65477, 16 },
            [2]u16{ 65478, 16 },
            [2]u16{ 65479, 16 },
            [2]u16{ 65480, 16 },
            [2]u16{ 0, 0 },
            [2]u16{ 0, 0 },
            [2]u16{ 0, 0 },
            [2]u16{ 0, 0 },
            [2]u16{ 0, 0 },
            [2]u16{ 0, 0 },
            [2]u16{ 504, 9 },
            [2]u16{ 65481, 16 },
            [2]u16{ 65482, 16 },
            [2]u16{ 65483, 16 },
            [2]u16{ 65484, 16 },
            [2]u16{ 65485, 16 },
            [2]u16{ 65486, 16 },
            [2]u16{ 65487, 16 },
            [2]u16{ 65488, 16 },
            [2]u16{ 65489, 16 },
            [2]u16{ 0, 0 },
            [2]u16{ 0, 0 },
            [2]u16{ 0, 0 },
            [2]u16{ 0, 0 },
            [2]u16{ 0, 0 },
            [2]u16{ 0, 0 },
            [2]u16{ 505, 9 },
            [2]u16{ 65490, 16 },
            [2]u16{ 65491, 16 },
            [2]u16{ 65492, 16 },
            [2]u16{ 65493, 16 },
            [2]u16{ 65494, 16 },
            [2]u16{ 65495, 16 },
            [2]u16{ 65496, 16 },
            [2]u16{ 65497, 16 },
            [2]u16{ 65498, 16 },
            [2]u16{ 0, 0 },
            [2]u16{ 0, 0 },
            [2]u16{ 0, 0 },
            [2]u16{ 0, 0 },
            [2]u16{ 0, 0 },
            [2]u16{ 0, 0 },
            [2]u16{ 506, 9 },
            [2]u16{ 65499, 16 },
            [2]u16{ 65500, 16 },
            [2]u16{ 65501, 16 },
            [2]u16{ 65502, 16 },
            [2]u16{ 65503, 16 },
            [2]u16{ 65504, 16 },
            [2]u16{ 65505, 16 },
            [2]u16{ 65506, 16 },
            [2]u16{ 65507, 16 },
            [2]u16{ 0, 0 },
            [2]u16{ 0, 0 },
            [2]u16{ 0, 0 },
            [2]u16{ 0, 0 },
            [2]u16{ 0, 0 },
            [2]u16{ 0, 0 },
            [2]u16{ 2041, 11 },
            [2]u16{ 65508, 16 },
            [2]u16{ 65509, 16 },
            [2]u16{ 65510, 16 },
            [2]u16{ 65511, 16 },
            [2]u16{ 65512, 16 },
            [2]u16{ 65513, 16 },
            [2]u16{ 65514, 16 },
            [2]u16{ 65515, 16 },
            [2]u16{ 65516, 16 },
            [2]u16{ 0, 0 },
            [2]u16{ 0, 0 },
            [2]u16{ 0, 0 },
            [2]u16{ 0, 0 },
            [2]u16{ 0, 0 },
            [2]u16{ 0, 0 },
            [2]u16{ 16352, 14 },
            [2]u16{ 65517, 16 },
            [2]u16{ 65518, 16 },
            [2]u16{ 65519, 16 },
            [2]u16{ 65520, 16 },
            [2]u16{ 65521, 16 },
            [2]u16{ 65522, 16 },
            [2]u16{ 65523, 16 },
            [2]u16{ 65524, 16 },
            [2]u16{ 65525, 16 },
            [2]u16{ 0, 0 },
            [2]u16{ 0, 0 },
            [2]u16{ 0, 0 },
            [2]u16{ 0, 0 },
            [2]u16{ 0, 0 },
            [2]u16{ 1018, 10 },
            [2]u16{ 32707, 15 },
            [2]u16{ 65526, 16 },
            [2]u16{ 65527, 16 },
            [2]u16{ 65528, 16 },
            [2]u16{ 65529, 16 },
            [2]u16{ 65530, 16 },
            [2]u16{ 65531, 16 },
            [2]u16{ 65532, 16 },
            [2]u16{ 65533, 16 },
            [2]u16{ 65534, 16 },
            [2]u16{ 0, 0 },
            [2]u16{ 0, 0 },
            [2]u16{ 0, 0 },
            [2]u16{ 0, 0 },
            [2]u16{ 0, 0 },
        };
    };
    const static_local_YQT = struct {
        const YQT: [64]i32 = [64]i32{
            16, 11, 10, 16, 24,  40,  51,  61,  12, 12, 14, 19, 26,  58,  60,  55,
            14, 13, 16, 24, 40,  57,  69,  56,  14, 17, 22, 29, 51,  87,  80,  62,
            18, 22, 37, 56, 68,  109, 103, 77,  24, 35, 55, 64, 81,  104, 113, 92,
            49, 64, 78, 87, 103, 121, 120, 101, 72, 92, 95, 98, 112, 100, 103, 99,
        };
    };
    const static_local_UVQT = struct {
        const UVQT: [64]i32 = [64]i32{
            17, 18, 24, 47, 99, 99, 99, 99, 18, 21, 26, 66, 99, 99, 99, 99,
            24, 26, 56, 99, 99, 99, 99, 99, 47, 66, 99, 99, 99, 99, 99, 99,
            99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99,
            99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99,
        };
    };
    const static_local_aasf = struct {
        const aasf: [8]f32 = [8]f32{
            @as(f32, 1.0) * @as(f32, 2.828427),
            @as(f32, 1.38704) * @as(f32, 2.828427),
            @as(f32, 1.306563) * @as(f32, 2.828427),
            @as(f32, 1.175876) * @as(f32, 2.828427),
            @as(f32, 1.0) * @as(f32, 2.828427),
            @as(f32, 0.785695) * @as(f32, 2.828427),
            @as(f32, 0.541196) * @as(f32, 2.828427),
            @as(f32, 0.275899) * @as(f32, 2.828427),
        };
    };
    var row: i32 = undefined;
    var col: i32 = undefined;
    var i: i32 = undefined;
    var k: i32 = undefined;
    var subsample: i32 = undefined;
    var fdtbl_Y: [64]f32 = undefined;
    var fdtbl_UV: [64]f32 = undefined;
    var YTable: [64]u8 = undefined;
    var UVTable: [64]u8 = undefined;
    if ((((!(data != null) or !(width != 0)) or !(height != 0)) or (comp > @as(i32, 4))) or (comp < @as(i32, 1))) {
        return 0;
    }
    quality = if (quality != 0) quality else @as(i32, 90);
    subsample = if (quality <= @as(i32, 90)) @as(i32, 1) else @as(i32, 0);
    quality = if (quality < @as(i32, 1)) @as(i32, 1) else if (quality > @as(i32, 100)) @as(i32, 100) else quality;
    quality = if (quality < @as(i32, 50)) @divTrunc(@as(i32, 5000), quality) else @as(i32, 200) - (quality * @as(i32, 2));
    {
        i = 0;
        while (i < @as(i32, 64)) : (i += 1) {
            var uvti: i32 = undefined;
            const yti: i32 = @divTrunc((static_local_YQT.YQT[@bitCast(@as(isize, @intCast(i)))] * quality) + @as(i32, 50), @as(i32, 100));
            YTable[stbiw__jpg_ZigZag[@bitCast(@as(isize, @intCast(i)))]] = @bitCast(@as(i8, @truncate(if (yti < @as(i32, 1)) @as(i32, 1) else if (yti > @as(i32, 255)) @as(i32, 255) else yti)));
            uvti = @divTrunc((static_local_UVQT.UVQT[@bitCast(@as(isize, @intCast(i)))] * quality) + @as(i32, 50), @as(i32, 100));
            UVTable[stbiw__jpg_ZigZag[@bitCast(@as(isize, @intCast(i)))]] = @bitCast(@as(i8, @truncate(if (uvti < @as(i32, 1)) @as(i32, 1) else if (uvti > @as(i32, 255)) @as(i32, 255) else uvti)));
        }
    }
    {
        row = 0;
        k = 0;
        while (row < @as(i32, 8)) : (row += 1) {
            {
                col = 0;
                while (col < @as(i32, 8)) : ({
                    col += 1;
                    k += 1;
                }) {
                    fdtbl_Y[@bitCast(@as(isize, @intCast(k)))] = @as(f32, @floatFromInt(@as(i32, 1))) / ((@as(f32, @floatFromInt(@as(i32, YTable[stbiw__jpg_ZigZag[@bitCast(@as(isize, @intCast(k)))]]))) * static_local_aasf.aasf[@bitCast(@as(isize, @intCast(row)))]) * static_local_aasf.aasf[@bitCast(@as(isize, @intCast(col)))]);
                    fdtbl_UV[@bitCast(@as(isize, @intCast(k)))] = @as(f32, @floatFromInt(@as(i32, 1))) / ((@as(f32, @floatFromInt(@as(i32, UVTable[stbiw__jpg_ZigZag[@bitCast(@as(isize, @intCast(k)))]]))) * static_local_aasf.aasf[@bitCast(@as(isize, @intCast(row)))]) * static_local_aasf.aasf[@bitCast(@as(isize, @intCast(col)))]);
                }
            }
        }
    }
    {
        const static_local_head0 = struct {
            const head0: [25]u8 = [25]u8{
                255,
                216,
                255,
                224,
                0,
                16,
                'J',
                'F',
                'I',
                'F',
                0,
                1,
                1,
                0,
                0,
                1,
                0,
                1,
                0,
                0,
                255,
                219,
                0,
                132,
                0,
            };
        };
        const static_local_head2 = struct {
            const head2: [14]u8 = [14]u8{
                255, 218, 0, 12, 3, 1, 0, 2, 17, 3, 17, 0, 63, 0,
            };
        };
        const head1: [24]u8 = [24]u8{
            255,
            192,
            0,
            17,
            8,
            @bitCast(@as(i8, @truncate(height >> @intCast(@as(i32, 8))))),
            @bitCast(@as(i8, @truncate(height & @as(i32, 255)))),
            @bitCast(@as(i8, @truncate(width >> @intCast(@as(i32, 8))))),
            @bitCast(@as(i8, @truncate(width & @as(i32, 255)))),
            3,
            1,
            @bitCast(@as(i8, @truncate(if (subsample != 0) @as(i32, 34) else @as(i32, 17)))),
            0,
            2,
            17,
            1,
            3,
            17,
            1,
            255,
            196,
            1,
            162,
            0,
        };
        s.*.func.?(s.*.context, @ptrCast(@alignCast(@constCast(@as([*c]const u8, @ptrCast(@alignCast(&static_local_head0.head0)))))), @bitCast(@as(u32, @truncate(@sizeOf(@TypeOf(static_local_head0.head0))))));
        s.*.func.?(s.*.context, @ptrCast(@alignCast(@as([*c]u8, @ptrCast(@alignCast(&YTable))))), @bitCast(@as(u32, @truncate(@sizeOf(@TypeOf(YTable))))));
        stbiw__putc(s, 1);
        s.*.func.?(s.*.context, @ptrCast(@alignCast(@as([*c]u8, @ptrCast(@alignCast(&UVTable))))), @bitCast(@as(u32, @truncate(@sizeOf(@TypeOf(UVTable))))));
        s.*.func.?(s.*.context, @ptrCast(@alignCast(@constCast(@as([*c]const u8, @ptrCast(@alignCast(&head1)))))), @bitCast(@as(u32, @truncate(@sizeOf(@TypeOf(head1))))));
        s.*.func.?(s.*.context, @ptrCast(@alignCast(@constCast(@as([*c]const u8, @ptrCast(@alignCast(&static_local_std_dc_luminance_nrcodes.std_dc_luminance_nrcodes))) + @as(usize, @bitCast(@as(isize, @intCast(@as(i32, 1)))))))), @bitCast(@as(u32, @truncate(@sizeOf(@TypeOf(static_local_std_dc_luminance_nrcodes.std_dc_luminance_nrcodes)) -% @as(u64, 1)))));
        s.*.func.?(s.*.context, @ptrCast(@alignCast(@constCast(@as([*c]const u8, @ptrCast(@alignCast(&static_local_std_dc_luminance_values.std_dc_luminance_values)))))), @bitCast(@as(u32, @truncate(@sizeOf(@TypeOf(static_local_std_dc_luminance_values.std_dc_luminance_values))))));
        stbiw__putc(s, 16);
        s.*.func.?(s.*.context, @ptrCast(@alignCast(@constCast(@as([*c]const u8, @ptrCast(@alignCast(&static_local_std_ac_luminance_nrcodes.std_ac_luminance_nrcodes))) + @as(usize, @bitCast(@as(isize, @intCast(@as(i32, 1)))))))), @bitCast(@as(u32, @truncate(@sizeOf(@TypeOf(static_local_std_ac_luminance_nrcodes.std_ac_luminance_nrcodes)) -% @as(u64, 1)))));
        s.*.func.?(s.*.context, @ptrCast(@alignCast(@constCast(@as([*c]const u8, @ptrCast(@alignCast(&static_local_std_ac_luminance_values.std_ac_luminance_values)))))), @bitCast(@as(u32, @truncate(@sizeOf(@TypeOf(static_local_std_ac_luminance_values.std_ac_luminance_values))))));
        stbiw__putc(s, 1);
        s.*.func.?(s.*.context, @ptrCast(@alignCast(@constCast(@as([*c]const u8, @ptrCast(@alignCast(&static_local_std_dc_chrominance_nrcodes.std_dc_chrominance_nrcodes))) + @as(usize, @bitCast(@as(isize, @intCast(@as(i32, 1)))))))), @bitCast(@as(u32, @truncate(@sizeOf(@TypeOf(static_local_std_dc_chrominance_nrcodes.std_dc_chrominance_nrcodes)) -% @as(u64, 1)))));
        s.*.func.?(s.*.context, @ptrCast(@alignCast(@constCast(@as([*c]const u8, @ptrCast(@alignCast(&static_local_std_dc_chrominance_values.std_dc_chrominance_values)))))), @bitCast(@as(u32, @truncate(@sizeOf(@TypeOf(static_local_std_dc_chrominance_values.std_dc_chrominance_values))))));
        stbiw__putc(s, 17);
        s.*.func.?(s.*.context, @ptrCast(@alignCast(@constCast(@as([*c]const u8, @ptrCast(@alignCast(&static_local_std_ac_chrominance_nrcodes.std_ac_chrominance_nrcodes))) + @as(usize, @bitCast(@as(isize, @intCast(@as(i32, 1)))))))), @bitCast(@as(u32, @truncate(@sizeOf(@TypeOf(static_local_std_ac_chrominance_nrcodes.std_ac_chrominance_nrcodes)) -% @as(u64, 1)))));
        s.*.func.?(s.*.context, @ptrCast(@alignCast(@constCast(@as([*c]const u8, @ptrCast(@alignCast(&static_local_std_ac_chrominance_values.std_ac_chrominance_values)))))), @bitCast(@as(u32, @truncate(@sizeOf(@TypeOf(static_local_std_ac_chrominance_values.std_ac_chrominance_values))))));
        s.*.func.?(s.*.context, @ptrCast(@alignCast(@constCast(@as([*c]const u8, @ptrCast(@alignCast(&static_local_head2.head2)))))), @bitCast(@as(u32, @truncate(@sizeOf(@TypeOf(static_local_head2.head2))))));
    }
    {
        const static_local_fillBits = struct {
            const fillBits: [2]u16 = [2]u16{ 127, 7 };
        };
        var DCY: i32 = 0;
        var DCU: i32 = 0;
        var DCV: i32 = 0;
        var bitBuf: i32 = 0;
        var bitCnt: i32 = 0;
        const ofsG: i32 = if (comp > @as(i32, 2)) @as(i32, 1) else @as(i32, 0);
        const ofsB: i32 = if (comp > @as(i32, 2)) @as(i32, 2) else @as(i32, 0);
        const dataR: [*c]const u8 = @ptrCast(@alignCast(data));
        const dataG: [*c]const u8 = dataR + @as(usize, @bitCast(@as(isize, @intCast(ofsG))));
        const dataB: [*c]const u8 = dataR + @as(usize, @bitCast(@as(isize, @intCast(ofsB))));
        var x: i32 = undefined;
        var y: i32 = undefined;
        var pos: i32 = undefined;
        if (subsample != 0) {
            {
                y = 0;
                while (y < height) : (y += 16) {
                    {
                        x = 0;
                        while (x < width) : (x += 16) {
                            var Y: [256]f32 = undefined;
                            var U: [256]f32 = undefined;
                            var V: [256]f32 = undefined;
                            {
                                row = y;
                                pos = 0;
                                while (row < (y + @as(i32, 16))) : (row += 1) {
                                    const clamped_row: i32 = if (row < height) row else height - @as(i32, 1);
                                    const base_p: i32 = ((if (stbi__flip_vertically_on_write != 0) (height - @as(i32, 1)) - clamped_row else clamped_row) * width) * comp;
                                    {
                                        col = x;
                                        while (col < (x + @as(i32, 16))) : ({
                                            col += 1;
                                            pos += 1;
                                        }) {
                                            const p: i32 = base_p + ((if (col < width) col else width - @as(i32, 1)) * comp);
                                            const r: f32 = @floatFromInt(dataR[@bitCast(@as(isize, @intCast(p)))]);
                                            const g: f32 = @floatFromInt(dataG[@bitCast(@as(isize, @intCast(p)))]);
                                            const b: f32 = @floatFromInt(dataB[@bitCast(@as(isize, @intCast(p)))]);
                                            Y[@bitCast(@as(isize, @intCast(pos)))] = (((@as(f32, 0.299) * r) + (@as(f32, 0.587) * g)) + (@as(f32, 0.114) * b)) - @as(f32, @floatFromInt(@as(i32, 128)));
                                            U[@bitCast(@as(isize, @intCast(pos)))] = ((-@as(f32, 0.16874) * r) - (@as(f32, 0.33126) * g)) + (@as(f32, 0.5) * b);
                                            V[@bitCast(@as(isize, @intCast(pos)))] = ((@as(f32, 0.5) * r) - (@as(f32, 0.41869) * g)) - (@as(f32, 0.08131) * b);
                                        }
                                    }
                                }
                            }
                            DCY = stbiw__jpg_processDU(s, &bitBuf, &bitCnt, @as([*c]f32, @ptrCast(@alignCast(&Y))) + @as(usize, @bitCast(@as(isize, @intCast(@as(i32, 0))))), 16, @ptrCast(@alignCast(&fdtbl_Y)), DCY, @ptrCast(@alignCast(&static_local_YDC_HT.YDC_HT)), @ptrCast(@alignCast(&static_local_YAC_HT.YAC_HT)));
                            DCY = stbiw__jpg_processDU(s, &bitBuf, &bitCnt, @as([*c]f32, @ptrCast(@alignCast(&Y))) + @as(usize, @bitCast(@as(isize, @intCast(@as(i32, 8))))), 16, @ptrCast(@alignCast(&fdtbl_Y)), DCY, @ptrCast(@alignCast(&static_local_YDC_HT.YDC_HT)), @ptrCast(@alignCast(&static_local_YAC_HT.YAC_HT)));
                            DCY = stbiw__jpg_processDU(s, &bitBuf, &bitCnt, @as([*c]f32, @ptrCast(@alignCast(&Y))) + @as(usize, @bitCast(@as(isize, @intCast(@as(i32, 128))))), 16, @ptrCast(@alignCast(&fdtbl_Y)), DCY, @ptrCast(@alignCast(&static_local_YDC_HT.YDC_HT)), @ptrCast(@alignCast(&static_local_YAC_HT.YAC_HT)));
                            DCY = stbiw__jpg_processDU(s, &bitBuf, &bitCnt, @as([*c]f32, @ptrCast(@alignCast(&Y))) + @as(usize, @bitCast(@as(isize, @intCast(@as(i32, 136))))), 16, @ptrCast(@alignCast(&fdtbl_Y)), DCY, @ptrCast(@alignCast(&static_local_YDC_HT.YDC_HT)), @ptrCast(@alignCast(&static_local_YAC_HT.YAC_HT)));
                            {
                                var subU: [64]f32 = undefined;
                                var subV: [64]f32 = undefined;
                                var yy: i32 = undefined;
                                var xx: i32 = undefined;
                                {
                                    yy = 0;
                                    pos = 0;
                                    while (yy < @as(i32, 8)) : (yy += 1) {
                                        {
                                            xx = 0;
                                            while (xx < @as(i32, 8)) : ({
                                                xx += 1;
                                                pos += 1;
                                            }) {
                                                const j: i32 = (yy * @as(i32, 32)) + (xx * @as(i32, 2));
                                                subU[@bitCast(@as(isize, @intCast(pos)))] = (((U[@bitCast(@as(isize, @intCast(j + @as(i32, 0))))] + U[@bitCast(@as(isize, @intCast(j + @as(i32, 1))))]) + U[@bitCast(@as(isize, @intCast(j + @as(i32, 16))))]) + U[@bitCast(@as(isize, @intCast(j + @as(i32, 17))))]) * @as(f32, 0.25);
                                                subV[@bitCast(@as(isize, @intCast(pos)))] = (((V[@bitCast(@as(isize, @intCast(j + @as(i32, 0))))] + V[@bitCast(@as(isize, @intCast(j + @as(i32, 1))))]) + V[@bitCast(@as(isize, @intCast(j + @as(i32, 16))))]) + V[@bitCast(@as(isize, @intCast(j + @as(i32, 17))))]) * @as(f32, 0.25);
                                            }
                                        }
                                    }
                                }
                                DCU = stbiw__jpg_processDU(s, &bitBuf, &bitCnt, @ptrCast(@alignCast(&subU)), 8, @ptrCast(@alignCast(&fdtbl_UV)), DCU, @ptrCast(@alignCast(&static_local_UVDC_HT.UVDC_HT)), @ptrCast(@alignCast(&static_local_UVAC_HT.UVAC_HT)));
                                DCV = stbiw__jpg_processDU(s, &bitBuf, &bitCnt, @ptrCast(@alignCast(&subV)), 8, @ptrCast(@alignCast(&fdtbl_UV)), DCV, @ptrCast(@alignCast(&static_local_UVDC_HT.UVDC_HT)), @ptrCast(@alignCast(&static_local_UVAC_HT.UVAC_HT)));
                            }
                        }
                    }
                }
            }
        } else {
            {
                y = 0;
                while (y < height) : (y += 8) {
                    {
                        x = 0;
                        while (x < width) : (x += 8) {
                            var Y: [64]f32 = undefined;
                            var U: [64]f32 = undefined;
                            var V: [64]f32 = undefined;
                            {
                                row = y;
                                pos = 0;
                                while (row < (y + @as(i32, 8))) : (row += 1) {
                                    const clamped_row: i32 = if (row < height) row else height - @as(i32, 1);
                                    const base_p: i32 = ((if (stbi__flip_vertically_on_write != 0) (height - @as(i32, 1)) - clamped_row else clamped_row) * width) * comp;
                                    {
                                        col = x;
                                        while (col < (x + @as(i32, 8))) : ({
                                            col += 1;
                                            pos += 1;
                                        }) {
                                            const p: i32 = base_p + ((if (col < width) col else width - @as(i32, 1)) * comp);
                                            const r: f32 = @floatFromInt(dataR[@bitCast(@as(isize, @intCast(p)))]);
                                            const g: f32 = @floatFromInt(dataG[@bitCast(@as(isize, @intCast(p)))]);
                                            const b: f32 = @floatFromInt(dataB[@bitCast(@as(isize, @intCast(p)))]);
                                            Y[@bitCast(@as(isize, @intCast(pos)))] = (((@as(f32, 0.299) * r) + (@as(f32, 0.587) * g)) + (@as(f32, 0.114) * b)) - @as(f32, @floatFromInt(@as(i32, 128)));
                                            U[@bitCast(@as(isize, @intCast(pos)))] = ((-@as(f32, 0.16874) * r) - (@as(f32, 0.33126) * g)) + (@as(f32, 0.5) * b);
                                            V[@bitCast(@as(isize, @intCast(pos)))] = ((@as(f32, 0.5) * r) - (@as(f32, 0.41869) * g)) - (@as(f32, 0.08131) * b);
                                        }
                                    }
                                }
                            }
                            DCY = stbiw__jpg_processDU(s, &bitBuf, &bitCnt, @ptrCast(@alignCast(&Y)), 8, @ptrCast(@alignCast(&fdtbl_Y)), DCY, @ptrCast(@alignCast(&static_local_YDC_HT.YDC_HT)), @ptrCast(@alignCast(&static_local_YAC_HT.YAC_HT)));
                            DCU = stbiw__jpg_processDU(s, &bitBuf, &bitCnt, @ptrCast(@alignCast(&U)), 8, @ptrCast(@alignCast(&fdtbl_UV)), DCU, @ptrCast(@alignCast(&static_local_UVDC_HT.UVDC_HT)), @ptrCast(@alignCast(&static_local_UVAC_HT.UVAC_HT)));
                            DCV = stbiw__jpg_processDU(s, &bitBuf, &bitCnt, @ptrCast(@alignCast(&V)), 8, @ptrCast(@alignCast(&fdtbl_UV)), DCV, @ptrCast(@alignCast(&static_local_UVDC_HT.UVDC_HT)), @ptrCast(@alignCast(&static_local_UVAC_HT.UVAC_HT)));
                        }
                    }
                }
            }
        }
        stbiw__jpg_writeBits(s, &bitBuf, &bitCnt, @ptrCast(@alignCast(&static_local_fillBits.fillBits)));
    }
    stbiw__putc(s, 255);
    stbiw__putc(s, 217);
    return 1;
}
pub fn stbi_write_jpg_to_func(func: ?*const stbi_write_func, context: ?*anyopaque, x: i32, y: i32, comp: i32, data: ?*const anyopaque, quality: i32) i32 {
    var s: stbi__write_context = stbi__write_context{
        .func = null,
        .context = null,
        .buffer = std.mem.zeroes([64]u8),
        .buf_used = 0,
    };
    stbi__start_write_callbacks(&s, func, context);
    return stbi_write_jpg_core(&s, x, y, comp, @ptrCast(@alignCast(data)), quality);
}

const __SHRT_MAX__: i32 = 32767;
const FAST_BITS: i32 = 9;
const STBI__MARKER_none: i32 = 0xff;
const STBI__ZFAST_BITS: i32 = 9;
const STBI__ZNSYMS: i32 = 288;
const __INT_MAX__: i32 = 2147483647;
