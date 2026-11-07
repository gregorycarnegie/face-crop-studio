//! RGB images for the dataset tools: stb for the codecs, and OpenCV's own fixed-point arithmetic
//! for `resize` (INTER_LINEAR, INTER_AREA) and `warpAffine` (INTER_LINEAR, constant border), so
//! the Zig ports feed the models what the Python versions did. Decoding is stb's, not libjpeg's,
//! so pixels can differ from `cv2.imread` by a level or two; the geometry is OpenCV's exactly.
//!
//! Link with `-lc tools/dataset/vendor/stb.c`.
const std = @import("std");
const Allocator = std.mem.Allocator;

extern fn stbi_load_from_memory(buffer: [*]const u8, len: c_int, x: *c_int, y: *c_int, comp: *c_int, req_comp: c_int) ?[*]u8;
extern fn stbi_image_free(retval_from_stbi_load: ?*anyopaque) void;
const WriteFn = *const fn (?*anyopaque, ?*anyopaque, c_int) callconv(.c) void;
extern fn stbi_write_jpg_to_func(func: WriteFn, context: ?*anyopaque, w: c_int, h: c_int, comp: c_int, data: *const anyopaque, quality: c_int) c_int;
extern fn stbi_write_png_to_func(func: WriteFn, context: ?*anyopaque, w: c_int, h: c_int, comp: c_int, data: *const anyopaque, stride: c_int) c_int;

/// Interleaved RGB, row-major. (OpenCV's BGR order is swapped wherever a model wants BGR.)
pub const Image = struct {
    width: usize,
    height: usize,
    pixels: []u8,

    pub fn init(gpa: Allocator, width: usize, height: usize) !Image {
        const pixels = try gpa.alloc(u8, try std.math.mul(usize, try std.math.mul(usize, width, height), 3));
        @memset(pixels, 0);
        return .{ .width = width, .height = height, .pixels = pixels };
    }

    pub fn deinit(self: Image, gpa: Allocator) void {
        gpa.free(self.pixels);
    }

    pub fn at(self: Image, x: usize, y: usize) *[3]u8 {
        return self.pixels[(y * self.width + x) * 3 ..][0..3];
    }
};

pub const Orientation = enum {
    /// `cv2.imread(path)`: the EXIF orientation is applied, as OpenCV does by default.
    apply_exif,
    /// `cv2.IMREAD_IGNORE_ORIENTATION`: pixels as stored, which is what Open Images boxes describe.
    as_stored,
};

/// Decode a JPEG or PNG. `error.UndecodableImage` where `cv2.imread` would return None.
pub fn decode(gpa: Allocator, data: []const u8, orientation: Orientation) !Image {
    var w: c_int = 0;
    var h: c_int = 0;
    var channels: c_int = 0;
    const len = std.math.cast(c_int, data.len) orelse return error.UndecodableImage;
    const raw = stbi_load_from_memory(data.ptr, len, &w, &h, &channels, 3) orelse return error.UndecodableImage;
    defer stbi_image_free(raw);
    const width: usize = @intCast(w);
    const height: usize = @intCast(h);
    const stored: Image = .{ .width = width, .height = height, .pixels = try gpa.dupe(u8, raw[0 .. width * height * 3]) };
    if (orientation == .as_stored) return stored;
    const exif = exifOrientation(data);
    if (exif == 1) return stored;
    defer stored.deinit(gpa);
    return orient(gpa, stored, exif);
}

pub fn load(gpa: Allocator, io: std.Io, path: []const u8, orientation: Orientation) !Image {
    const data = try std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .unlimited);
    defer gpa.free(data);
    return decode(gpa, data, orientation);
}

/// The EXIF orientation tag (1-8) of a JPEG, or 1 when there is none.
pub fn exifOrientation(data: []const u8) u16 {
    if (data.len < 4 or data[0] != 0xFF or data[1] != 0xD8) return 1;
    var i: usize = 2;
    while (i + 4 <= data.len) {
        if (data[i] != 0xFF) return 1;
        while (i + 1 < data.len and data[i + 1] == 0xFF) i += 1; // fill bytes
        if (i + 4 > data.len) return 1;
        const marker = data[i + 1];
        if (marker == 0xDA or marker == 0xD9) return 1; // image data starts: no EXIF came first
        const len = std.mem.readInt(u16, data[i + 2 ..][0..2], .big);
        if (len < 2 or i + 2 + len > data.len) return 1;
        const segment = data[i + 4 .. i + 2 + len];
        if (marker == 0xE1 and segment.len >= 14 and std.mem.eql(u8, segment[0..6], "Exif\x00\x00")) return tiffOrientation(segment[6..]);
        i += 2 + len;
    }
    return 1;
}

fn tiffOrientation(tiff: []const u8) u16 {
    if (tiff.len < 8) return 1;
    const endian: std.builtin.Endian = if (std.mem.eql(u8, tiff[0..2], "II")) .little else if (std.mem.eql(u8, tiff[0..2], "MM")) .big else return 1;
    const ifd = std.mem.readInt(u32, tiff[4..8], endian);
    if (ifd > tiff.len - 2) return 1;
    const count = std.mem.readInt(u16, tiff[ifd..][0..2], endian);
    for (0..count) |k| {
        const entry = ifd + 2 + k * 12;
        if (entry + 12 > tiff.len) return 1;
        if (std.mem.readInt(u16, tiff[entry..][0..2], endian) == 0x0112) {
            const value = std.mem.readInt(u16, tiff[entry + 8 ..][0..2], endian);
            return if (value >= 1 and value <= 8) value else 1;
        }
    }
    return 1;
}

/// OpenCV's `ExifTransform`: flips for 2-4, transpose plus a flip for 5-8.
fn orient(gpa: Allocator, src: Image, exif: u16) !Image {
    const w = src.width;
    const h = src.height;
    const swap = exif >= 5;
    const out = try Image.init(gpa, if (swap) h else w, if (swap) w else h);
    for (0..out.height) |y| for (0..out.width) |x| {
        const sx, const sy = switch (exif) {
            2 => .{ w - 1 - x, y },
            3 => .{ w - 1 - x, h - 1 - y },
            4 => .{ x, h - 1 - y },
            5 => .{ y, x },
            6 => .{ y, h - 1 - x },
            7 => .{ w - 1 - y, h - 1 - x },
            8 => .{ w - 1 - y, x },
            else => .{ x, y },
        };
        out.at(x, y).* = src.at(sx, sy).*;
    };
    return out;
}

const Sink = struct {
    gpa: Allocator,
    bytes: std.ArrayList(u8) = .empty,
    failed: bool = false,

    fn write(context: ?*anyopaque, data: ?*anyopaque, size: c_int) callconv(.c) void {
        const self: *Sink = @ptrCast(@alignCast(context.?));
        const chunk = @as([*]const u8, @ptrCast(data.?))[0..@intCast(size)];
        self.bytes.appendSlice(self.gpa, chunk) catch {
            self.failed = true;
        };
    }
};

pub fn encodeJpeg(gpa: Allocator, img: Image, quality: u8) ![]u8 {
    var sink: Sink = .{ .gpa = gpa };
    errdefer sink.bytes.deinit(gpa);
    if (stbi_write_jpg_to_func(Sink.write, &sink, try dim(img.width), try dim(img.height), 3, img.pixels.ptr, quality) == 0 or sink.failed) return error.EncodeFailed;
    return sink.bytes.toOwnedSlice(gpa);
}

pub fn encodePng(gpa: Allocator, img: Image) ![]u8 {
    var sink: Sink = .{ .gpa = gpa };
    errdefer sink.bytes.deinit(gpa);
    if (stbi_write_png_to_func(Sink.write, &sink, try dim(img.width), try dim(img.height), 3, img.pixels.ptr, try dim(img.width * 3)) == 0 or sink.failed) return error.EncodeFailed;
    return sink.bytes.toOwnedSlice(gpa);
}

fn dim(n: usize) !c_int {
    return std.math.cast(c_int, n) orelse error.ImageTooLarge;
}

/// `cvRound`: to nearest, ties to even (SSE2's default rounding mode).
pub fn rint(x: f64) f64 {
    if (@abs(x - @trunc(x)) == 0.5) return 2 * @round(x / 2);
    return @round(x);
}

fn saturateInt(x: f64) i32 {
    return @intFromFloat(std.math.clamp(rint(x), std.math.minInt(i32), std.math.maxInt(i32)));
}

fn saturateShort(x: f64) i32 {
    return @intFromFloat(std.math.clamp(rint(x), -32768, 32767));
}

const Tap = struct { first: usize, second: usize, a0: i32, a1: i32 };

/// `resizeGeneric_`'s INTER_LINEAR table: centre-aligned source coordinate, clamped at the edges,
/// weights in 11-bit fixed point.
fn linearTaps(gpa: Allocator, src_n: usize, dst_n: usize, scale: f64) ![]Tap {
    const taps = try gpa.alloc(Tap, dst_n);
    for (taps, 0..) |*t, d| {
        var f: f32 = @floatCast((@as(f64, @floatFromInt(d)) + 0.5) * scale - 0.5);
        var s: i64 = @intFromFloat(@floor(f));
        f -= @floatFromInt(s);
        if (s < 0) {
            f = 0;
            s = 0;
        }
        if (s >= @as(i64, @intCast(src_n)) - 1) {
            f = 0;
            s = @as(i64, @intCast(src_n)) - 1;
        }
        const first: usize = @intCast(s);
        t.* = .{ .first = first, .second = @min(first + 1, src_n - 1), .a0 = saturateShort((1 - f) * 2048), .a1 = saturateShort(f * 2048) };
    }
    return taps;
}

/// `cv2.resize(src, (width, height))`, INTER_LINEAR.
pub fn resizeLinear(gpa: Allocator, src: Image, width: usize, height: usize) !Image {
    if (width == 0 or height == 0 or src.width == 0 or src.height == 0) return error.EmptyImage;
    if (width == src.width and height == src.height) return .{ .width = width, .height = height, .pixels = try gpa.dupe(u8, src.pixels) };
    const scale_x = 1 / (@as(f64, @floatFromInt(width)) / @as(f64, @floatFromInt(src.width)));
    const scale_y = 1 / (@as(f64, @floatFromInt(height)) / @as(f64, @floatFromInt(src.height)));
    // OpenCV hands an exact halving to INTER_AREA, whose rounded 2x2 mean equals bit-exact linear.
    if (scale_x == 2 and scale_y == 2) return resizeArea(gpa, src, width, height);
    const xs = try linearTaps(gpa, src.width, width, scale_x);
    defer gpa.free(xs);
    const ys = try linearTaps(gpa, src.height, height, scale_y);
    defer gpa.free(ys);
    const out = try Image.init(gpa, width, height);
    const row0 = try gpa.alloc(i32, width * 3);
    defer gpa.free(row0);
    const row1 = try gpa.alloc(i32, width * 3);
    defer gpa.free(row1);
    for (ys, 0..) |ty, y| {
        for ([_][]i32{ row0, row1 }, [_]usize{ ty.first, ty.second }) |row, sy| {
            const line = src.pixels[sy * src.width * 3 ..][0 .. src.width * 3];
            for (xs, 0..) |tx, x| for (0..3) |c| {
                row[x * 3 + c] = @as(i32, line[tx.first * 3 + c]) * tx.a0 + @as(i32, line[tx.second * 3 + c]) * tx.a1;
            };
        }
        const dst = out.pixels[y * width * 3 ..][0 .. width * 3];
        // VResizeLinear<uchar>: the >>4 / >>16 split keeps the products inside 32 bits.
        for (dst, row0, row1) |*d, s0, s1| d.* = @truncate(@as(u32, @bitCast((((ty.a0 * (s0 >> 4)) >> 16) + ((ty.a1 * (s1 >> 4)) >> 16) + 2) >> 2)));
    }
    return out;
}

/// `cv2.resize` of a float32 3-channel HWC image, INTER_LINEAR: OpenCV's float path, whose
/// weights stay floating point (no 11-bit quantisation).
pub fn resizeLinearF32(gpa: Allocator, src: []const f32, sw: usize, sh: usize, dw: usize, dh: usize) ![]f32 {
    if (sw == 0 or sh == 0 or dw == 0 or dh == 0 or src.len != sw * sh * 3) return error.EmptyImage;
    if (sw == dw and sh == dh) return gpa.dupe(f32, src);
    const scale_x = 1 / (@as(f64, @floatFromInt(dw)) / @as(f64, @floatFromInt(sw)));
    const scale_y = 1 / (@as(f64, @floatFromInt(dh)) / @as(f64, @floatFromInt(sh)));
    const Weights = struct { first: usize, second: usize, a0: f32, a1: f32 };
    const taps = struct {
        fn make(g: Allocator, n: usize, d: usize, scale: f64) ![]Weights {
            const out = try g.alloc(Weights, d);
            for (out, 0..) |*t, i| {
                var f: f32 = @floatCast((@as(f64, @floatFromInt(i)) + 0.5) * scale - 0.5);
                var s: i64 = @intFromFloat(@floor(f));
                f -= @floatFromInt(s);
                if (s < 0) {
                    f = 0;
                    s = 0;
                }
                if (s >= @as(i64, @intCast(n)) - 1) {
                    f = 0;
                    s = @as(i64, @intCast(n)) - 1;
                }
                const first: usize = @intCast(s);
                t.* = .{ .first = first, .second = @min(first + 1, n - 1), .a0 = 1 - f, .a1 = f };
            }
            return out;
        }
    }.make;
    const xs = try taps(gpa, sw, dw, scale_x);
    defer gpa.free(xs);
    const ys = try taps(gpa, sh, dh, scale_y);
    defer gpa.free(ys);
    const out = try gpa.alloc(f32, dw * dh * 3);
    const rows = try gpa.alloc(f32, dw * 3 * 2);
    defer gpa.free(rows);
    for (ys, 0..) |ty, y| {
        for ([_]usize{ ty.first, ty.second }, 0..) |sy, k| {
            const line = src[sy * sw * 3 ..][0 .. sw * 3];
            for (xs, 0..) |tx, x| for (0..3) |c| {
                rows[k * dw * 3 + x * 3 + c] = line[tx.first * 3 + c] * tx.a0 + line[tx.second * 3 + c] * tx.a1;
            };
        }
        for (out[y * dw * 3 ..][0 .. dw * 3], rows[0 .. dw * 3], rows[dw * 3 ..]) |*d, r0, r1| d.* = r0 * ty.a0 + r1 * ty.a1;
    }
    return out;
}

const AreaTap = struct { dst: usize, src: usize, alpha: f32 };

/// `computeResizeAreaTab`: each destination cell's overlap with the source cells it covers.
fn areaTaps(gpa: Allocator, src_n: usize, dst_n: usize, scale: f64) ![]AreaTap {
    var taps: std.ArrayList(AreaTap) = .empty;
    const n: f64 = @floatFromInt(src_n);
    for (0..dst_n) |d| {
        const f1 = @as(f64, @floatFromInt(d)) * scale;
        const f2 = f1 + scale;
        const cell = @min(scale, n - f1);
        var s2: i64 = @intFromFloat(@floor(f2));
        s2 = @min(s2, @as(i64, @intCast(src_n)) - 1);
        const s1: i64 = @min(@as(i64, @intFromFloat(@ceil(f1))), s2);
        if (@as(f64, @floatFromInt(s1)) - f1 > 1e-3) try taps.append(gpa, .{ .dst = d, .src = @intCast(s1 - 1), .alpha = @floatCast((@as(f64, @floatFromInt(s1)) - f1) / cell) });
        var s = s1;
        while (s < s2) : (s += 1) try taps.append(gpa, .{ .dst = d, .src = @intCast(s), .alpha = @floatCast(1.0 / cell) });
        if (f2 - @as(f64, @floatFromInt(s2)) > 1e-3) try taps.append(gpa, .{ .dst = d, .src = @intCast(s2), .alpha = @floatCast(@min(@min(f2 - @as(f64, @floatFromInt(s2)), 1.0), cell) / cell) });
    }
    return taps.toOwnedSlice(gpa);
}

/// `cv2.resize(src, (width, height), interpolation=cv2.INTER_AREA)`, for shrinking.
pub fn resizeArea(gpa: Allocator, src: Image, width: usize, height: usize) !Image {
    if (width == 0 or height == 0 or width > src.width or height > src.height) return error.UnsupportedResize;
    if (width == src.width and height == src.height) return .{ .width = width, .height = height, .pixels = try gpa.dupe(u8, src.pixels) };
    const scale_x = 1 / (@as(f64, @floatFromInt(width)) / @as(f64, @floatFromInt(src.width)));
    const scale_y = 1 / (@as(f64, @floatFromInt(height)) / @as(f64, @floatFromInt(src.height)));
    const out = try Image.init(gpa, width, height);
    const ix: usize = @intFromFloat(@round(scale_x));
    const iy: usize = @intFromFloat(@round(scale_y));
    if (@abs(scale_x - @as(f64, @floatFromInt(ix))) < std.math.floatEps(f64) and @abs(scale_y - @as(f64, @floatFromInt(iy))) < std.math.floatEps(f64)) {
        // resizeAreaFast: a block mean. Halving runs the SIMD kernel's (sum + 2) >> 2.
        const area: f32 = @floatFromInt(ix * iy);
        for (0..height) |y| for (0..width) |x| for (0..3) |c| {
            var sum: u32 = 0;
            for (0..iy) |dy| for (0..ix) |dx| {
                sum += src.at(x * ix + dx, y * iy + dy)[c];
            };
            out.at(x, y)[c] = if (ix == 2 and iy == 2) @intCast((sum + 2) >> 2) else @intFromFloat(std.math.clamp(rint(@as(f32, @floatFromInt(sum)) * (1 / area)), 0, 255));
        };
        return out;
    }
    const xs = try areaTaps(gpa, src.width, width, scale_x);
    defer gpa.free(xs);
    const ys = try areaTaps(gpa, src.height, height, scale_y);
    defer gpa.free(ys);
    const buf = try gpa.alloc(f32, width * 3);
    defer gpa.free(buf);
    const sum = try gpa.alloc(f32, width * 3);
    defer gpa.free(sum);
    @memset(sum, 0);
    var prev: usize = 0;
    for (ys) |ty| {
        @memset(buf, 0);
        const line = src.pixels[ty.src * src.width * 3 ..][0 .. src.width * 3];
        for (xs) |tx| for (0..3) |c| {
            buf[tx.dst * 3 + c] = buf[tx.dst * 3 + c] + @as(f32, @floatFromInt(line[tx.src * 3 + c])) * tx.alpha;
        };
        if (ty.dst != prev) {
            flushArea(out, prev, sum);
            for (sum, buf) |*s, b| s.* = ty.alpha * b;
            prev = ty.dst;
        } else for (sum, buf) |*s, b| {
            s.* += ty.alpha * b;
        }
    }
    flushArea(out, prev, sum);
    return out;
}

fn flushArea(out: Image, y: usize, sum: []const f32) void {
    for (out.pixels[y * out.width * 3 ..][0 .. out.width * 3], sum) |*d, s| d.* = @intFromFloat(std.math.clamp(rint(s), 0, 255));
}

/// A 2x3 affine matrix, row-major, mapping source to destination as `cv2.warpAffine` takes it.
pub const Affine = [6]f64;

/// `cv2.getRotationMatrix2D(centre, degrees, scale)`: counter-clockwise for positive degrees.
pub fn rotationMatrix(cx: f64, cy: f64, degrees: f64, scale: f64) Affine {
    const radians = degrees * std.math.pi / 180;
    const alpha = @cos(radians) * scale;
    const beta = @sin(radians) * scale;
    return .{ alpha, beta, (1 - alpha) * cx - beta * cy, -beta, alpha, beta * cx + (1 - alpha) * cy };
}

/// `cv2.warpAffine(src, m, (width, height))` with INTER_LINEAR and a constant black border.
pub fn warpAffine(gpa: Allocator, src: Image, m: Affine, width: usize, height: usize) !Image {
    // Invert, as warpAffine does without WARP_INVERSE_MAP.
    var inv = m;
    var det = inv[0] * inv[4] - inv[1] * inv[3];
    det = if (det != 0) 1 / det else 0;
    const a11 = inv[4] * det;
    const a22 = inv[0] * det;
    inv[0] = a11;
    inv[1] *= -det;
    inv[3] *= -det;
    inv[4] = a22;
    const b1 = -inv[0] * inv[2] - inv[1] * inv[5];
    const b2 = -inv[3] * inv[2] - inv[4] * inv[5];
    inv[2] = b1;
    inv[5] = b2;

    const ab_scale = 1024.0; // AB_BITS = 10; INTER_BITS = 5
    const round_delta = 16; // AB_SCALE / INTER_TAB_SIZE / 2
    const out = try Image.init(gpa, width, height);
    const sw: i64 = @intCast(src.width);
    const sh: i64 = @intCast(src.height);
    for (0..height) |y| {
        const fy: f64 = @floatFromInt(y);
        const x0 = saturateInt((inv[1] * fy + inv[2]) * ab_scale) +% round_delta;
        const y0 = saturateInt((inv[4] * fy + inv[5]) * ab_scale) +% round_delta;
        for (0..width) |x| {
            const fx: f64 = @floatFromInt(x);
            const xx = (x0 +% saturateInt(inv[0] * fx * ab_scale)) >> 5;
            const yy = (y0 +% saturateInt(inv[3] * fx * ab_scale)) >> 5;
            const sx: i64 = std.math.clamp(xx >> 5, -32768, 32767);
            const sy: i64 = std.math.clamp(yy >> 5, -32768, 32767);
            const ax: i32 = xx & 31;
            const ay: i32 = yy & 31;
            // initInterTab2D's linear weights: exact multiples of 32, summing to 32768.
            const w = [4]i32{ (32 - ay) * (32 - ax) * 32, (32 - ay) * ax * 32, ay * (32 - ax) * 32, ay * ax * 32 };
            const px = [4]i64{ sx, sx + 1, sx, sx + 1 };
            const py = [4]i64{ sy, sy, sy + 1, sy + 1 };
            for (0..3) |c| {
                var acc: i32 = 0;
                for (w, px, py) |wk, xk, yk| {
                    if (xk >= 0 and xk < sw and yk >= 0 and yk < sh) acc += wk * @as(i32, src.at(@intCast(xk), @intCast(yk))[c]);
                }
                out.at(x, y)[c] = @intCast(std.math.clamp((acc + (1 << 14)) >> 15, 0, 255));
            }
        }
    }
    return out;
}

test "PNG round-trips exactly and JPEG closely through stb" {
    const gpa = std.testing.allocator;
    const src = try Image.init(gpa, 16, 8);
    defer src.deinit(gpa);
    for (0..8) |y| for (0..16) |x| {
        src.at(x, y).* = .{ @intCast(x * 16), @intCast(y * 32), 128 };
    };
    const png = try encodePng(gpa, src);
    defer gpa.free(png);
    const back = try decode(gpa, png, .apply_exif);
    defer back.deinit(gpa);
    try std.testing.expectEqualSlices(u8, src.pixels, back.pixels);
    const jpg = try encodeJpeg(gpa, src, 95);
    defer gpa.free(jpg);
    const lossy = try decode(gpa, jpg, .as_stored);
    defer lossy.deinit(gpa);
    try std.testing.expectEqual(@as(usize, 16), lossy.width);
    for (src.pixels, lossy.pixels) |a, b| try std.testing.expect(@abs(@as(i32, a) - b) <= 12);
    try std.testing.expectError(error.UndecodableImage, decode(gpa, "not an image", .apply_exif));
}

test "cvRound ties to even" {
    try std.testing.expectEqual(@as(f64, 2), rint(2.5));
    try std.testing.expectEqual(@as(f64, 4), rint(3.5));
    try std.testing.expectEqual(@as(f64, -2), rint(-2.5));
    try std.testing.expectEqual(@as(f64, 3), rint(2.6));
}

test "linear resize keeps flat colour, interpolates edges, and halving is a rounded mean" {
    const gpa = std.testing.allocator;
    const src = try Image.init(gpa, 4, 2);
    defer src.deinit(gpa);
    for (0..2) |y| for (0..4) |x| {
        src.at(x, y).* = .{ @intCast(x * 60), 100, @intCast(y * 3) };
    };
    const up = try resizeLinear(gpa, src, 8, 4);
    defer up.deinit(gpa);
    try std.testing.expectEqual(@as(u8, 100), up.at(5, 2)[1]);
    try std.testing.expectEqual(@as(u8, 0), up.at(0, 0)[0]); // clamped at the left edge
    try std.testing.expectEqual(@as(u8, 180), up.at(7, 0)[0]); // and the right
    try std.testing.expectEqual(@as(u8, 15), up.at(1, 0)[0]); // a quarter of the way to 60
    const half = try resizeLinear(gpa, src, 2, 1);
    defer half.deinit(gpa);
    try std.testing.expectEqualSlices(u8, &.{ 30, 100, 2, 150, 100, 2 }, half.pixels); // (0+0+3+3+2)>>2 = 2
}

test "area resize averages covered cells by overlap" {
    const gpa = std.testing.allocator;
    const src = try Image.init(gpa, 3, 1);
    defer src.deinit(gpa);
    src.at(0, 0).* = .{ 0, 0, 0 };
    src.at(1, 0).* = .{ 90, 90, 90 };
    src.at(2, 0).* = .{ 180, 180, 180 };
    const out = try resizeArea(gpa, src, 2, 1);
    defer out.deinit(gpa);
    // Cell 0 covers source 0 fully and half of 1: (0 + 45) / 1.5 = 30.
    try std.testing.expectEqual(@as(u8, 30), out.at(0, 0)[0]);
    try std.testing.expectEqual(@as(u8, 150), out.at(1, 0)[0]);
}

test "warpAffine with the identity copies, and a shift fills with black" {
    const gpa = std.testing.allocator;
    const src = try Image.init(gpa, 3, 3);
    defer src.deinit(gpa);
    for (src.pixels, 0..) |*p, i| p.* = @intCast(i * 9);
    const same = try warpAffine(gpa, src, .{ 1, 0, 0, 0, 1, 0 }, 3, 3);
    defer same.deinit(gpa);
    try std.testing.expectEqualSlices(u8, src.pixels, same.pixels);
    const shifted = try warpAffine(gpa, src, .{ 1, 0, 1, 0, 1, 0 }, 3, 3);
    defer shifted.deinit(gpa);
    try std.testing.expectEqualSlices(u8, &.{ 0, 0, 0 }, shifted.at(0, 0));
    try std.testing.expectEqualSlices(u8, src.at(0, 1), shifted.at(1, 1));
}

test "the rotation matrix levels the eye line it was measured from" {
    // A 25-degree line, rotated by its own measured angle, comes out level (not 50 degrees).
    const m = rotationMatrix(80, 80, 25, 1);
    const r = std.math.degreesToRadians(25.0);
    const a = [2]f64{ 80 - 40 * @cos(r), 80 - 40 * @sin(r) };
    const b = [2]f64{ 80 + 40 * @cos(r), 80 + 40 * @sin(r) };
    const ay = m[3] * a[0] + m[4] * a[1] + m[5];
    const by = m[3] * b[0] + m[4] * b[1] + m[5];
    try std.testing.expectApproxEqAbs(ay, by, 1e-9);
}

test "EXIF orientation: tag read from either byte order, and pixels turned as OpenCV does" {
    const gpa = std.testing.allocator;
    // SOI, APP1 "Exif" with a big-endian TIFF holding one entry: orientation 6.
    const jpeg = [_]u8{ 0xFF, 0xD8, 0xFF, 0xE1, 0x00, 0x22 } ++ "Exif\x00\x00".* ++ [_]u8{ 'M', 'M', 0, 42, 0, 0, 0, 8, 0, 1, 0x01, 0x12, 0, 3, 0, 0, 0, 1, 0, 6, 0, 0, 0, 0, 0, 0, 0, 0 } ++ [_]u8{ 0xFF, 0xD9 };
    try std.testing.expectEqual(@as(u16, 6), exifOrientation(&jpeg));
    try std.testing.expectEqual(@as(u16, 1), exifOrientation(&.{ 0xFF, 0xD8, 0xFF, 0xD9 }));
    // 2x1 image [A B]; orientation 6 (rotate 90 clockwise) gives a 1x2 column [A; B].
    const src = try Image.init(gpa, 2, 1);
    defer src.deinit(gpa);
    src.at(0, 0).* = .{ 1, 1, 1 };
    src.at(1, 0).* = .{ 2, 2, 2 };
    const turned = try orient(gpa, src, 6);
    defer turned.deinit(gpa);
    try std.testing.expectEqual(@as(usize, 1), turned.width);
    try std.testing.expectEqualSlices(u8, &.{ 1, 1, 1, 2, 2, 2 }, turned.pixels);
}
