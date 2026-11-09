//! SCRFD's training data pipeline, as the recipe configures MMDetection's: the labelv2 file
//! (`RetinaFaceDataset`), then `RandomSquareCrop` (crop choices 0.3-2.0 of the short side, grey
//! padding), `Resize` to 640x640 without keeping the aspect, `RandomFlip` (0.5, swapping eye
//! identity), `PhotoMetricDistortion` (brightness 32, contrast and saturation 0.5-1.5, hue 18)
//! and `Normalize` ((x - 127.5) / 128, RGB). Images are float32 and never clipped between
//! steps, as MMDetection's are. Each random step is split into drawing its parameters and
//! applying them, so the application can be checked against MMDetection with the same draws.
const std = @import("std");
const img = @import("image.zig");
const t = @import("scrfd_targets.zig");
const Allocator = std.mem.Allocator;

pub const crop_choice = [_]f64{ 0.3, 0.45, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0 };

/// An RGB float32 HWC image, values nominally 0-255.
pub const Image = struct { width: usize, height: usize, pixels: []f32 };

pub const Annotation = struct { faces: []t.Face, ignores: []t.Box };

pub const Entry = struct { name: []const u8, width: usize, height: usize, faces: []t.Face, ignores: []t.Box };

/// `RetinaFaceDataset`: "# name w h" headers; 4 box values plus 15 keypoint values (x, y, flag;
/// a point of all -1 gets weight 0, any other weight 1) for a face; 4 plus a flag of 1 for an
/// ignore region. Images with no lines at all are dropped, as the dataset drops them.
pub fn parseLabels(gpa: Allocator, text: []const u8) ![]Entry {
    var entries: std.ArrayList(Entry) = .empty;
    var faces: std.ArrayList(t.Face) = .empty;
    var ignores: std.ArrayList(t.Box) = .empty;
    var current: ?Entry = null;
    var lines = std.mem.splitScalar(u8, text, '\n');
    while (true) {
        const raw = lines.next();
        const line = std.mem.trim(u8, raw orelse "", " \t\r");
        if (raw == null or std.mem.startsWith(u8, line, "#")) {
            if (current) |*e| if (faces.items.len + ignores.items.len > 0) {
                e.faces = try faces.toOwnedSlice(gpa);
                e.ignores = try ignores.toOwnedSlice(gpa);
                try entries.append(gpa, e.*);
            };
            faces = .empty;
            ignores = .empty;
            if (raw == null) break;
            var words = std.mem.tokenizeAny(u8, line[1..], " \t");
            const name = words.next() orelse return error.BadLabels;
            current = .{ .name = name, .width = try std.fmt.parseInt(usize, words.next() orelse return error.BadLabels, 10), .height = try std.fmt.parseInt(usize, words.next() orelse return error.BadLabels, 10), .faces = &.{}, .ignores = &.{} };
            continue;
        }
        if (line.len == 0) continue;
        if (current == null) return error.BadLabels;
        var values: [19]f32 = undefined;
        var n: usize = 0;
        var words = std.mem.tokenizeAny(u8, line, " \t");
        while (words.next()) |w| : (n += 1) {
            if (n == values.len) return error.BadLabels;
            values[n] = try std.fmt.parseFloat(f32, w);
        }
        const box: t.Box = values[0..4].*;
        if (n == 5) {
            if (values[4] == 1) try ignores.append(gpa, box) else try faces.append(gpa, .{ .box = box, .kps = @splat(.{ 0, 0, 0 }) });
        } else if (n == 19) {
            var face: t.Face = .{ .box = box, .kps = undefined };
            for (0..5) |k| {
                const p = values[4 + k * 3 ..][0..3];
                face.kps[k] = .{ p[0], p[1], if (p[0] == -1 and p[1] == -1 and p[2] == -1) 0 else 1 };
            }
            try faces.append(gpa, face);
        } else if (n == 4) {
            try faces.append(gpa, .{ .box = box, .kps = @splat(.{ 0, 0, 0 }) });
        } else return error.BadLabels;
    }
    return entries.toOwnedSlice(gpa);
}

/// Face-free images for `train_scrfd --negatives`: "# name w h" headers and nothing else. They
/// come in through their own file because `parseLabels` drops empty images, as MMDetection's
/// dataset does, and its recipes must stay as they are. A face line here is an error rather than
/// a positive slipping in by mistake.
pub fn parseNegatives(gpa: Allocator, text: []const u8) ![]Entry {
    var entries: std.ArrayList(Entry) = .empty;
    var lines = std.mem.splitScalar(u8, text, '\n');
    while (lines.next()) |raw| {
        const line = std.mem.trim(u8, raw, " \t\r");
        if (line.len == 0) continue;
        if (!std.mem.startsWith(u8, line, "#")) return error.NegativeHasFaces;
        var words = std.mem.tokenizeAny(u8, line[1..], " \t");
        const name = words.next() orelse return error.BadLabels;
        const width = try std.fmt.parseInt(usize, words.next() orelse return error.BadLabels, 10);
        const height = try std.fmt.parseInt(usize, words.next() orelse return error.BadLabels, 10);
        try entries.append(gpa, .{ .name = name, .width = width, .height = height, .faces = &.{}, .ignores = &.{} });
    }
    return entries.toOwnedSlice(gpa);
}

pub fn fromRgb8(gpa: Allocator, src: img.Image) !Image {
    const pixels = try gpa.alloc(f32, src.pixels.len);
    for (pixels, src.pixels) |*d, s| d.* = @floatFromInt(s);
    return .{ .width = src.width, .height = src.height, .pixels = pixels };
}

/// A square patch in source pixels; left/top may be negative (grey padding).
pub const Patch = struct { left: i64, top: i64, side: i64 };

fn centreInside(b: t.Box, p: Patch) bool {
    const c = t.centre(b);
    return c[0] > @as(f32, @floatFromInt(p.left)) and c[1] > @as(f32, @floatFromInt(p.top)) and
        c[0] < @as(f32, @floatFromInt(p.left + p.side)) and c[1] < @as(f32, @floatFromInt(p.top + p.side));
}

/// numpy's randint(low, high): [low, high).
fn randint(r: std.Random, low: i64, high: i64) i64 {
    return low + @as(i64, @intCast(r.uintLessThan(u64, @intCast(high - low))));
}

/// `RandomSquareCrop`'s draws: a crop choice times the short side, then a position, retried
/// until some face's centre lies inside. (With no faces MMDetection would loop forever; the
/// first draw is taken.)
pub fn drawPatch(r: std.Random, width: usize, height: usize, faces: []const t.Face) Patch {
    const w: i64 = @intCast(width);
    const h: i64 = @intCast(height);
    while (true) {
        const scale = crop_choice[r.uintLessThan(usize, crop_choice.len)];
        for (0..250) |_| {
            const side: i64 = @intFromFloat(@trunc(scale * @as(f64, @floatFromInt(@min(w, h)))));
            const left = if (w == side) 0 else if (w > side) randint(r, 0, w - side) else randint(r, w - side, 0);
            const top = if (h == side) 0 else if (h > side) randint(r, 0, h - side) else randint(r, h - side, 0);
            const p: Patch = .{ .left = left, .top = top, .side = side };
            if (faces.len == 0) return p;
            for (faces) |f| if (centreInside(f.box, p)) return p;
        }
    }
}

fn clipShift(b: t.Box, p: Patch) t.Box {
    const l: f32 = @floatFromInt(p.left);
    const tp: f32 = @floatFromInt(p.top);
    const r: f32 = @floatFromInt(p.left + p.side);
    const bt: f32 = @floatFromInt(p.top + p.side);
    return .{ @max(b[0], l) - l, @max(b[1], tp) - tp, @min(b[2], r) - l, @min(b[3], bt) - tp };
}

/// Apply a patch: keep boxes whose centre is inside, clip them to it, move to its origin; copy
/// the image into a grey (128) square.
pub fn crop(gpa: Allocator, src: Image, ann: Annotation, p: Patch) !struct { Image, Annotation } {
    const side: usize = @intCast(p.side);
    const pixels = try gpa.alloc(f32, side * side * 3);
    @memset(pixels, 128);
    const w: i64 = @intCast(src.width);
    const h: i64 = @intCast(src.height);
    const x0 = @max(0, p.left);
    const y0 = @max(0, p.top);
    const x1 = @min(w, p.left + p.side);
    const y1 = @min(h, p.top + p.side);
    if (x1 > x0) {
        var y = y0;
        while (y < y1) : (y += 1) {
            const from = src.pixels[@intCast((y * w + x0) * 3)..][0..@intCast((x1 - x0) * 3)];
            @memcpy(pixels[@intCast(((y - p.top) * p.side + (x0 - p.left)) * 3)..][0..from.len], from);
        }
    }
    var faces: std.ArrayList(t.Face) = .empty;
    for (ann.faces) |f| if (centreInside(f.box, p)) {
        var out: t.Face = .{ .box = clipShift(f.box, p), .kps = f.kps };
        for (&out.kps) |*k| {
            k[0] = @max(@min(k[0], @as(f32, @floatFromInt(p.left + p.side))), @as(f32, @floatFromInt(p.left))) - @as(f32, @floatFromInt(p.left));
            k[1] = @max(@min(k[1], @as(f32, @floatFromInt(p.top + p.side))), @as(f32, @floatFromInt(p.top))) - @as(f32, @floatFromInt(p.top));
        }
        try faces.append(gpa, out);
    };
    var ignores: std.ArrayList(t.Box) = .empty;
    for (ann.ignores) |b| if (centreInside(b, p)) try ignores.append(gpa, clipShift(b, p));
    return .{ .{ .width = side, .height = side, .pixels = pixels }, .{ .faces = try faces.toOwnedSlice(gpa), .ignores = try ignores.toOwnedSlice(gpa) } };
}

/// `Resize((640, 640), keep_ratio=False)`: bilinear, boxes and points scaled by float32
/// factors and clipped to the image.
pub fn resize(gpa: Allocator, src: Image, ann: Annotation) !struct { Image, Annotation } {
    const out = try img.resizeLinearF32(gpa, src.pixels, src.width, src.height, t.size, t.size);
    const sx: f32 = @floatCast(@as(f64, t.size) / @as(f64, @floatFromInt(src.width)));
    const sy: f32 = @floatCast(@as(f64, t.size) / @as(f64, @floatFromInt(src.height)));
    const limit: f32 = t.size;
    const faces = try gpa.dupe(t.Face, ann.faces);
    for (faces) |*f| {
        f.box = .{ std.math.clamp(f.box[0] * sx, 0, limit), std.math.clamp(f.box[1] * sy, 0, limit), std.math.clamp(f.box[2] * sx, 0, limit), std.math.clamp(f.box[3] * sy, 0, limit) };
        for (&f.kps) |*k| k.* = .{ std.math.clamp(k[0] * sx, 0, limit), std.math.clamp(k[1] * sy, 0, limit), k[2] };
    }
    const ignores = try gpa.dupe(t.Box, ann.ignores);
    for (ignores) |*b| b.* = .{ std.math.clamp(b[0] * sx, 0, limit), std.math.clamp(b[1] * sy, 0, limit), std.math.clamp(b[2] * sx, 0, limit), std.math.clamp(b[3] * sy, 0, limit) };
    return .{ .{ .width = t.size, .height = t.size, .pixels = out }, .{ .faces = faces, .ignores = ignores } };
}

/// `RandomFlip` horizontally, in place: mirrored pixels and boxes, keypoints re-ordered
/// [1, 0, 2, 4, 3] so the subject's left eye stays "left".
pub fn flip(im: Image, ann: Annotation) void {
    const w: f32 = @floatFromInt(im.width);
    for (0..im.height) |y| std.mem.reverse([3]f32, std.mem.bytesAsSlice([3]f32, std.mem.sliceAsBytes(im.pixels[y * im.width * 3 ..][0 .. im.width * 3])));
    for (ann.faces) |*f| {
        const b = f.box; // a literal assigned in place would read b[0] after writing it
        f.box = .{ w - b[2], b[1], w - b[0], b[3] };
        const old = f.kps;
        for ([_]usize{ 1, 0, 2, 4, 3 }, &f.kps) |from, *k| k.* = .{ w - old[from][0], old[from][1], old[from][2] };
    }
    for (ann.ignores) |*b| {
        const o = b.*;
        b.* = .{ w - o[2], o[1], w - o[0], o[3] };
    }
}

/// `PhotoMetricDistortion`'s draws; null where its coin came up "skip".
pub const Photo = struct {
    brightness: ?f32 = null,
    contrast_first: bool = false,
    contrast: ?f32 = null,
    saturation: ?f32 = null,
    hue: ?f32 = null,
    /// The channel order after a swap, as indices into BGR (numpy's permutation(3)).
    swap: ?[3]u8 = null,
};

fn uniform(r: std.Random, lo: f64, hi: f64) f32 {
    return @floatCast(lo + (hi - lo) * r.float(f64));
}

pub fn drawPhoto(r: std.Random) Photo {
    var p: Photo = .{};
    if (r.boolean()) p.brightness = uniform(r, -32, 32);
    p.contrast_first = r.boolean(); // mode == 1
    if (p.contrast_first and r.boolean()) p.contrast = uniform(r, 0.5, 1.5);
    if (r.boolean()) p.saturation = uniform(r, 0.5, 1.5);
    if (r.boolean()) p.hue = uniform(r, -18, 18);
    if (!p.contrast_first and r.boolean()) p.contrast = uniform(r, 0.5, 1.5);
    if (r.boolean()) {
        var order = [3]u8{ 0, 1, 2 };
        r.shuffle(u8, &order);
        p.swap = order;
    }
    return p;
}

/// OpenCV's float BGR2HSV: H in degrees, S in [0, 1] for in-range input, V the maximum.
fn toHsv(b: f32, g: f32, r: f32) [3]f32 {
    const v = @max(@max(r, g), b);
    const lo = @min(@min(r, g), b);
    var diff = v - lo;
    const s = diff / (@abs(v) + std.math.floatEps(f32));
    diff = 60 / (diff + std.math.floatEps(f32));
    var h = if (v == r) (g - b) * diff else if (v == g) (b - r) * diff + 120 else (r - g) * diff + 240;
    if (h < 0) h += 360;
    return .{ h, s, v };
}

/// OpenCV's float HSV2BGR.
fn toBgr(hsv: [3]f32) [3]f32 {
    var h = hsv[0];
    const s = hsv[1];
    const v = hsv[2];
    if (s == 0) return .{ v, v, v };
    h *= 6.0 / 360.0;
    while (h < 0) h += 6;
    while (h >= 6) h -= 6;
    var sector: usize = @intFromFloat(@floor(h));
    h -= @floatFromInt(sector);
    if (sector > 5) {
        sector = 0;
        h = 0;
    }
    const tab = [4]f32{ v, v * (1 - s), v * (1 - s * h), v * (1 - s * (1 - h)) };
    const order = [6][3]u8{ .{ 1, 3, 0 }, .{ 1, 0, 2 }, .{ 3, 0, 1 }, .{ 0, 2, 1 }, .{ 0, 1, 3 }, .{ 2, 1, 0 } };
    return .{ tab[order[sector][0]], tab[order[sector][1]], tab[order[sector][2]] };
}

/// Apply the draws in MMDetection's order, on RGB storage (the HSV round trip treats it as the
/// BGR it was in OpenCV).
pub fn photometric(im: Image, p: Photo) void {
    const px = std.mem.bytesAsSlice([3]f32, std.mem.sliceAsBytes(im.pixels));
    for (px) |*rgb| {
        var bgr = [3]f32{ rgb[2], rgb[1], rgb[0] };
        if (p.brightness) |d| for (&bgr) |*c| {
            c.* += d;
        };
        if (p.contrast_first) if (p.contrast) |a| for (&bgr) |*c| {
            c.* *= a;
        };
        var hsv = toHsv(bgr[0], bgr[1], bgr[2]);
        if (p.saturation) |s| hsv[1] *= s;
        if (p.hue) |dh| {
            hsv[0] += dh;
            if (hsv[0] > 360) hsv[0] -= 360;
            if (hsv[0] < 0) hsv[0] += 360;
        }
        bgr = toBgr(hsv);
        if (!p.contrast_first) if (p.contrast) |a| for (&bgr) |*c| {
            c.* *= a;
        };
        if (p.swap) |order| {
            const old = bgr; // not in place: element 0 would be overwritten before it is read
            bgr = .{ old[order[0]], old[order[1]], old[order[2]] };
        }
        rgb.* = .{ bgr[2], bgr[1], bgr[0] };
    }
}

/// `Normalize` into the network's CHW input.
pub fn normalise(im: Image, out: []f32) void {
    const plane = im.width * im.height;
    for (0..plane) |p| for (0..3) |c| {
        out[c * plane + p] = (im.pixels[p * 3 + c] - 127.5) / 128.0;
    };
}

pub const Draws = struct { patch: Patch, flip: bool, photo: Photo };

/// Draws for an image as decoded: the crop is sized from its pixels, as MMDetection sizes it
/// from `img.shape`, not from the label header.
pub fn draw(r: std.Random, im: Image, faces: []const t.Face) Draws {
    return .{ .patch = drawPatch(r, im.width, im.height, faces), .flip = r.float(f64) < 0.5, .photo = drawPhoto(r) };
}

/// The whole pipeline after decoding: crop, resize, flip, photometric. Returns the 640x640
/// float image (not yet normalised) and the annotation in its coordinates.
pub fn apply(gpa: Allocator, src: Image, ann: Annotation, d: Draws) !struct { Image, Annotation } {
    const cropped, const a1 = try crop(gpa, src, ann, d.patch);
    defer gpa.free(cropped.pixels);
    const sized, const a2 = try resize(gpa, cropped, a1);
    if (d.flip) flip(sized, a2);
    photometric(sized, d.photo);
    return .{ sized, a2 };
}

test "labelv2 parsing: keypoint weights, ignore regions, empty images dropped" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const text = "# a.jpg 640 480\n10 20 50 70 15 30 0.0 40 30 0.0 -1 -1 -1 -1 -1 -1 -1 -1 -1\n100 100 200 200 1\n# empty.jpg 10 10\n# b.jpg 5 5\n1 1 3 3 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1\n";
    const entries = try parseLabels(arena.allocator(), text);
    try std.testing.expectEqual(@as(usize, 2), entries.len);
    try std.testing.expectEqual(@as(usize, 1), entries[0].ignores.len);
    try std.testing.expectEqual(@as(f32, 1), entries[0].faces[0].kps[1][2]);
    try std.testing.expectEqual(@as(f32, 0), entries[0].faces[0].kps[2][2]);
    try std.testing.expectEqualStrings("b.jpg", entries[1].name);
}

test "negatives keep their empty images and refuse face lines" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const entries = try parseNegatives(arena.allocator(), "# clock.jpg 640 480\r\n\n# dog.jpg 1024 768\n");
    try std.testing.expectEqual(@as(usize, 2), entries.len);
    try std.testing.expectEqualStrings("dog.jpg", entries[1].name);
    try std.testing.expectEqual(@as(usize, 768), entries[1].height);
    try std.testing.expectEqual(@as(usize, 0), entries[0].faces.len);
    try std.testing.expectError(error.NegativeHasFaces, parseNegatives(arena.allocator(), "# a.jpg 1 1\n1 1 3 3 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1\n"));
}

test "HSV round trip is the identity on in-range colours" {
    for ([_][3]f32{ .{ 10, 200, 30 }, .{ 255, 0, 0 }, .{ 0, 0, 0 }, .{ 128, 128, 128 }, .{ 12, 34, 250 } }) |bgr| {
        const back = toBgr(toHsv(bgr[0], bgr[1], bgr[2]));
        for (bgr, back) |a, b| try std.testing.expectApproxEqAbs(a, b, 1e-3);
    }
}

test "flip swaps eye identity and mirrors coordinates" {
    var pixels = [_]f32{ 1, 1, 1, 2, 2, 2 };
    var faces = [_]t.Face{.{ .box = .{ 0, 0, 1, 1 }, .kps = .{ .{ 0.2, 0, 1 }, .{ 0.8, 0, 1 }, .{ 0.5, 0, 1 }, .{ 0.1, 0, 1 }, .{ 0.9, 0, 1 } } }};
    flip(.{ .width = 2, .height = 1, .pixels = &pixels }, .{ .faces = &faces, .ignores = &.{} });
    try std.testing.expectEqualSlices(f32, &.{ 2, 2, 2, 1, 1, 1 }, &pixels);
    try std.testing.expectEqualSlices(f32, &.{ 1, 0, 2, 1 }, &faces[0].box);
    try std.testing.expectApproxEqAbs(@as(f32, 1.2), faces[0].kps[0][0], 1e-6); // was the right eye at 0.8
    try std.testing.expectApproxEqAbs(@as(f32, 1.1), faces[0].kps[3][0], 1e-6);
}
