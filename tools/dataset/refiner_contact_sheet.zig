//! Aligned crops, YuNet's eye line beside the refiner's, for judging by eye.
//!
//!   zig run -O ReleaseFast -lc tools/dataset/refiner_contact_sheet.zig tools/dataset/vendor/stb.c -- <detections.json> <eye_refiner.pt> <out.html> [--count 40] [--min-box 80] [--ort path]
//!
//! Every number measured so far comes from Open Images photos with clicked eye points. A real
//! corpus has neither, so this compares the two on images that do not belong to the test set at
//! all -- and compares what a user actually sees.
//!
//! It shows *aligned crops*, not dots on faces. A three-degree eye-line error is invisible as two
//! markers and obvious as a tilted portrait, and tilt is the thing `fcs-core::face_cropper`
//! produces. Each row is one face: the crop levelled by YuNet's own landmarks, then the same crop
//! levelled by the refiner's, then the disagreement in degrees. Rows are ordered by that
//! disagreement, largest first, so the faces where the two models differ most come first -- if the
//! refiner is worse anywhere, it shows up at the top rather than buried.
//!
//! Levelling passes the measured tilt to the rotation unnegated: `image.rotationMatrix` is
//! OpenCV's, counter-clockwise for positive angles, whereas `imageproc::rotate_about_center` in
//! the Rust cropper turns clockwise, so Rust negates and this must not. The wrong sign turns a
//! tilt of theta into 2*theta, which still looks like a plausibly rotated face; `image.zig`'s
//! "levels the eye line" test pins it.
const std = @import("std");
const u = @import("util.zig");
const img = @import("image.zig");
const ort = @import("ort.zig");
const ck = @import("checkpoint.zig");
const net = @import("refiner.zig");
const exporter = @import("export_eye_refiner.zig");

const TILE = 160;

/// The crop face_cropper would produce: centred, scaled, rotated to level the eyes.
fn level(gpa: std.mem.Allocator, frame: img.Image, centre: [2]f32, side: f64, tilt_degrees: f64) !img.Image {
    var m = img.rotationMatrix(centre[0], centre[1], tilt_degrees, TILE / side);
    m[2] += TILE / 2.0 - @as(f64, centre[0]);
    m[5] += TILE / 2.0 - @as(f64, centre[1]);
    return img.warpAffine(gpa, frame, m, TILE, TILE);
}

fn dataUri(gpa: std.mem.Allocator, tile: img.Image) ![]const u8 {
    const jpeg = img.encodeJpeg(gpa, tile, 88) catch return "";
    const encoder = std.base64.standard.Encoder;
    const out = try gpa.alloc(u8, "data:image/jpeg;base64,".len + encoder.calcSize(jpeg.len));
    @memcpy(out[0..23], "data:image/jpeg;base64,");
    _ = encoder.encode(out[23..], jpeg);
    return out;
}

const Row = struct { name: []const u8, gap: f64, yunet: []const u8, refiner: []const u8, yunet_angle: f64, refiner_angle: f64 };

fn widerGap(_: void, a: Row, b: Row) bool {
    return a.gap > b.gap;
}

pub fn main(init: std.process.Init) !void {
    const gpa = init.arena.allocator();
    const io = init.io;
    const args = try init.minimal.args.toSlice(gpa);
    var positional: std.ArrayList([]const u8) = .empty;
    var runtime = ort.runtimePath(init.environ_map);
    var count: usize = 40;
    var min_box: f64 = 80;
    var i: usize = 1;
    while (i < args.len) : (i += 1) {
        const arg = args[i];
        if (std.mem.startsWith(u8, arg, "--") and i + 1 < args.len) {
            i += 1;
            if (std.mem.eql(u8, arg, "--ort")) runtime = args[i] else if (std.mem.eql(u8, arg, "--count")) count = try std.fmt.parseInt(usize, args[i], 10) else if (std.mem.eql(u8, arg, "--min-box")) min_box = try std.fmt.parseFloat(f64, args[i]) else return error.UnknownOption;
        } else try positional.append(gpa, arg);
    }
    if (positional.items.len != 3) {
        std.debug.print("usage: refiner_contact_sheet <detections.json> <eye_refiner.pt> <out.html> [--count 40] [--min-box 80] [--ort path]\n", .{});
        return error.Usage;
    }
    const state = try ck.load(gpa, io, positional.items[1]);
    const model = try net.Refiner.fromCheckpoint(gpa, state);
    var oracle = try ort.Oracle.open(gpa, runtime, try exporter.buildOnnx(gpa, model, exporter.trainedBy(state)));
    defer oracle.close();

    var rows: std.ArrayList(Row) = .empty;
    for ((try u.readJson(gpa, io, positional.items[0])).array.items) |record| {
        const image = u.str(u.field(record, "image"));
        var arena: std.heap.ArenaAllocator = .init(std.heap.smp_allocator);
        defer arena.deinit();
        const tmp = arena.allocator();
        var frame: ?img.Image = null;
        for (u.field(record, "detections").array.items) |detection| {
            const b = u.field(detection, "bbox").array.items;
            const x, const y, const w, const h = .{ u.num(b[0]), u.num(b[1]), u.num(b[2]), u.num(b[3]) };
            if (w < min_box) continue;
            if (frame == null) frame = img.load(tmp, io, try u.toLocal(tmp, image), .apply_exif) catch break;
            const centre = [2]f32{ @floatCast(x + w / 2), @floatCast(y + h / 2) };
            const side = @max(w, h) * 1.25;
            const predicted = try net.readEyes(tmp, &oracle, frame.?, centre, side);

            const own = u.field(detection, "landmarks").array.items;
            const o0, const o1 = .{ own[0].array.items, own[1].array.items };
            const yunet_angle = std.math.atan2(u.num(o1[1]) - u.num(o0[1]), u.num(o1[0]) - u.num(o0[0])) * (180.0 / std.math.pi);
            const refiner_angle = std.math.atan2(@as(f64, predicted[1][1] - predicted[0][1]), @as(f64, predicted[1][0] - predicted[0][0])) * (180.0 / std.math.pi);
            var gap = @mod(@abs(yunet_angle - refiner_angle), 360);
            gap = @min(gap, 360 - gap);
            const stem = u.stemOf(image);
            try rows.append(gpa, .{
                .name = try gpa.dupe(u8, stem[0..@min(24, stem.len)]),
                .gap = gap,
                .yunet = try dataUri(gpa, try level(tmp, frame.?, centre, side, yunet_angle)),
                .refiner = try dataUri(gpa, try level(tmp, frame.?, centre, side, refiner_angle)),
                .yunet_angle = yunet_angle,
                .refiner_angle = refiner_angle,
            });
        }
    }

    std.sort.block(Row, rows.items, {}, widerGap); // stable, like Python's sort
    const shown = rows.items[0..@min(count, rows.items.len)];
    var out: u.Stdout = undefined;
    out.init(io);
    const o = out.w();
    try o.print("{d} faces at least {s}px wide; showing the {d} where the two disagree most\n", .{ rows.items.len, try u.fixedAlloc(gpa, min_box, 0), shown.len });
    if (rows.items.len > 0) {
        const gaps = try gpa.alloc(f64, rows.items.len);
        for (gaps, rows.items) |*g, r| g.* = r.gap;
        std.mem.sort(f64, gaps, {}, std.sort.asc(f64));
        const p90: usize = @intFromFloat(@trunc(@as(f64, @floatFromInt(gaps.len)) * 0.9));
        try o.print("disagreement median {s} deg, 90th percentile {s} deg, max {s} deg\n", .{ try u.fixedAlloc(gpa, gaps[gaps.len / 2], 2), try u.fixedAlloc(gpa, gaps[p90], 2), try u.fixedAlloc(gpa, gaps[gaps.len - 1], 2) });
    }
    try o.flush();

    var page: std.Io.Writer.Allocating = .init(gpa);
    const w = &page.writer;
    try w.writeAll("<!doctype html><meta charset=utf-8><title>Eye-line alignment: YuNet vs refiner</title>" ++
        "<style>body{font:13px system-ui;margin:24px;background:#111;color:#eee}" ++
        "section{display:flex;flex-wrap:wrap;gap:18px}figure{margin:0}" ++
        ".pair{display:flex;gap:4px}.pair img{width:160px;height:160px;border-radius:4px}" ++
        "figcaption{text-align:center;color:#9ad;font-size:11px;margin-top:4px}" ++
        "h1{font-size:18px}p{color:#aaa;max-width:62em}</style>" ++
        "<h1>Eye-line alignment: YuNet (left) vs refiner (right)</h1>" ++
        "<p>Each pair is the same face levelled by each model's eye points, which is what the " ++
        "cropper produces. Sorted by how much the two disagree, largest first, so if the " ++
        "refiner is worse anywhere it appears at the top. The question is simply which column " ++
        "looks more upright.</p><section>");
    for (shown, 0..) |r, k| {
        if (k > 0) try w.writeAll("\n");
        try w.print("<figure><div class=\"pair\"><img src=\"{s}\" title=\"YuNet {s} deg\">", .{ r.yunet, try u.fixedAlloc(gpa, r.yunet_angle, 1) });
        try w.print("<img src=\"{s}\" title=\"refiner {s} deg\"></div>", .{ r.refiner, try u.fixedAlloc(gpa, r.refiner_angle, 1) });
        try w.print("<figcaption>{s}<br>{s} deg apart</figcaption></figure>", .{ r.name, try u.fixedAlloc(gpa, r.gap, 1) });
    }
    try w.writeAll("</section>");
    try u.writeFile(io, positional.items[2], page.written());
    try o.print("wrote {s}\n", .{positional.items[2]});
    try o.flush();
}

test "a data URI carries the JPEG it was given" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const gpa = arena.allocator();
    const tile = try img.Image.init(gpa, 8, 8);
    const uri = try dataUri(gpa, tile);
    try std.testing.expect(std.mem.startsWith(u8, uri, "data:image/jpeg;base64,/9j/")); // FF D8 FF
}
