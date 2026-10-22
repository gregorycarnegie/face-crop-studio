//! Are YuNet's unmatched detections false positives, or faces Open Images never boxed?
//!
//!   zig run -O ReleaseFast tools/dataset/sample_unmatched.zig -- <data_dir> <detections.json> <out.html> [count]
//!
//! Builds a contact sheet from the detections that matched no ground-truth box. Tiles are cut
//! out of the local JPEGs with CSS, so it needs no image library -- open the HTML locally.
//! Matched detections are shown underneath as a control: if those look like faces and the
//! unmatched ones do too, the "false positives" are mostly missing labels.
const std = @import("std");
const u = @import("util.zig");

const TILE = 170; // px
const PAD = 0.35; // of box size, so the tile shows context around the detection

const Box = [4]f64;
const Det = struct { score: f64, box: Box };
const Hit = struct { stem: []const u8, box: Box, score: f64 };

fn iou(a: Box, b: Box) f64 {
    const ix = @max(0.0, @min(a[2], b[2]) - @max(a[0], b[0]));
    const iy = @max(0.0, @min(a[3], b[3]) - @max(a[1], b[1]));
    const inter = ix * iy;
    const union_ = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter;
    return if (union_ > 0) inter / union_ else 0.0;
}

fn stemOf(path: []const u8) []const u8 {
    const base = path[if (std.mem.findScalarLast(u8, path, '/')) |i| i + 1 else 0..];
    const base2 = base[if (std.mem.findScalarLast(u8, base, '\\')) |i| i + 1 else 0..];
    return base2[0 .. std.mem.findScalarLast(u8, base2, '.') orelse base2.len];
}

fn desc(_: void, a: Det, b: Det) bool {
    return a.score > b.score;
}

fn appendTo(gpa: std.mem.Allocator, map: *std.StringHashMapUnmanaged(std.ArrayList(Box)), key: []const u8, box: Box) !void {
    const e = try map.getOrPut(gpa, key);
    if (!e.found_existing) e.value_ptr.* = .empty;
    try e.value_ptr.append(gpa, box);
}

pub fn main(init: std.process.Init) !void {
    const gpa = init.arena.allocator();
    const io = init.io;
    const args = try init.minimal.args.toSlice(gpa);
    if (args.len < 4) return error.Usage;
    const data = try std.Io.Dir.cwd().realPathFileAlloc(io, args[1], gpa); // absolute: tile backgrounds are file:// URIs
    const out_path = args[3];
    const count: usize = if (args.len > 4) try std.fmt.parseInt(usize, args[4], 10) else 60;
    const images_dir = try std.fs.path.join(gpa, &.{ data, "images" });

    var detected: std.StringHashMapUnmanaged(std.ArrayList(Det)) = .empty;
    var n_detections: usize = 0;
    for ((try u.readJson(gpa, io, args[2])).array.items) |rec| {
        const e = try detected.getOrPut(gpa, stemOf(u.str(u.field(rec, "image"))));
        if (!e.found_existing) e.value_ptr.* = .empty;
        for (u.field(rec, "detections").array.items) |d| {
            const b = u.field(d, "bbox").array.items;
            const x, const y, const w, const h = .{ u.num(b[0]), u.num(b[1]), u.num(b[2]), u.num(b[3]) };
            try e.value_ptr.append(gpa, .{ .score = u.num(u.field(d, "score")), .box = .{ x, y, x + w, y + h } });
            n_detections += 1;
        }
    }

    var truth: std.StringHashMapUnmanaged(std.ArrayList(Box)) = .empty;
    var ignore: std.StringHashMapUnmanaged(std.ArrayList(Box)) = .empty;
    var sizes: std.StringArrayHashMapUnmanaged(?[2]u32) = .empty;
    for ([_][]const u8{ "validation", "test" }) |split| {
        const csv = try u.Csv.load(gpa, io, try std.fmt.allocPrint(gpa, "{s}/faces-{s}-annotations-bbox.csv", .{ data, split }));
        const c_id, const c_x0, const c_x1, const c_y0, const c_y1 = .{ csv.col("ImageID"), csv.col("XMin"), csv.col("XMax"), csv.col("YMin"), csv.col("YMax") };
        const c_group, const c_dep = .{ csv.col("IsGroupOf"), csv.col("IsDepiction") };
        for (csv.rows) |r| {
            const stem = r[c_id];
            const entry = try sizes.getOrPut(gpa, stem);
            if (!entry.found_existing) {
                const path = try std.fmt.allocPrint(gpa, "{s}/{s}.jpg", .{ images_dir, stem });
                entry.value_ptr.* = if (std.Io.Dir.cwd().access(io, path, .{})) |_| u.jpegSizeOfFile(gpa, io, path) else |_| null;
            }
            const size = entry.value_ptr.* orelse continue;
            const iw: f64 = @floatFromInt(size[0]);
            const ih: f64 = @floatFromInt(size[1]);
            const box: Box = .{
                try std.fmt.parseFloat(f64, r[c_x0]) * iw, try std.fmt.parseFloat(f64, r[c_y0]) * ih,
                try std.fmt.parseFloat(f64, r[c_x1]) * iw, try std.fmt.parseFloat(f64, r[c_y1]) * ih,
            };
            try appendTo(gpa, if (std.mem.eql(u8, r[c_group], "1") or std.mem.eql(u8, r[c_dep], "1")) &ignore else &truth, stem, box);
        }
    }

    var unmatched: std.ArrayList(Hit) = .empty;
    var matched: std.ArrayList(Hit) = .empty;
    for (sizes.keys(), sizes.values()) |stem, size| {
        if (size == null) continue;
        var gt: std.ArrayList(Box) = .empty;
        if (truth.get(stem)) |t| try gt.appendSlice(gpa, t.items);
        const dets = if (detected.get(stem)) |d| try gpa.dupe(Det, d.items) else try gpa.alloc(Det, 0);
        std.mem.sort(Det, dets, {}, desc);
        for (dets) |det| {
            var best_iou: f64 = 0.0;
            var best_i: isize = -1;
            for (gt.items, 0..) |g, i| {
                const v = iou(det.box, g);
                if (v >= best_iou) { // Python's max() over (iou, index) pairs: ties go to the later index
                    best_iou = v;
                    best_i = @intCast(i);
                }
            }
            if (best_iou >= 0.5) {
                _ = gt.orderedRemove(@intCast(best_i));
                try matched.append(gpa, .{ .stem = stem, .box = det.box, .score = det.score });
            } else {
                var ignored = false;
                if (ignore.get(stem)) |igs| for (igs.items) |ig| {
                    if (iou(det.box, ig) > 0.3) ignored = true;
                };
                if (!ignored) try unmatched.append(gpa, .{ .stem = stem, .box = det.box, .score = det.score });
            }
        }
    }

    var rng = u.PyRandom.init(0);
    const picked = try rng.sample(gpa, Hit, unmatched.items, @min(count, unmatched.items.len));
    const controls = try rng.sample(gpa, Hit, matched.items, @min(count / 3, matched.items.len));

    var so: u.Stdout = undefined;
    so.init(io);
    try so.w().print("{d} unmatched of {d} detections; showing {d} plus {d} matched controls\n", .{ unmatched.items.len, n_detections, picked.len, controls.len });

    var out: std.Io.Writer.Allocating = .init(gpa);
    const w = &out.writer;
    try w.print("<!doctype html><meta charset=utf-8><title>Unmatched detections</title>" ++
        "<style>body{{font:14px system-ui;margin:24px;background:#111;color:#eee}}" ++
        "section{{display:flex;flex-wrap:wrap;gap:10px;margin:16px 0 32px}}" ++
        "figure{{margin:0}}" ++
        ".t{{width:{d}px;height:{d}px;background-repeat:no-repeat;border-radius:4px}}" ++
        "figcaption{{text-align:center;color:#999;font-size:12px}}" ++
        "h2{{font-weight:600}} p{{color:#aaa;max-width:60em}}</style>" ++
        "<h1>Do these look like faces?</h1><p>{d} detections matched no Open Images " ++
        "box. If most of the first group are real faces, they are missing labels rather than " ++
        "false positives, and the measured recall is pessimistic.</p>" ++
        "<h2>Unmatched ({d} sampled)</h2><section>", .{ TILE, TILE, unmatched.items.len, picked.len });
    try tiles(gpa, w, picked, sizes, images_dir);
    try w.print("</section><h2>Matched, as a control ({d})</h2><section>", .{controls.len});
    try tiles(gpa, w, controls, sizes, images_dir);
    try w.writeAll("</section>");
    try u.writeFile(io, out_path, out.written());
    try so.w().print("wrote {s}\n", .{out_path});
    try so.w().flush();
}

fn tiles(gpa: std.mem.Allocator, w: *std.Io.Writer, items: []const Hit, sizes: std.StringArrayHashMapUnmanaged(?[2]u32), images_dir: []const u8) !void {
    for (items, 0..) |h, n| {
        if (n > 0) try w.writeAll("\n");
        const size = sizes.get(h.stem).?.?;
        const iw: f64 = @floatFromInt(size[0]);
        const ih: f64 = @floatFromInt(size[1]);
        const x0, const y0, const x1, const y1 = h.box;
        const pad = PAD * @max(x1 - x0, y1 - y0);
        const rx = x0 - pad;
        const ry = y0 - pad;
        const side = @max(x1 - x0, y1 - y0) + 2 * pad;
        const scale = TILE / side;
        const url = try u.asUri(gpa, try std.fmt.allocPrint(gpa, "{s}{c}{s}.jpg", .{ images_dir, std.fs.path.sep, h.stem }));
        try w.print("<figure><div class=\"t\" style=\"background-image:url({s});background-size:{s}px {s}px;background-position:{s}px {s}px\"></div><figcaption>{s}</figcaption></figure>", .{
            url,
            try u.fixedAlloc(gpa, iw * scale, 1),
            try u.fixedAlloc(gpa, ih * scale, 1),
            try u.fixedAlloc(gpa, -rx * scale, 1),
            try u.fixedAlloc(gpa, -ry * scale, 1),
            try u.fixedAlloc(gpa, h.score, 2),
        });
    }
}
