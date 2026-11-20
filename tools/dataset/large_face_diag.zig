//! Why are large faces fragile once face-free negatives join training? Inference and data
//! statistics only (HARD_NEGATIVES.md, "Diagnostics"). Four modes:
//!
//!   zig build-exe -O ReleaseFast -lc tools/dataset/large_face_diag.zig
//!   large_face_diag scale <labels.txt> <images_dir> <model.onnx> <out.tsv>
//!   large_face_diag aug   <labelv2_train_all.txt> <labelv2_negatives.txt> <negatives_oi_boxes.csv> <negatives_scrfd.json> <class-descriptions-boxable.csv>
//!   large_face_diag audit <labelv2_train_all.txt> <images_dir> <retinaface_r34.onnx> <count> <out_prefix>
//!   large_face_diag sheet <labels.txt> <images_dir> <count> <out_prefix>
//!
//! * `scale`: every face >= 45% of the image width, presented natively (the gate's letterbox) and
//!   then smaller -- the photo shrunk into a black canvas so the face spans 45/30/20/10% of the
//!   canvas width. Per presentation: the score of the detection matched to it (IoU >= 0.5, as the
//!   gate matches), the best raw score per stride among anchors decoding to IoU >= 0.5 with it, and
//!   how many ATSS positives each stride would get for it in training (`scrfd_targets.assign`).
//! * `aug`: the trainer's own `RandomSquareCrop` draws (`scrfd_augment.drawPatch`, 3 per photo,
//!   seed 0), no pixels: face width as a fraction of the crop for positives; for negatives the
//!   largest Open Images object box and the widest SCRFD-80k hit >= 0.3 (the mining detections) in
//!   the crop, and the ATSS positives that hit would have drawn as a face -- anchors trained as
//!   background on face-like content. Also partly visible real faces whose centre falls outside
//!   the crop: dropped from the targets, so their pixels train as background.
//! * `audit`: RetinaFace >= 0.9 on a seeded sample of training photos; detections overlapping no
//!   face box (IoU < 0.3) and no ignore region (IoU or IoF > 0.3) are candidate unlabelled faces.
//!   Writes `<prefix>.tsv` and contact sheets `<prefix>_N.jpg` to judge them by eye.
//! * `sheet`: a seeded sample of a label file drawn on its photos (faces green, ignore regions
//!   red), for spot-checking labels.
//!
//! Photos load with EXIF applied, as detector_gate loads them.
const std = @import("std");
const Io = std.Io;
const u = @import("util.zig");
const img = @import("image.zig");
const ort = @import("ort.zig");
const bench = @import("detector_bench.zig");
const aug = @import("scrfd_augment.zig");
const t = @import("scrfd_targets.zig");
const detector = @import("scrfd_detect.zig");

const LARGE = 0.45;
const SIZE = 640;
/// 0 = native presentation.
const FRACS = [_]f64{ 0, 0.45, 0.30, 0.20, 0.10 };
const BINS = [_]f64{ 0.1, 0.2, 0.3, 0.45, 0.6, 0.8 }; // upper edges; last bin is >= 0.8
const DRAWS = 3;
const ANCHORS = t.anchors();

fn fail(comptime fmt: []const u8, args: anytype) noreturn {
    std.debug.print(fmt ++ "\n", args);
    std.process.exit(1);
}

fn iou(a: [4]f64, b: [4]f64) f64 {
    const ix = @max(0.0, @min(a[2], b[2]) - @max(a[0], b[0]));
    const iy = @max(0.0, @min(a[3], b[3]) - @max(a[1], b[1]));
    const inter = ix * iy;
    const un = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter;
    return if (un > 0) inter / un else 0;
}

fn iof(a: [4]f64, region: [4]f64) f64 {
    const ix = @max(0.0, @min(a[2], region[2]) - @max(a[0], region[0]));
    const iy = @max(0.0, @min(a[3], region[3]) - @max(a[1], region[1]));
    const area = (a[2] - a[0]) * (a[3] - a[1]);
    return if (area > 0) ix * iy / area else 0;
}

fn bin(x: f64) usize {
    for (BINS, 0..) |edge, k| if (x < edge) return k;
    return BINS.len;
}

fn f32box(b: [4]f64) t.Box {
    return .{ @floatCast(b[0]), @floatCast(b[1]), @floatCast(b[2]), @floatCast(b[3]) };
}

/// ATSS positives per stride for truth `k` among `gts` (640-square coordinates).
fn owners(gpa: std.mem.Allocator, gts: []const t.Box, ignores: []const t.Box, k: usize) ![3]usize {
    const assigned = try t.assign(gpa, &ANCHORS, gts, ignores, 0.5, 9);
    defer gpa.free(assigned);
    var out: [3]usize = .{ 0, 0, 0 };
    for (assigned, 0..) |a, i| if (a == @as(i32, @intCast(k + 1))) {
        out[std.mem.findScalar(usize, &t.strides, t.strideOf(i)).?] += 1;
    };
    return out;
}

// ── scale ─────────────────────────────────────────────────────────────────────

fn scale(gpa: std.mem.Allocator, io: Io, runtime: []const u8, labels_path: []const u8, images_dir: []const u8, model_path: []const u8, out_path: []const u8) !void {
    const labels = try bench.readOiLabels(gpa, io, labels_path);
    var model = try bench.Model.unchecked(gpa, io, runtime, model_path);
    // RetinaFace (3 outputs) as a reference: matched scores only, per-stride columns left 0.
    const kind: bench.Kind = switch (model.outputs.len) {
        9 => .scrfd,
        3 => .retinaface,
        else => fail("{s}: neither SCRFD (9 outputs) nor RetinaFace (3)", .{model_path}),
    };
    if (kind == .retinaface) model.priors = try bench.retinaPriors(gpa);
    var out: Io.Writer.Allocating = .init(gpa);
    const w = &out.writer;
    try w.writeAll("image\tface\tnative\tpresented\tscore\tbest8\tbest16\tbest32\tpos8\tpos16\tpos32\n");
    var done: usize = 0;
    for (labels.images.items) |name| {
        var arena: std.heap.ArenaAllocator = .init(std.heap.smp_allocator);
        defer arena.deinit();
        const tmp = arena.allocator();
        const faces = labels.faces.get(name).?.items;
        const ignore = labels.ignore.get(name).?.items;
        const iw = labels.widths.get(name).?;
        var any = false;
        for (faces) |b| any = any or (b[2] - b[0]) / iw >= LARGE;
        if (!any) continue;
        const image = img.load(tmp, io, try std.fs.path.join(tmp, &.{ images_dir, name }), .apply_exif) catch continue;
        const W: f64 = @floatFromInt(image.width);
        const H: f64 = @floatFromInt(image.height);
        for (faces, 0..) |face, fi| {
            const native = (face[2] - face[0]) / iw;
            if (native < LARGE) continue;
            for (FRACS) |frac| {
                const k = if (frac == 0) 1.0 else native / frac;
                const s = @as(f64, SIZE) / (k * @max(W, H));
                const rw: usize = @intFromFloat(@trunc(W * s));
                const rh: usize = @intFromFloat(@trunc(H * s));
                const ox: usize = @intFromFloat(@trunc((k - 1) * W / 2 * s));
                const oy: usize = @intFromFloat(@trunc((k - 1) * H / 2 * s));
                const resized = try img.resizeLinear(tmp, image, rw, rh);
                const canvas = try img.Image.init(tmp, SIZE, SIZE);
                for (0..rh) |y| @memcpy(canvas.pixels[((y + oy) * SIZE + ox) * 3 ..][0 .. rw * 3], resized.pixels[y * rw * 3 ..][0 .. rw * 3]);
                const map = struct {
                    fn f(b: [4]f64, sc: f64, x0: f64, y0: f64) [4]f64 {
                        return .{ b[0] * sc + x0, b[1] * sc + y0, b[2] * sc + x0, b[3] * sc + y0 };
                    }
                }.f;
                const fx: f64 = @floatFromInt(ox);
                const fy: f64 = @floatFromInt(oy);
                const truths = try tmp.alloc([4]f64, faces.len);
                for (truths, faces) |*o, b| o.* = map(b, s, fx, fy);
                const regions = try tmp.alloc([4]f64, ignore.len);
                for (regions, ignore) |*o, b| o.* = map(b, s, fx, fy);
                // Matched score, as the gate scores it.
                const rows = try bench.detect(tmp, &model, kind, canvas);
                const dets = try tmp.alloc(bench.Det, rows.len);
                for (dets, rows) |*d, r| d.* = .{ .box = .{ r[0], r[1], r[2], r[3] }, .score = r[4], .eyes = .{ r[5], r[6], r[7], r[8] } };
                var unmatched: std.ArrayList(f64) = .empty;
                const slots = try bench.matchOi(tmp, dets, truths, regions, &unmatched);
                const score = if (slots[fi]) |d| d.score else 0;
                // Best raw score per stride among anchors decoding onto the face.
                var best: [3]f64 = .{ 0, 0, 0 };
                const outputs: []const ort.Output = if (kind == .scrfd) try model.oracle.runAll(tmp, try bench.blob(tmp, canvas, .{ 0, 1, 2 }, .{ 127.5, 127.5, 127.5 }, 1.0 / 128.0), &.{ 1, 3, SIZE, SIZE }, model.input, model.outputs) else &.{};
                if (kind == .scrfd) for (t.strides, 0..) |stride, level| {
                    const side = SIZE / stride;
                    const scores = outputs[level].data;
                    const boxes = outputs[level + 3].data;
                    const sf: f64 = @floatFromInt(stride);
                    for (scores, 0..) |sc, j| {
                        if (!(sc > best[level])) continue;
                        const cell = j / 2;
                        const cx: f64 = @floatFromInt((cell % side) * stride);
                        const cy: f64 = @floatFromInt((cell / side) * stride);
                        const bx = boxes[j * 4 ..][0..4];
                        if (iou(.{ cx - bx[0] * sf, cy - bx[1] * sf, cx + bx[2] * sf, cy + bx[3] * sf }, truths[fi]) >= 0.5) best[level] = sc;
                    }
                };
                const gts = try tmp.alloc(t.Box, truths.len);
                for (gts, truths) |*g, b| g.* = f32box(b);
                const igs = try tmp.alloc(t.Box, regions.len);
                for (igs, regions) |*g, b| g.* = f32box(b);
                const pos = try owners(tmp, gts, igs, fi);
                try w.print("{s}\t{d}\t{d:.4}\t{d:.2}\t{d:.4}\t{d:.4}\t{d:.4}\t{d:.4}\t{d}\t{d}\t{d}\n", .{ name, fi, native, if (frac == 0) native else frac, score, best[0], best[1], best[2], pos[0], pos[1], pos[2] });
            }
            done += 1;
        }
    }
    try u.writeFile(io, out_path, out.written());
    std.debug.print("{d} large faces -> {s}\n", .{ done, out_path });
}

// ── aug ───────────────────────────────────────────────────────────────────────

const Hist = struct {
    n: [BINS.len + 1]usize = @splat(0),
    total: usize = 0,
    fn add(h: *Hist, x: f64) void {
        h.n[bin(x)] += 1;
        h.total += 1;
    }
    fn print(h: Hist, w: *Io.Writer, name: []const u8) !void {
        try u.padRight(w, name, 58);
        for (h.n) |c| try w.print(" {d:>7.2}", .{100 * @as(f64, @floatFromInt(c)) / @as(f64, @floatFromInt(@max(h.total, 1)))});
        try w.print("  n={d}\n", .{h.total});
    }
    /// Count at or above BINS[k].
    fn above(h: Hist, k: usize) usize {
        var s: usize = 0;
        for (h.n[k + 1 ..]) |c| s += c;
        return s;
    }
};
const HEADER = "   <0.10 0.10-.2 0.20-.3 0.3-.45 .45-.60 .60-.80   >=.80\n";

fn clipFrac(b: [4]f64, p: aug.Patch) struct { f64, f64 } {
    const l: f64 = @floatFromInt(p.left);
    const tp: f64 = @floatFromInt(p.top);
    const side: f64 = @floatFromInt(p.side);
    const cw = @max(0, @min(b[2], l + side) - @max(b[0], l));
    const ch = @max(0, @min(b[3], tp + side) - @max(b[1], tp));
    return .{ cw / side, cw * ch / ((b[2] - b[0]) * (b[3] - b[1])) }; // width in crop, visible share
}

fn centreIn(b: [4]f64, p: aug.Patch) bool {
    const cx = (b[0] + b[2]) / 2;
    const cy = (b[1] + b[3]) / 2;
    const l: f64 = @floatFromInt(p.left);
    const tp: f64 = @floatFromInt(p.top);
    const side: f64 = @floatFromInt(p.side);
    return cx > l and cy > tp and cx < l + side and cy < tp + side;
}

/// A box clipped to the patch, in the 640 square the crop is resized to.
fn toSquare(b: [4]f64, p: aug.Patch) t.Box {
    const l: f64 = @floatFromInt(p.left);
    const tp: f64 = @floatFromInt(p.top);
    const side: f64 = @floatFromInt(p.side);
    const sc = @as(f64, SIZE) / side;
    return f32box(.{ (@max(b[0], l) - l) * sc, (@max(b[1], tp) - tp) * sc, (@min(b[2], l + side) - l) * sc, (@min(b[3], tp + side) - tp) * sc });
}

fn augStats(gpa: std.mem.Allocator, io: Io, train_path: []const u8, neg_path: []const u8, boxes_path: []const u8, scrfd_path: []const u8, classes_path: []const u8) !void {
    var so: u.Stdout = undefined;
    so.init(io);
    const w = so.w();
    var prng: std.Random.DefaultPrng = .init(0);
    const r = prng.random();

    // Positives.
    const train = try aug.parseLabels(gpa, try u.readFile(gpa, io, train_path));
    var max_face: Hist = .{};
    var all_faces: Hist = .{};
    var native_max: Hist = .{};
    var pos_stride: [3]usize = .{ 0, 0, 0 };
    var hidden_faces: usize = 0; // >= 30% visible, centre outside: trained as background
    var hidden_big: usize = 0; // ... and >= 0.3 of the crop wide
    var crops: usize = 0;
    for (train, 0..) |e, idx| {
        var nm: f64 = 0;
        for (e.faces) |f| nm = @max(nm, (f.box[2] - f.box[0]) / @as(f64, @floatFromInt(e.width)));
        native_max.add(nm);
        for (0..DRAWS) |_| {
            const p = aug.drawPatch(r, e.width, e.height, e.faces);
            crops += 1;
            var m: f64 = 0;
            var gts: std.ArrayList(t.Box) = .empty;
            defer gts.deinit(gpa);
            for (e.faces) |f| {
                const b: [4]f64 = .{ f.box[0], f.box[1], f.box[2], f.box[3] };
                const frac, const vis = clipFrac(b, p);
                if (centreIn(b, p)) {
                    m = @max(m, frac);
                    all_faces.add(frac);
                    try gts.append(gpa, toSquare(b, p));
                } else if (vis >= 0.3) {
                    hidden_faces += 1;
                    hidden_big += @intFromBool(frac >= 0.3);
                }
            }
            max_face.add(m);
            if (idx % 8 == 0 and gts.items.len > 0) { // ATSS on a 1-in-8 subsample, scaled back below
                var igs: std.ArrayList(t.Box) = .empty;
                defer igs.deinit(gpa);
                for (e.ignores) |g| if (centreIn(.{ g[0], g[1], g[2], g[3] }, p)) try igs.append(gpa, toSquare(.{ g[0], g[1], g[2], g[3] }, p));
                const assigned = try t.assign(gpa, &ANCHORS, gts.items, igs.items, 0.5, 9);
                defer gpa.free(assigned);
                for (assigned, 0..) |a, i| if (a > 0) {
                    pos_stride[std.mem.findScalar(usize, &t.strides, t.strideOf(i)).?] += 8;
                };
            }
        }
    }

    // Negatives: OI object boxes and SCRFD-80k mining hits per image.
    const negs = try aug.parseNegatives(gpa, try u.readFile(gpa, io, neg_path));
    var objects: std.StringHashMapUnmanaged(std.ArrayList(struct { box: [4]f64, label: []const u8 })) = .empty;
    {
        const csv = try u.Csv.load(gpa, io, boxes_path);
        for (csv.rows) |row| {
            const e = try objects.getOrPut(gpa, row[0]);
            if (!e.found_existing) e.value_ptr.* = .empty;
            try e.value_ptr.append(gpa, .{ .box = .{ try std.fmt.parseFloat(f64, row[2]), try std.fmt.parseFloat(f64, row[4]), try std.fmt.parseFloat(f64, row[3]), try std.fmt.parseFloat(f64, row[5]) }, .label = row[1] });
        }
    }
    var names: std.StringHashMapUnmanaged([]const u8) = .empty;
    {
        const text = try u.readFile(gpa, io, classes_path);
        var it = std.mem.splitScalar(u8, text, '\n');
        while (it.next()) |line| {
            const l = std.mem.trimEnd(u8, line, "\r");
            const comma = std.mem.findScalar(u8, l, ',') orelse continue;
            try names.put(gpa, l[0..comma], l[comma + 1 ..]);
        }
    }
    var hits: std.StringHashMapUnmanaged([4]f64) = .empty; // widest hit >= 0.3, pixels
    for ((try u.readJson(gpa, io, scrfd_path)).array.items) |rec| {
        var widest: ?[4]f64 = null;
        for (u.field(rec, "detections").array.items) |d| {
            if (u.num(u.field(d, "score")) < 0.3) continue;
            const b = u.field(d, "bbox").array.items;
            const x, const y, const bw, const bh = .{ u.num(b[0]), u.num(b[1]), u.num(b[2]), u.num(b[3]) };
            if (widest == null or bw > widest.?[2] - widest.?[0]) widest = .{ x, y, x + bw, y + bh };
        }
        if (widest) |b| try hits.put(gpa, try gpa.dupe(u8, u.baseName(u.str(u.field(rec, "image")))), b);
    }
    var obj_hist: Hist = .{};
    var hit_hist: Hist = .{};
    var hit_native: Hist = .{};
    var neg_stride: [3]usize = .{ 0, 0, 0 };
    var big_classes: u.Counter = .{};
    var neg_crops: usize = 0;
    var hard: usize = 0;
    for (negs) |e| {
        const W: f64 = @floatFromInt(e.width);
        const H: f64 = @floatFromInt(e.height);
        const hit = hits.get(e.name);
        if (hit) |b| {
            hard += 1;
            hit_native.add((b[2] - b[0]) / W);
        }
        for (0..DRAWS) |_| {
            const p = aug.drawPatch(r, e.width, e.height, &.{});
            neg_crops += 1;
            var best: f64 = 0;
            var best_area: f64 = 0;
            var label: []const u8 = "";
            if (objects.get(u.stemOf(e.name))) |list| for (list.items) |o| {
                const b: [4]f64 = .{ o.box[0] * W, o.box[1] * H, o.box[2] * W, o.box[3] * H };
                const frac, const vis = clipFrac(b, p);
                const area = vis * (b[2] - b[0]) * (b[3] - b[1]);
                if (area > best_area) {
                    best_area = area;
                    best = frac;
                    label = o.label;
                }
            };
            obj_hist.add(best);
            if (best >= LARGE) try big_classes.inc(gpa, names.get(label) orelse label);
            if (hit) |b| {
                const frac, _ = clipFrac(b, p);
                if (centreIn(b, p)) {
                    hit_hist.add(frac);
                    const pos = try owners(gpa, &.{toSquare(b, p)}, &.{}, 0);
                    for (&neg_stride, pos) |*a, c| a.* += c;
                } else hit_hist.add(0);
            }
        }
    }

    try w.print("RandomSquareCrop statistics: {d} draws per photo, seed 0 (scrfd_augment.drawPatch)\n", .{DRAWS});
    try u.padRight(w, "share of crops (%), by width / crop side", 58);
    try w.writeAll(HEADER);
    try native_max.print(w, "positives: widest face / image width, before crop");
    try max_face.print(w, "positives: widest kept face / crop side, per crop");
    try all_faces.print(w, "positives: every kept face / crop side");
    try obj_hist.print(w, "negatives: largest OI object (clipped) / crop side");
    try hit_native.print(w, "hard negatives: widest SCRFD hit / image width");
    try hit_hist.print(w, "hard negatives: widest hit in crop / crop side (0=out)");
    const per_epoch = struct {
        fn f(x: usize) f64 {
            return @as(f64, @floatFromInt(x)) / DRAWS;
        }
    }.f;
    try w.print("\nPer epoch ({d} positive photos, {d} negative photos, {d} with a hit >= 0.3):\n", .{ train.len, negs.len, hard });
    for ([_]Hist{ max_face, obj_hist, hit_hist }, [_][]const u8{ "positive crops with a face", "negative crops with an OI object", "negative crops with a SCRFD hit" }) |h, what|
        try w.print("  {s} >= 0.45 / 0.6 / 0.8 of the crop: {d:.0} / {d:.0} / {d:.0}\n", .{ what, per_epoch(h.above(3)), per_epoch(h.above(4)), per_epoch(h.above(5)) });
    try w.print("  ATSS positives from real faces, stride 8/16/32: {d:.0} / {d:.0} / {d:.0}\n", .{ per_epoch(pos_stride[0]), per_epoch(pos_stride[1]), per_epoch(pos_stride[2]) });
    try w.print("  anchors ATSS would give a SCRFD hit, now background, stride 8/16/32: {d:.0} / {d:.0} / {d:.0}\n", .{ per_epoch(neg_stride[0]), per_epoch(neg_stride[1]), per_epoch(neg_stride[2]) });
    try w.print("  real faces >= 30% visible but centre outside the crop (trained as background): {d:.0} ({d:.0} of them >= 0.3 of the crop wide)\n", .{ per_epoch(hidden_faces), per_epoch(hidden_big) });
    try w.writeAll("\nNegative crops whose largest object is >= 0.45 of the crop, top classes:\n");
    const keys = try gpa.dupe([]const u8, big_classes.map.keys());
    std.mem.sort([]const u8, keys, big_classes, struct {
        fn more(c: u.Counter, a: []const u8, b: []const u8) bool {
            return c.get(a) > c.get(b);
        }
    }.more);
    for (keys[0..@min(keys.len, 15)]) |k| try w.print("  {d:>6}  {s}\n", .{ big_classes.get(k), k });
    try w.print("(positive crops sampled: {d}; negative: {d})\n", .{ crops, neg_crops });
    try w.flush();
}

// ── contact sheets ────────────────────────────────────────────────────────────

fn rect(im: img.Image, b: [4]f64, colour: [3]u8) void {
    const x0: i64 = @intFromFloat(@max(0, @floor(b[0])));
    const y0: i64 = @intFromFloat(@max(0, @floor(b[1])));
    const x1: i64 = @intFromFloat(@min(@as(f64, @floatFromInt(im.width - 1)), @floor(b[2])));
    const y1: i64 = @intFromFloat(@min(@as(f64, @floatFromInt(im.height - 1)), @floor(b[3])));
    if (x1 < x0 or y1 < y0) return;
    for (0..2) |d_| {
        const d: i64 = @intCast(d_);
        var x = x0;
        while (x <= x1) : (x += 1) for ([_]i64{ y0 + d, y1 - d }) |y| if (y >= 0 and y < im.height) {
            im.at(@intCast(x), @intCast(y)).* = colour;
        };
        var y = y0;
        while (y <= y1) : (y += 1) for ([_]i64{ x0 + d, x1 - d }) |xx| if (xx >= 0 and xx < im.width) {
            im.at(@intCast(xx), @intCast(y)).* = colour;
        };
    }
}

/// Tiles of at most `tile` px a side into sheets of `cols` x `rows`, written as `<prefix>_N.jpg`.
fn writeSheets(gpa: std.mem.Allocator, io: Io, tiles: []const img.Image, tile: usize, cols: usize, rows: usize, prefix: []const u8) !void {
    var start: usize = 0;
    var n: usize = 1;
    while (start < tiles.len) : (n += 1) {
        const end = @min(tiles.len, start + cols * rows);
        const page = try img.Image.init(gpa, cols * tile, ((end - start + cols - 1) / cols) * tile);
        @memset(page.pixels, 40);
        for (tiles[start..end], 0..) |tl, k| {
            const ox = (k % cols) * tile;
            const oy = (k / cols) * tile;
            for (0..tl.height) |y| @memcpy(page.pixels[((oy + y) * page.width + ox) * 3 ..][0 .. tl.width * 3], tl.pixels[y * tl.width * 3 ..][0 .. tl.width * 3]);
        }
        try u.writeFile(io, try std.fmt.allocPrint(gpa, "{s}_{d}.jpg", .{ prefix, n }), try img.encodeJpeg(gpa, page, 85));
        start = end;
    }
}

/// `k` distinct indices below `n`, seeded.
fn sample(gpa: std.mem.Allocator, n: usize, k: usize, seed: u64) ![]usize {
    const idx = try gpa.alloc(usize, n);
    for (idx, 0..) |*x, i| x.* = i;
    var prng: std.Random.DefaultPrng = .init(seed);
    prng.random().shuffle(usize, idx);
    const out = idx[0..@min(k, n)];
    std.mem.sort(usize, out, {}, std.sort.asc(usize));
    return out;
}

fn sheet(gpa: std.mem.Allocator, io: Io, labels_path: []const u8, images_dir: []const u8, count: usize, prefix: []const u8) !void {
    const TILE = 320;
    const labels = try bench.readOiLabels(gpa, io, labels_path);
    var tiles: std.ArrayList(img.Image) = .empty;
    var listing: Io.Writer.Allocating = .init(gpa);
    for (try sample(gpa, labels.images.items.len, count, 7)) |i| {
        const name = labels.images.items[i];
        const image = try img.load(gpa, io, try std.fs.path.join(gpa, &.{ images_dir, name }), .apply_exif);
        const s = @as(f64, TILE) / @as(f64, @floatFromInt(@max(image.width, image.height)));
        const small = try img.resizeArea(gpa, image, @max(1, @as(usize, @intFromFloat(@trunc(@as(f64, @floatFromInt(image.width)) * s)))), @max(1, @as(usize, @intFromFloat(@trunc(@as(f64, @floatFromInt(image.height)) * s)))));
        image.deinit(gpa);
        for (labels.ignore.get(name).?.items) |b| rect(small, .{ b[0] * s, b[1] * s, b[2] * s, b[3] * s }, .{ 255, 40, 40 });
        for (labels.faces.get(name).?.items) |b| rect(small, .{ b[0] * s, b[1] * s, b[2] * s, b[3] * s }, .{ 40, 255, 40 });
        try tiles.append(gpa, small);
        try listing.writer.print("{d}\t{s}\t{d}\t{d}\n", .{ tiles.items.len, name, labels.faces.get(name).?.items.len, labels.ignore.get(name).?.items.len });
    }
    try writeSheets(gpa, io, tiles.items, TILE, 5, 6, prefix);
    try u.writeFile(io, try std.fmt.allocPrint(gpa, "{s}.tsv", .{prefix}), listing.written());
    std.debug.print("{d} photos -> {s}_N.jpg (5 x 6 per sheet, row-major; list in {s}.tsv)\n", .{ tiles.items.len, prefix, prefix });
}

// ── audit ─────────────────────────────────────────────────────────────────────

fn audit(gpa: std.mem.Allocator, io: Io, runtime: []const u8, labels_path: []const u8, images_dir: []const u8, model_path: []const u8, count: usize, prefix: []const u8) !void {
    const TILE = 128;
    const labels = try bench.readOiLabels(gpa, io, labels_path);
    var model = try bench.Model.unchecked(gpa, io, runtime, model_path);
    if (model.outputs.len != 3) fail("{s}: not RetinaFace (3 outputs)", .{model_path});
    model.priors = try bench.retinaPriors(gpa);
    var tsv: Io.Writer.Allocating = .init(gpa);
    try tsv.writer.writeAll("image\tscore\tx1\ty1\tx2\ty2\twidth_frac\n");
    var tiles: std.ArrayList(img.Image) = .empty;
    var n_img: usize = 0;
    var n_conf: usize = 0;
    var n_unl: usize = 0;
    var img_unl: usize = 0;
    var n_faces: usize = 0;
    var size_hist: Hist = .{};
    for (try sample(gpa, labels.images.items.len, count, 1)) |i| {
        var arena: std.heap.ArenaAllocator = .init(std.heap.smp_allocator);
        defer arena.deinit();
        const tmp = arena.allocator();
        const name = labels.images.items[i];
        const image = img.load(tmp, io, try std.fs.path.join(tmp, &.{ images_dir, name }), .apply_exif) catch continue;
        n_img += 1;
        const faces = labels.faces.get(name).?.items;
        const ignore = labels.ignore.get(name).?.items;
        n_faces += faces.len;
        var found = false;
        for (try bench.detect(tmp, &model, .retinaface, image)) |r| {
            if (r[4] < 0.9) continue;
            n_conf += 1;
            const b: [4]f64 = .{ r[0], r[1], r[2], r[3] };
            var labelled = false;
            for (faces) |f| labelled = labelled or iou(b, f) >= 0.3;
            for (ignore) |g| labelled = labelled or iou(b, g) > 0.3 or iof(b, g) > 0.3;
            if (labelled) continue;
            n_unl += 1;
            found = true;
            const frac = (b[2] - b[0]) / @as(f64, @floatFromInt(image.width));
            size_hist.add(frac);
            try tsv.writer.print("{s}\t{d:.3}\t{d:.0}\t{d:.0}\t{d:.0}\t{d:.0}\t{d:.3}\n", .{ name, r[4], b[0], b[1], b[2], b[3], frac });
            // Tile: the box grown 1.6x, square, resized to TILE.
            const cx = (b[0] + b[2]) / 2;
            const cy = (b[1] + b[3]) / 2;
            const half = 0.8 * @max(b[2] - b[0], b[3] - b[1]);
            const x0: usize = @intFromFloat(@max(0, cx - half));
            const y0: usize = @intFromFloat(@max(0, cy - half));
            const x1: usize = @intFromFloat(@min(@as(f64, @floatFromInt(image.width)), cx + half));
            const y1: usize = @intFromFloat(@min(@as(f64, @floatFromInt(image.height)), cy + half));
            if (x1 <= x0 + 1 or y1 <= y0 + 1) continue;
            const crop = try img.Image.init(tmp, x1 - x0, y1 - y0);
            for (y0..y1) |y| @memcpy(crop.pixels[(y - y0) * crop.width * 3 ..][0 .. crop.width * 3], image.pixels[(y * image.width + x0) * 3 ..][0 .. crop.width * 3]);
            const s = @as(f64, TILE) / @as(f64, @floatFromInt(@max(crop.width, crop.height)));
            const tw = @max(1, @as(usize, @intFromFloat(@trunc(@as(f64, @floatFromInt(crop.width)) * s))));
            const th = @max(1, @as(usize, @intFromFloat(@trunc(@as(f64, @floatFromInt(crop.height)) * s))));
            try tiles.append(gpa, try if (s < 1) img.resizeArea(gpa, crop, tw, th) else img.resizeLinear(gpa, crop, tw, th)); // area only shrinks
        }
        img_unl += @intFromBool(found);
        if (n_img % 250 == 0) std.debug.print("  {d}/{d}\n", .{ n_img, count });
    }
    try u.writeFile(io, try std.fmt.allocPrint(gpa, "{s}.tsv", .{prefix}), tsv.written());
    try writeSheets(gpa, io, tiles.items, TILE, 10, 10, prefix);
    var so: u.Stdout = undefined;
    so.init(io);
    const w = so.w();
    try w.print("RetinaFace >= 0.9 on {d} training photos ({d} labelled faces): {d} confident detections, {d} on no face box or ignore region, in {d} photos\n", .{ n_img, n_faces, n_conf, n_unl, img_unl });
    try u.padRight(w, "unlabelled detections by width / image width (%)", 58);
    try w.writeAll(HEADER);
    try size_hist.print(w, "");
    try w.print("tiles in {s}_N.jpg (10 x 10, in {s}.tsv order)\n", .{ prefix, prefix });
    try w.flush();
}

pub fn main(init: std.process.Init) !void {
    const gpa = init.arena.allocator();
    const io = init.io;
    const a = try init.minimal.args.toSlice(gpa);
    const runtime = ort.runtimePath(init.environ_map);
    const mode = if (a.len > 1) a[1] else "";
    if (std.mem.eql(u8, mode, "scale") and a.len == 6) return scale(gpa, io, runtime, a[2], a[3], a[4], a[5]);
    if (std.mem.eql(u8, mode, "aug") and a.len == 7) return augStats(gpa, io, a[2], a[3], a[4], a[5], a[6]);
    if (std.mem.eql(u8, mode, "audit") and a.len == 7) return audit(gpa, io, runtime, a[2], a[3], a[4], try std.fmt.parseInt(usize, a[5], 10), a[6]);
    if (std.mem.eql(u8, mode, "sheet") and a.len == 6) return sheet(gpa, io, a[2], a[3], try std.fmt.parseInt(usize, a[4], 10), a[5]);
    fail("usage: see the header of tools/dataset/large_face_diag.zig", .{});
}

test "size bins and crop geometry" {
    try std.testing.expectEqual(@as(usize, 0), bin(0.05));
    try std.testing.expectEqual(@as(usize, 3), bin(0.44));
    try std.testing.expectEqual(@as(usize, 4), bin(0.45));
    try std.testing.expectEqual(@as(usize, 6), bin(0.9));
    const p: aug.Patch = .{ .left = 100, .top = 0, .side = 200 };
    const frac, const vis = clipFrac(.{ 50, 0, 150, 100 }, p); // half of a 100-wide face inside
    try std.testing.expectApproxEqAbs(@as(f64, 0.25), frac, 1e-9);
    try std.testing.expectApproxEqAbs(@as(f64, 0.5), vis, 1e-9);
    try std.testing.expect(!centreIn(.{ 50, 0, 150, 100 }, p) and centreIn(.{ 150, 0, 250, 100 }, p));
}

test "a 400 px face is owned by the stride-32 head" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const pos = try owners(arena.allocator(), &.{.{ 100, 100, 500, 500 }}, &.{}, 0);
    try std.testing.expect(pos[2] > 0 and pos[0] == 0);
}
