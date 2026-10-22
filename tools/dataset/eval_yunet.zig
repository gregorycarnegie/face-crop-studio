//! Score YuNet against Open Images' human-drawn face boxes (validation + test).
//!
//!   zig run -O ReleaseFast tools/dataset/eval_yunet.zig -- <detections.json> [images_dir] [data_dir]
//!
//! Ground truth: faces-{validation,test}-annotations-bbox.csv, minus group-of boxes (one box over
//! several faces, used as ignore regions) and depictions (drawings, statues, screens).
//!
//! Face size is reported as box width scaled to the detector's 640 px input, since that, not the
//! original pixel size, decides whether a face is findable. Image dimensions come from the JPEG
//! header, so no image library is needed.
const std = @import("std");
const u = @import("util.zig");
const json = std.json;

const IOU_MATCH = 0.5;
const BUCKET_LOS = [_]f64{ 16, 32, 64, 128 };
const BUCKET_NAMES = [_][]const u8{ "16-32px", "32-64px", "64-128px", "128-infpx" };

const Box = [4]f64;
const Det = struct { score: f64, box: Box };
const Truth = struct { box: Box, width640: f64 };

fn iou(a: Box, b: Box) f64 {
    const ix = @max(0.0, @min(a[2], b[2]) - @max(a[0], b[0]));
    const iy = @max(0.0, @min(a[3], b[3]) - @max(a[1], b[1]));
    const inter = ix * iy;
    const union_ = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter;
    return if (union_ > 0) inter / union_ else 0.0;
}

fn bucket(width640: f64) []const u8 {
    for (BUCKET_LOS, 0..) |lo, i| {
        const hi: f64 = if (i + 1 < BUCKET_LOS.len) BUCKET_LOS[i + 1] else 1e9;
        if (lo <= width640 and width640 < hi) return BUCKET_NAMES[i];
    }
    return "<16px";
}

fn stemOf(path: []const u8) []const u8 {
    const base = path[if (std.mem.findScalarLast(u8, path, '/')) |i| i + 1 else 0..];
    const base2 = base[if (std.mem.findScalarLast(u8, base, '\\')) |i| i + 1 else 0..];
    return base2[0 .. std.mem.findScalarLast(u8, base2, '.') orelse base2.len];
}

fn pct(gpa: std.mem.Allocator, num: i64, den: i64) ![]u8 {
    const v = @as(f64, @floatFromInt(num)) / @as(f64, @floatFromInt(den)) * 100;
    return std.fmt.allocPrint(gpa, "{s}%", .{try u.fixedAlloc(gpa, v, 1)});
}

fn desc(_: void, a: Det, b: Det) bool {
    return a.score > b.score;
}

pub fn main(init: std.process.Init) !void {
    const gpa = init.arena.allocator();
    const io = init.io;
    const args = try init.minimal.args.toSlice(gpa);
    if (args.len < 2) return error.Usage;
    const data = if (args.len > 3) args[3] else ".";
    const images_dir = if (args.len > 2) args[2] else try std.fs.path.join(gpa, &.{ data, "images" });

    // Detections, keyed by image id (the file stem).
    var detected: std.StringHashMapUnmanaged(std.ArrayList(Det)) = .empty;
    const raw = try u.readJson(gpa, io, args[1]);
    const records = switch (raw) {
        .array => |a| a.items,
        else => if (raw.object.get("images")) |v| v.array.items else if (raw.object.get("results")) |v| v.array.items else &.{},
    };
    var n_detections: usize = 0;
    for (records) |rec| {
        const e = try detected.getOrPut(gpa, stemOf(u.str(u.field(rec, "image"))));
        if (!e.found_existing) e.value_ptr.* = .empty;
        for (u.field(rec, "detections").array.items) |d| {
            const b = u.field(d, "bbox").array.items;
            const x, const y, const bw, const bh = .{ u.num(b[0]), u.num(b[1]), u.num(b[2]), u.num(b[3]) };
            try e.value_ptr.append(gpa, .{ .score = u.num(u.field(d, "score")), .box = .{ x, y, x + bw, y + bh } });
        }
    }

    // Ground truth, in pixels, plus the ignore regions.
    var truth: std.StringHashMapUnmanaged(std.ArrayList(Truth)) = .empty;
    var ignore: std.StringHashMapUnmanaged(std.ArrayList(Box)) = .empty;
    var sizes: std.StringArrayHashMapUnmanaged(?[2]u32) = .empty;
    var missing_size: usize = 0;
    var n_truth: usize = 0;
    var n_ignore: usize = 0;
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
            const size = entry.value_ptr.* orelse {
                missing_size += 1;
                continue;
            };
            const iw: f64 = @floatFromInt(size[0]);
            const ih: f64 = @floatFromInt(size[1]);
            const x0, const x1 = .{ try std.fmt.parseFloat(f64, r[c_x0]), try std.fmt.parseFloat(f64, r[c_x1]) };
            const y0, const y1 = .{ try std.fmt.parseFloat(f64, r[c_y0]), try std.fmt.parseFloat(f64, r[c_y1]) };
            const box: Box = .{ x0 * iw, y0 * ih, x1 * iw, y1 * ih };
            if (std.mem.eql(u8, r[c_group], "1") or std.mem.eql(u8, r[c_dep], "1")) {
                const e = try ignore.getOrPut(gpa, stem);
                if (!e.found_existing) e.value_ptr.* = .empty;
                try e.value_ptr.append(gpa, box);
                n_ignore += 1;
            } else {
                const e = try truth.getOrPut(gpa, stem);
                if (!e.found_existing) e.value_ptr.* = .empty;
                try e.value_ptr.append(gpa, .{ .box = box, .width640 = (x1 - x0) * 640 });
                n_truth += 1;
            }
        }
    }

    var scored: std.ArrayList([]const u8) = .empty;
    for (sizes.keys(), sizes.values()) |k, v| {
        if (v != null) {
            try scored.append(gpa, k);
            if (detected.get(k)) |d| n_detections += d.items.len;
        }
    }
    var so: u.Stdout = undefined;
    so.init(io);
    const w = so.w();
    try w.print("{d} images scored, {d} missing or unreadable ({d} boxes skipped)\n", .{ scored.items.len, sizes.count() - scored.items.len, missing_size });
    try w.print("{d} ground-truth faces, {d} ignore regions, {d} detections\n\n", .{ n_truth, n_ignore, n_detections });

    var found: u.Counter = .{};
    var total: u.Counter = .{};
    var extra: usize = 0;
    var h_ratio: std.ArrayList(f64) = .empty;
    var w_ratio: std.ArrayList(f64) = .empty;
    var y_shift: std.ArrayList(f64) = .empty;
    for (scored.items) |stem| {
        const gt: []const Truth = if (truth.get(stem)) |t| t.items else &.{};
        const dets = if (detected.get(stem)) |d| try gpa.dupe(Det, d.items) else try gpa.alloc(Det, 0);
        std.mem.sort(Det, dets, {}, desc);
        const taken = try gpa.alloc(bool, gt.len);
        @memset(taken, false);
        for (dets) |det| {
            var best: ?usize = null;
            var best_iou: f64 = IOU_MATCH;
            for (gt, 0..) |g, i| {
                if (taken[i]) continue;
                const s = iou(det.box, g.box);
                if (s >= best_iou) {
                    best = i;
                    best_iou = s;
                }
            }
            const b = best orelse {
                var ignored = false;
                if (ignore.get(stem)) |igs| for (igs.items) |ig| {
                    if (iou(det.box, ig) > 0.3) ignored = true;
                };
                if (!ignored) extra += 1;
                continue;
            };
            taken[b] = true;
            try found.inc(gpa, bucket(gt[b].width640));
            const gbox = gt[b].box;
            const gh = gbox[3] - gbox[1];
            const gw = gbox[2] - gbox[0];
            try h_ratio.append(gpa, (det.box[3] - det.box[1]) / gh);
            try w_ratio.append(gpa, (det.box[2] - det.box[0]) / gw);
            try y_shift.append(gpa, (((det.box[1] + det.box[3]) / 2) - ((gbox[1] + gbox[3]) / 2)) / gh);
        }
        for (gt) |g| try total.inc(gpa, bucket(g.width640));
    }

    try w.print("{s:<22} {s:>8} {s:>8} {s:>8}\n", .{ "face width at 640px", "faces", "found", "recall" });
    var all_t: i64 = 0;
    var all_f: i64 = 0;
    var cro_t: i64 = 0;
    var cro_f: i64 = 0;
    for ([_][]const u8{"<16px"} ++ BUCKET_NAMES) |key| {
        const t, const f = .{ total.get(key), found.get(key) };
        all_t += t;
        all_f += f;
        if (!std.mem.eql(u8, key, "<16px") and !std.mem.eql(u8, key, "16-32px")) {
            cro_t += t;
            cro_f += f;
        }
        if (t == 0) continue;
        try u.padRight(w, key, 22);
        try w.writeAll(" ");
        try u.padLeft(w, try std.fmt.allocPrint(gpa, "{d}", .{t}), 8);
        try w.writeAll(" ");
        try u.padLeft(w, try std.fmt.allocPrint(gpa, "{d}", .{f}), 8);
        try w.writeAll(" ");
        try u.padLeft(w, try pct(gpa, f, t), 7);
        try w.writeAll("\n");
    }
    for ([_]struct { []const u8, i64, i64 }{ .{ "all", all_t, all_f }, .{ "croppable (>=32px)", cro_t, cro_f } }) |r| {
        try u.padRight(w, r[0], 22);
        try w.writeAll(" ");
        try u.padLeft(w, try std.fmt.allocPrint(gpa, "{d}", .{r[1]}), 8);
        try w.writeAll(" ");
        try u.padLeft(w, try std.fmt.allocPrint(gpa, "{d}", .{r[2]}), 8);
        try w.writeAll(" ");
        try u.padLeft(w, try pct(gpa, r[2], r[1]), 7);
        try w.writeAll("\n");
    }
    try w.print("\nunmatched detections (excluding ignore regions): {d}\n", .{extra});
    if (h_ratio.items.len > 0) {
        try w.print("\nbox extent, YuNet / Open Images, on matched pairs:\n  height ratio   median {s}\n  width ratio    median {s}\n", .{
            try u.fixedAlloc(gpa, try u.median(gpa, h_ratio.items), 2),
            try u.fixedAlloc(gpa, try u.median(gpa, w_ratio.items), 2),
        });
        const ys = try u.median(gpa, y_shift.items);
        try w.print("  centre y shift median {s}{s} of GT height (+ is lower)\n", .{ if (ys >= 0) "+" else "", try u.fixedAlloc(gpa, ys, 2) });
    }
    try w.flush();
}
