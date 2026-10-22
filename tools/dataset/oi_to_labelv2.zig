//! Write SCRFD labelv2 files that box *every* face Open Images drew, not just the clicked ones.
//!
//!   zig run -O ReleaseFast tools/dataset/oi_to_labelv2.zig -- <openimages_dir> train [--detections det.json] [--out file]
//!   zig run -O ReleaseFast tools/dataset/oi_to_labelv2.zig -- <openimages_dir> test  [--out file]
//!
//! Why this exists: `to_labelv2` builds from the eye-labelling JSON, which holds one entry per
//! *clicked* face. So `labelv2_train.txt` carried 2,500 boxes for 2,482 images whose Open Images
//! boxes number 12,070, and `labelv2_test.txt` 2,000 against 4,597. Every curve run was trained to
//! call ~9,300 real faces background, and every FP/image figure in CURVE_RESULTS.md counted real,
//! merely unclicked faces as false positives. The eye labels themselves were right; the boxes
//! around them were missing.
//!
//! Each image's block is written from the Open Images CSV, so it holds every face box:
//!
//! * A real face (not group-of, not a depiction) is a full 19-value line. If that face's eyes were
//!   clicked -- its box matches a line in the old labelv2 file at IoU >= 0.9 -- the old line is
//!   carried over verbatim, eye points and all. Otherwise its keypoints are `-1`, so it trains
//!   the box and not the landmark head.
//! * Group-of boxes and depictions become 5-value ignore regions (`x1 y1 x2 y2 1`): neither a face
//!   to find nor background to suppress. Depictions are the judgement call here -- statues, drawings,
//!   faces on screens -- and ignoring them leaves the detector free either way rather than teaching
//!   it one answer.
//! * With `--detections` (train only): a YuNet detection at 0.8 overlapping no Open Images box
//!   (IoU < 0.3) also becomes an ignore region. Open Images leaves ~0.4 faces per image unboxed,
//!   and DATA_CARD.md measured YuNet's unmatched detections at ~97-99% real faces; unlabelled, those
//!   would train as background.
//!
//! The test file gets no YuNet-derived regions, deliberately: it is the ruler, and a ruler built
//! from the incumbent's own output would flatter the incumbent.
const std = @import("std");
const u = @import("util.zig");
const json = std.json;

const NO_POINTS = "-1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1";

const Box = [4]f64;
const Old = struct { box: Box, line: []const u8 };
const BoxRow = struct { box: [4]f64, group: bool, depiction: bool };

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

/// Clicked faces from the previous labelv2 file, keyed by image id.
fn oldLines(gpa: std.mem.Allocator, io: std.Io, path: []const u8) !std.StringArrayHashMapUnmanaged(std.ArrayList(Old)) {
    var out: std.StringArrayHashMapUnmanaged(std.ArrayList(Old)) = .empty;
    const text = u.readFile(gpa, io, path) catch |e| switch (e) {
        error.FileNotFound => return out,
        else => return e,
    };
    var image: []const u8 = "";
    var lines = std.mem.splitScalar(u8, text, '\n');
    while (lines.next()) |raw| {
        const line = std.mem.trim(u8, raw, " \t\r");
        if (std.mem.startsWith(u8, line, "#")) {
            var parts = std.mem.tokenizeAny(u8, line, " \t");
            _ = parts.next();
            const file = parts.next() orelse "";
            image = file[0 .. std.mem.findScalarLast(u8, file, '.') orelse file.len];
        } else if (line.len > 0) {
            var parts = std.mem.tokenizeAny(u8, line, " \t");
            var box: Box = undefined;
            for (&box) |*b| b.* = try std.fmt.parseFloat(f64, parts.next().?);
            const e = try out.getOrPut(gpa, image);
            if (!e.found_existing) e.value_ptr.* = .empty;
            try e.value_ptr.append(gpa, .{ .box = box, .line = line });
        }
    }
    return out;
}

fn commas(gpa: std.mem.Allocator, n: i64) ![]u8 {
    const digits = try std.fmt.allocPrint(gpa, "{d}", .{n});
    var out: std.ArrayList(u8) = .empty;
    for (digits, 0..) |c, i| {
        if (i > 0 and (digits.len - i) % 3 == 0) try out.append(gpa, ',');
        try out.append(gpa, c);
    }
    return out.toOwnedSlice(gpa);
}

fn corners(w: *std.Io.Writer, b: Box) !void {
    for (b, 0..) |v, i| {
        if (i > 0) try w.writeAll(" ");
        try u.fixed(w, v, 2);
    }
}

pub fn main(init: std.process.Init) !void {
    const gpa = init.arena.allocator();
    const io = init.io;
    const argv = try init.minimal.args.toSlice(gpa);
    var data: ?[]const u8 = null;
    var split: ?[]const u8 = null;
    var detections_path: ?[]const u8 = null;
    var out_arg: ?[]const u8 = null;
    var i: usize = 1;
    while (i < argv.len) : (i += 1) {
        if (std.mem.eql(u8, argv[i], "--detections") and i + 1 < argv.len) {
            i += 1;
            detections_path = argv[i];
        } else if (std.mem.eql(u8, argv[i], "--out") and i + 1 < argv.len) {
            i += 1;
            out_arg = argv[i];
        } else if (data == null) data = argv[i] else split = argv[i];
    }
    const dir = data orelse return error.Usage;
    const sp = split orelse return error.Usage;
    const is_train = std.mem.eql(u8, sp, "train");
    if (!is_train and !std.mem.eql(u8, sp, "test")) return error.Usage;
    if (detections_path != null and !is_train) {
        std.debug.print("the test file must not carry YuNet-derived regions\n", .{});
        std.process.exit(2);
    }

    const csv_files: []const []const u8 = if (is_train) &.{"faces-oidv6-train-annotations-bbox.csv"} else &.{ "faces-validation-annotations-bbox.csv", "faces-test-annotations-bbox.csv" };
    var boxes: std.StringHashMapUnmanaged(std.ArrayList([]const []const u8)) = .empty;
    var box_csvs: std.ArrayList(u.Csv) = .empty;
    for (csv_files) |name| {
        const csv = try u.Csv.load(gpa, io, try std.fs.path.join(gpa, &.{ dir, name }));
        try box_csvs.append(gpa, csv);
        const c_id = csv.col("ImageID");
        for (csv.rows) |r| {
            const e = try boxes.getOrPut(gpa, r[c_id]);
            if (!e.found_existing) e.value_ptr.* = .empty;
            try e.value_ptr.append(gpa, r);
        }
    }

    const clicked = try oldLines(gpa, io, try std.fs.path.join(gpa, &.{ dir, if (is_train) "labelv2_train.txt" else "labelv2_test.txt" }));
    var images: std.ArrayList([]const u8) = .empty;
    if (is_train) {
        const m = try u.Csv.load(gpa, io, try std.fs.path.join(gpa, &.{ dir, "train_manifest.csv" }));
        const c = m.col("ImageID");
        for (m.rows) |r| try images.append(gpa, r[c]);
    } else { // the existing test images
        try images.appendSlice(gpa, clicked.keys());
        std.mem.sort([]const u8, images.items, {}, u.lessStr);
    }

    var detections: std.StringHashMapUnmanaged(std.ArrayList(Box)) = .empty;
    if (detections_path) |p| {
        for ((try u.readJson(gpa, io, p)).array.items) |rec| {
            const e = try detections.getOrPut(gpa, stemOf(u.str(u.field(rec, "image"))));
            if (!e.found_existing) e.value_ptr.* = .empty;
            for (u.field(rec, "detections").array.items) |det| {
                const b = u.field(det, "bbox").array.items;
                const x, const y, const bw, const bh = .{ u.num(b[0]), u.num(b[1]), u.num(b[2]), u.num(b[3]) };
                try e.value_ptr.append(gpa, .{ x, y, x + bw, y + bh });
            }
        }
    }

    // All box CSVs share a header layout; use the first one's column positions.
    const hc = box_csvs.items[0];
    const c_x0, const c_y0, const c_x1, const c_y1 = .{ hc.col("XMin"), hc.col("YMin"), hc.col("XMax"), hc.col("YMax") };
    const c_group, const c_dep = .{ hc.col("IsGroupOf"), hc.col("IsDepiction") };

    var counts: u.Counter = .{};
    var out: std.Io.Writer.Allocating = .init(gpa);
    const w = &out.writer;
    for (images.items) |image| {
        const path = try std.fmt.allocPrint(gpa, "{s}/images/{s}.jpg", .{ dir, image });
        std.Io.Dir.cwd().access(io, path, .{}) catch {
            try counts.inc(gpa, "missing_image");
            continue;
        };
        const size = u.jpegSizeOfFile(gpa, io, path) orelse {
            try counts.inc(gpa, "unreadable_header");
            continue;
        };
        const width: f64 = @floatFromInt(size[0]);
        const height: f64 = @floatFromInt(size[1]);
        try w.print("# {s}.jpg {d} {d}\n", .{ image, size[0], size[1] });
        try counts.inc(gpa, "images");

        var drawn: std.ArrayList(Box) = .empty;
        const old: []const Old = if (clicked.get(image)) |o| o.items else &.{};
        const carried = try gpa.alloc(bool, old.len);
        @memset(carried, false);
        var n_carried: usize = 0;
        if (boxes.get(image)) |rows| for (rows.items) |r| {
            const box: Box = .{
                try std.fmt.parseFloat(f64, r[c_x0]) * width, try std.fmt.parseFloat(f64, r[c_y0]) * height,
                try std.fmt.parseFloat(f64, r[c_x1]) * width, try std.fmt.parseFloat(f64, r[c_y1]) * height,
            };
            try drawn.append(gpa, box);
            if (std.mem.eql(u8, r[c_group], "1") or std.mem.eql(u8, r[c_dep], "1")) {
                try corners(w, box);
                try w.writeAll(" 1\n");
                try counts.inc(gpa, "ignore_group_or_depiction");
                continue;
            }
            var match: ?usize = null;
            for (old, 0..) |o, k| {
                if (!carried[k] and iou(box, o.box) >= 0.9) {
                    match = k;
                    break;
                }
            }
            if (match) |k| {
                carried[k] = true;
                n_carried += 1;
                try w.print("{s}\n", .{old[k].line});
                try counts.inc(gpa, "faces_with_eyes");
            } else {
                try corners(w, box);
                try w.print(" {s}\n", .{NO_POINTS});
                try counts.inc(gpa, "faces_box_only");
            }
        };
        try counts.add(gpa, "clicked_unmatched", @intCast(old.len - n_carried));

        if (detections.get(image)) |dets| for (dets.items) |det| {
            var all_below = true;
            for (drawn.items) |b| all_below = all_below and iou(det, b) < 0.3;
            if (all_below) {
                try corners(w, det);
                try w.writeAll(" 1\n");
                try counts.inc(gpa, "ignore_unboxed_yunet");
            }
        };
    }

    const out_path = out_arg orelse try std.fmt.allocPrint(gpa, "{s}/labelv2_{s}_all.txt", .{ dir, sp });
    if (out.written().len == 0) try w.writeAll("\n");
    try u.writeFile(io, out_path, out.written());

    var so: u.Stdout = undefined;
    so.init(io);
    const keys = try gpa.dupe([]const u8, counts.map.keys());
    std.mem.sort([]const u8, keys, {}, u.lessStr);
    for (keys) |k| {
        try u.padLeft(so.w(), k, 28);
        try so.w().print(": {s}\n", .{try commas(gpa, counts.get(k))});
    }
    try so.w().print("wrote {s}\n", .{out_path});
    try so.w().flush();
    // Every clicked face must land on an Open Images box; if one did not, its eyes would be lost.
    if (counts.get("clicked_unmatched") != 0) {
        std.debug.print("{d} clicked faces matched no Open Images box\n", .{counts.get("clicked_unmatched")});
        std.process.exit(1);
    }
}
