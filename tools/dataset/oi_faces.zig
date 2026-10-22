//! How much licence-clean, croppable face data does Open Images hold?
//!
//!   zig run tools/dataset/oi_faces.zig -- [data_dir]     (default: the current directory)
//!
//! Inputs, in `data_dir`: faces-*-annotations-bbox.csv (box rows filtered to /m/0dzct Human face)
//! and *-images*-with-rotation.csv (per-image licence, author, URLs).
//!
//! Box coordinates are normalised, and the image tables carry no pixel dimensions, so face size is
//! reported as box width as a fraction of image width. At the detector's 640 px input that fraction
//! times 640 is roughly the face width in pixels (exactly so for landscape images).
const std = @import("std");
const u = @import("util.zig");

const Split = struct { name: []const u8, boxes: []const u8, images: []const u8 };
const SPLITS = [_]Split{
    .{ .name = "train", .boxes = "faces-oidv6-train-annotations-bbox.csv", .images = "train-images-boxable-with-rotation.csv" },
    .{ .name = "validation", .boxes = "faces-validation-annotations-bbox.csv", .images = "validation-images-with-rotation.csv" },
    .{ .name = "test", .boxes = "faces-test-annotations-bbox.csv", .images = "test-images-with-rotation.csv" },
};
const WIDTH_BUCKETS = [_]f64{ 0.025, 0.05, 0.10, 0.20, 1.01 }; // upper bounds: <16, <32, <64, <128, >=128 px at 640

fn widthBucket(gpa: std.mem.Allocator, frac: f64) ![]const u8 {
    var lo: f64 = 0.0;
    for (WIDTH_BUCKETS) |hi| {
        if (frac < hi) {
            var out: std.Io.Writer.Allocating = .init(gpa);
            try u.padLeft(&out.writer, try u.fixedAlloc(gpa, lo * 640, 0), 4);
            try out.writer.writeAll("-");
            try u.padRight(&out.writer, try u.fixedAlloc(gpa, @min(hi, 1.0) * 640, 0), 4);
            try out.writer.writeAll("px");
            return out.toOwnedSlice();
        }
        lo = hi;
    }
    return "  full";
}

const Row = struct { source: []const u8, group: bool, depiction: bool, occluded: bool, truncated: bool, xmin: f64, xmax: f64 };
const Licence = struct { name: []const u8, has_author: bool };

fn isCcBy2(l: []const u8) bool {
    return std.mem.endsWith(u8, std.mem.trimEnd(u8, l, "/"), "licenses/by/2.0");
}

pub fn main(init: std.process.Init) !void {
    const gpa = init.arena.allocator();
    const io = init.io;
    const args = try init.minimal.args.toSlice(gpa);
    const data = if (args.len > 1) args[1] else ".";
    var so: u.Stdout = undefined;
    so.init(io);
    const w = so.w();

    const grand_names: [2][]const u8 = .{ "CC BY 2.0", "any licence" };
    var grand: [2]u.Counter = .{ .{}, .{} };

    for (SPLITS) |split| {
        var boxes: std.StringArrayHashMapUnmanaged(std.ArrayList(Row)) = .empty;
        const bc = try u.Csv.load(gpa, io, try std.fs.path.join(gpa, &.{ data, split.boxes }));
        const c_id, const c_src, const c_xmin, const c_xmax = .{ bc.col("ImageID"), bc.col("Source"), bc.col("XMin"), bc.col("XMax") };
        const c_group, const c_dep, const c_occ, const c_trunc = .{ bc.col("IsGroupOf"), bc.col("IsDepiction"), bc.col("IsOccluded"), bc.col("IsTruncated") };
        for (bc.rows) |r| {
            const e = try boxes.getOrPut(gpa, r[c_id]);
            if (!e.found_existing) e.value_ptr.* = .empty;
            try e.value_ptr.append(gpa, .{
                .source = r[c_src],
                .group = std.mem.eql(u8, r[c_group], "1"),
                .depiction = std.mem.eql(u8, r[c_dep], "1"),
                .occluded = std.mem.eql(u8, r[c_occ], "1"),
                .truncated = std.mem.eql(u8, r[c_trunc], "1"),
                .xmin = try std.fmt.parseFloat(f64, r[c_xmin]),
                .xmax = try std.fmt.parseFloat(f64, r[c_xmax]),
            });
        }

        var licence_of: std.StringArrayHashMapUnmanaged(Licence) = .empty;
        const ic = try u.Csv.load(gpa, io, try std.fs.path.join(gpa, &.{ data, split.images }));
        const i_id, const i_lic, const i_auth = .{ ic.col("ImageID"), ic.col("License"), ic.col("Author") };
        for (ic.rows) |r| {
            if (boxes.contains(r[i_id])) try licence_of.put(gpa, r[i_id], .{ .name = r[i_lic], .has_author = std.mem.trim(u8, r[i_auth], " \t\r\n").len > 0 });
        }

        var total_boxes: usize = 0;
        for (boxes.values()) |v| total_boxes += v.items.len;
        try w.print("\n=== {s}: {d} images with face boxes, {d} boxes, {d} without a licence row\n", .{ split.name, boxes.count(), total_boxes, boxes.count() - licence_of.count() });
        var lic: u.Counter = .{};
        var no_author: usize = 0;
        for (licence_of.values()) |l| {
            try lic.inc(gpa, l.name);
            no_author += @intFromBool(!l.has_author);
        }
        const Entry = struct { name: []const u8, n: i64 };
        var entries: std.ArrayList(Entry) = .empty;
        for (lic.map.keys(), lic.map.values()) |k, v| try entries.append(gpa, .{ .name = k, .n = v });
        std.mem.sort(Entry, entries.items, {}, struct { // stable, so ties keep first-seen order like most_common()
            fn gt(_: void, a: Entry, b: Entry) bool {
                return a.n > b.n;
            }
        }.gt);
        for (entries.items) |e| {
            try w.writeAll("  ");
            try u.padLeft(w, try std.fmt.allocPrint(gpa, "{d}", .{e.n}), 8);
            try w.print("  {s}\n", .{e.name});
        }
        try w.print("  images with no Author recorded: {d}\n", .{no_author});

        for (&grand, 0..) |*c, tier| {
            for (boxes.keys(), boxes.values()) |image_id, rows| {
                const entry = licence_of.get(image_id) orelse continue;
                if (tier == 0 and !isCcBy2(entry.name)) continue;
                try c.inc(gpa, "images");
                var croppable_here: usize = 0;
                var any_group = false;
                for (rows.items) |r| {
                    try c.inc(gpa, "boxes");
                    try c.inc(gpa, try std.fmt.allocPrint(gpa, "source {s}", .{r.source}));
                    any_group = any_group or r.group;
                    if (r.group) {
                        try c.inc(gpa, "group-of boxes (ignore regions)");
                        continue;
                    }
                    if (r.depiction) try c.inc(gpa, "depictions (drawings, statues, screens)");
                    if (r.occluded) try c.inc(gpa, "occluded");
                    if (r.truncated) try c.inc(gpa, "truncated");
                    const width = r.xmax - r.xmin;
                    try c.inc(gpa, try std.fmt.allocPrint(gpa, "width {s}", .{try widthBucket(gpa, width)}));
                    if (width >= 0.05 and !r.depiction) {
                        try c.inc(gpa, "croppable faces (>=32 px, real, single)");
                        croppable_here += 1;
                    }
                }
                if (croppable_here > 0) try c.inc(gpa, "images with a croppable face");
                if (any_group) try c.inc(gpa, "images containing a group-of box");
            }
        }
    }

    for (&grand, grand_names) |*c, tier| {
        try w.print("\n=== all splits, {s}\n", .{tier});
        const keys = try gpa.dupe([]const u8, c.map.keys());
        std.mem.sort([]const u8, keys, {}, u.lessStr);
        for (keys) |k| {
            try w.writeAll("  ");
            try u.padRight(w, k, 44);
            try w.writeAll(" ");
            try u.padLeft(w, try std.fmt.allocPrint(gpa, "{d}", .{c.get(k)}), 9);
            try w.writeAll("\n");
        }
    }
    try w.flush();
}
