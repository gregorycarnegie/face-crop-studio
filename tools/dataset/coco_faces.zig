//! How much licence-clean face data does COCO 2017 hold?
//!
//! Reads annotations/person_keypoints_{train,val}2017.json straight from the zip.
//! Keypoint order: 0 nose, 1 left_eye, 2 right_eye, 3 left_ear, 4 right_ear.
//! Visibility: 0 not labelled, 1 labelled but occluded, 2 labelled and visible.
//!
//!   zig run tools/dataset/coco_faces.zig -- [annotations_trainval2017.zip]
const std = @import("std");
const u = @import("util.zig");
const json = std.json;

const IOD_LOS = [_]f64{ 0, 8, 16, 32 }; // inter-ocular distance buckets, px
const IOD_NAMES = [_][]const u8{ "iod_0-8", "iod_8-16", "iod_16-32", "iod_32-inf" };

fn iodKey(iod: f64) []const u8 {
    for (IOD_LOS[0..3], 0..) |lo, i| {
        if (lo <= iod and iod < IOD_LOS[i + 1]) return IOD_NAMES[i];
    }
    return IOD_NAMES[3];
}

const REPORT = [_][]const u8{
    "images",                       "negatives (no person annotated)",
    "images with a both-eyes face", "  ...and no crowd / unlabelled person",
    "face_both_eyes",               "face_both_eyes_visible",
    IOD_NAMES[0],                   IOD_NAMES[1],
    IOD_NAMES[2],                   IOD_NAMES[3],
    "face_partial",                 "no_face_points",
    "unlabelled_person",            "crowd",
};

fn row(w: *std.Io.Writer, gpa: std.mem.Allocator, indent: []const u8, key: []const u8, n: i64, key_width: usize, n_width: usize) !void {
    try w.writeAll(indent);
    try u.padRight(w, key, key_width);
    try w.writeAll(" ");
    try u.padLeft(w, try std.fmt.allocPrint(gpa, "{d}", .{n}), n_width);
    try w.writeAll("\n");
}

pub fn main(init: std.process.Init) !void {
    const gpa = init.arena.allocator();
    const io = init.io;
    const args = try init.minimal.args.toSlice(gpa);
    const zip_path = if (args.len > 1) args[1] else "annotations_trainval2017.zip";
    var so: u.Stdout = undefined;
    so.init(io);
    const w = so.w();

    var grand: u.Counter = .{};
    for ([_][]const u8{ "train2017", "val2017" }) |split| {
        const name = try std.fmt.allocPrint(gpa, "annotations/person_keypoints_{s}.json", .{split});
        const data = try json.parseFromSliceLeaky(json.Value, gpa, try u.zipEntry(gpa, io, zip_path, name), .{});

        var licences: std.AutoArrayHashMapUnmanaged(i64, []const u8) = .empty;
        for (u.field(data, "licenses").array.items) |l| try licences.put(gpa, u.field(l, "id").integer, u.str(u.field(l, "name")));
        std.mem.sort(i64, licences.keys(), {}, std.sort.asc(i64));
        var strict: std.AutoHashMapUnmanaged(i64, void) = .empty;
        var lenient: std.AutoHashMapUnmanaged(i64, void) = .empty;
        for (licences.keys(), licences.values()) |lid, lname| { // keys sorted above; values re-read by key below
            _ = lname;
            const n = licences.get(lid).?;
            if (std.mem.indexOf(u8, n, "NonCommercial") != null) continue;
            try lenient.put(gpa, lid, {});
            if (std.mem.indexOf(u8, n, "ShareAlike") == null and std.mem.indexOf(u8, n, "NoDerivs") == null) try strict.put(gpa, lid, {});
        }

        const image_items = u.field(data, "images").array.items;
        var per_image: std.AutoHashMapUnmanaged(i64, u.Counter) = .empty;
        for (u.field(data, "annotations").array.items) |ann| {
            const e = try per_image.getOrPut(gpa, u.field(ann, "image_id").integer);
            if (!e.found_existing) e.value_ptr.* = .{};
            const s = e.value_ptr;
            if (u.truthy(u.field(ann, "iscrowd"))) {
                try s.inc(gpa, "crowd");
                continue;
            }
            const kp = u.field(ann, "keypoints").array.items;
            if (u.field(ann, "num_keypoints").integer == 0) {
                try s.inc(gpa, "unlabelled_person");
                continue;
            }
            const v0, const v1, const v2 = .{ u.num(kp[2]) != 0, u.num(kp[5]) != 0, u.num(kp[8]) != 0 };
            if (v1 and v2) {
                const iod = std.math.hypot(u.num(kp[3]) - u.num(kp[6]), u.num(kp[4]) - u.num(kp[7]));
                try s.inc(gpa, "face_both_eyes");
                try s.inc(gpa, iodKey(iod));
                if (u.num(kp[5]) == 2 and u.num(kp[8]) == 2) try s.inc(gpa, "face_both_eyes_visible");
            } else if (v0 or v1 or v2) {
                try s.inc(gpa, "face_partial"); // profile or partly occluded: needs a hand-drawn box
            } else {
                try s.inc(gpa, "no_face_points"); // person facing away or head out of frame
            }
        }

        try w.print("\n=== {s}: {d} images\n", .{ split, image_items.len });
        var lic_counts: std.AutoHashMapUnmanaged(i64, i64) = .empty;
        for (image_items) |im| {
            const e = try lic_counts.getOrPut(gpa, u.field(im, "license").integer);
            if (!e.found_existing) e.value_ptr.* = 0;
            e.value_ptr.* += 1;
        }
        for (licences.keys()) |lid| {
            const tag = if (strict.contains(lid)) "strict" else if (lenient.contains(lid)) "lenient" else "EXCLUDED";
            try w.print("  licence {d} ", .{lid});
            try u.padRight(w, licences.get(lid).?, 45);
            try w.writeAll(" ");
            try u.padLeft(w, try std.fmt.allocPrint(gpa, "{d}", .{lic_counts.get(lid) orelse 0}), 7);
            try w.print("  {s}\n", .{tag});
        }

        const tiers = [_][]const u8{ "strict", "lenient", "all licences" };
        for (tiers) |tier| {
            var c: u.Counter = .{};
            for (image_items) |im| {
                const lid = u.field(im, "license").integer;
                if (std.mem.eql(u8, tier, "strict") and !strict.contains(lid)) continue;
                if (std.mem.eql(u8, tier, "lenient") and !lenient.contains(lid)) continue;
                try c.inc(gpa, "images");
                const s = per_image.get(u.field(im, "id").integer) orelse {
                    try c.inc(gpa, "negatives (no person annotated)");
                    continue;
                };
                for (s.map.keys(), s.map.values()) |k, n| try c.add(gpa, k, n);
                if (s.get("face_both_eyes") != 0) {
                    try c.inc(gpa, "images with a both-eyes face");
                    if (s.get("crowd") == 0 and s.get("unlabelled_person") == 0) try c.inc(gpa, "  ...and no crowd / unlabelled person");
                }
            }
            for (c.map.keys(), c.map.values()) |k, n| try grand.add(gpa, try std.fmt.allocPrint(gpa, "{s}|{s}", .{ tier, k }), n);
            try w.print("  [{s}]\n", .{tier});
            for (REPORT) |k| try row(w, gpa, "    ", k, c.get(k), 42, 8);
        }
    }

    try w.writeAll("\n=== train + val\n");
    const keys = try gpa.dupe([]const u8, grand.map.keys());
    std.mem.sort([]const u8, keys, {}, u.lessStr);
    for ([_][]const u8{ "strict", "lenient", "all licences" }) |tier| {
        try w.print("  [{s}]\n", .{tier});
        for (keys) |k| {
            if (k.len > tier.len and std.mem.startsWith(u8, k, tier) and k[tier.len] == '|')
                try row(w, gpa, "    ", k[tier.len + 1 ..], grand.get(k), 42, 8);
        }
    }
    try w.flush();
}
