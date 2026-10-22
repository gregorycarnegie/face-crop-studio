//! Turn clicked eye points into COCO-format keypoint annotations.
//!
//!   zig run tools/dataset/to_coco_keypoints.zig -- <data_dir> [out_dir]
//!
//! Reads <data_dir>/eye_labels.jsonl (from label_eyes) plus the Open Images box CSVs, and
//! writes two COCO keypoint files: `face_keypoints_train.json` from the train split and
//! `face_keypoints_test.json` from validation+test. Each has its own images, one annotation per
//! labelled face, and a `face` category with five keypoints.
//!
//! Two files rather than one because validation+test is the held-out test set: a single file
//! containing both is one careless glob away from training on the only yardstick there is.
//!
//! Why COCO and not RetinaFace's label.txt, which is what SCRFD's own training scripts read:
//! these labels are partial. Eyes are clicked, nose and mouth corners are not, and label.txt has
//! no way to say "these two points are known and those three are not" -- a face either carries
//! all five or a -1 placeholder. Writing it would discard every eye point. COCO keypoints carry a
//! per-point visibility flag, mmdetection and mmpose read the format directly, and it matches the
//! shape of COCO's own eye points (DATA_CARD section 3), so the two sources can merge later.
//!
//! Keypoint order is the RetinaFace/YuNet one, in the subject's own frame:
//! right_eye, left_eye, nose, right_mouth, left_mouth. Resolving "which eye is which" from a
//! click needs the flags, because screen position only implies a side while the head's roll stays
//! within +/-90 degrees:
//!
//!   unflagged    viewer-left click is the subject's RIGHT eye (measured: YuNet agrees 99.8%)
//!   upside_down  the face is inverted, so the two swap
//!   steep_roll   eyes nearly in vertical line; side is genuinely unknown, so neither is
//!                written as a keypoint. Both points are kept under `fcs.eyes_unordered` so a
//!                trainer that can use unordered pairs still has them.
//!
//! Faces skipped during labelling keep their box with no keypoints: they are real human-drawn
//! faces where a second eye was not visible, which is exactly what a detector head should train
//! on and a landmark head should not.
const std = @import("std");
const u = @import("util.zig");
const json = std.json;

const SPLIT_FILES = [_][2][]const u8{
    .{ "validation", "faces-validation-annotations-bbox.csv" },
    .{ "test", "faces-test-annotations-bbox.csv" },
    .{ "train", "faces-oidv6-train-annotations-bbox.csv" },
};

const Extra = struct { occluded: bool, truncated: bool, split: []const u8 };

const ImageEntry = struct { id: usize, name: []const u8, width: u32, height: u32 };

/// A number that remembers whether Python would hold it as an int: `round(0 * 1024, 1)` is `0`,
/// not `0.0`, and `json.dumps` prints the difference.
const Num = struct {
    v: f64,
    int: bool,

    fn of(x: json.Value) Num {
        return .{ .v = u.num(x), .int = x == .integer };
    }
    fn scale(self: Num, k: f64) Num {
        return .{ .v = u.pyRound(self.v * k, 1), .int = self.int };
    }
    fn write(self: Num, w: *std.Io.Writer) !void {
        if (self.int) try w.print("{d}", .{@as(i64, @intFromFloat(self.v))}) else try u.pyFloat(w, self.v);
    }
};

const Tally = struct { with_eyes: usize = 0, side_unknown: usize = 0, skipped: usize = 0 };

const Group = struct {
    images: std.StringArrayHashMapUnmanaged(ImageEntry) = .empty,
    annotations: std.ArrayList(u8) = .empty, // already-serialised annotation objects, comma separated
    count: usize = 0,
    tally: Tally = .{},
};

fn lessLabel(_: void, a: json.Value, b: json.Value) bool {
    return std.mem.lessThan(u8, u.str(u.field(a, "id")), u.str(u.field(b, "id")));
}

pub fn main(init: std.process.Init) !void {
    const gpa = init.arena.allocator();
    const io = init.io;
    const args = try init.minimal.args.toSlice(gpa);
    if (args.len < 2) return error.Usage;
    const data = try std.Io.Dir.cwd().realPathFileAlloc(io, args[1], gpa);
    const out_dir = if (args.len > 2) args[2] else data;

    const last = try u.lastWins(gpa, io, try std.fs.path.join(gpa, &.{ data, "eye_labels.jsonl" }), "id");
    const labels = try gpa.dupe(json.Value, last.values());
    std.mem.sort(json.Value, labels, {}, lessLabel);

    // Boxes are stored normalised, so every image needs its pixel size, and the CSVs carry the
    // occlusion flags worth passing through. The train CSV holds a million rows, so it is only
    // read when the labels actually reference that split.
    var train_referenced = false;
    for (labels) |r| {
        if (u.optField(r, "split")) |s| if (s == .string and std.mem.eql(u8, s.string, "train")) {
            train_referenced = true;
        };
    }
    var flags: std.StringHashMapUnmanaged(Extra) = .empty;
    for (SPLIT_FILES) |entry| {
        const split, const name = entry;
        if (std.mem.eql(u8, split, "train") and !train_referenced) continue;
        const csv = u.Csv.load(gpa, io, try std.fs.path.join(gpa, &.{ data, name })) catch |e| switch (e) {
            error.FileNotFound => continue,
            else => return e,
        };
        const c_id, const c_xmin, const c_occ, const c_trunc = .{ csv.col("ImageID"), csv.col("XMin"), csv.col("IsOccluded"), csv.col("IsTruncated") };
        for (csv.rows) |row| {
            const x = try std.fmt.parseFloat(f64, row[c_xmin]);
            const key = try std.fmt.allocPrint(gpa, "{s}|{d:.6}", .{ row[c_id], x });
            try flags.put(gpa, key, .{ .occluded = std.mem.eql(u8, row[c_occ], "1"), .truncated = std.mem.eql(u8, row[c_trunc], "1"), .split = split });
        }
    }

    // Ids are numbered within each file, so either is valid COCO on its own.
    var groups: std.StringArrayHashMapUnmanaged(Group) = .empty;
    var dropped: usize = 0;
    for (labels) |record| {
        const stem = u.str(u.field(record, "image"));
        const box = u.field(record, "box").array.items;
        const nx0, const ny0, const nx1, const ny1 = .{ Num.of(box[0]), Num.of(box[1]), Num.of(box[2]), Num.of(box[3]) };
        const x0 = nx0.v;
        const extra = flags.get(try std.fmt.allocPrint(gpa, "{s}|{d:.6}", .{ stem, x0 }));
        // The record's own split wins: it was stamped by the server that queued the face, while
        // the CSV lookup is a best effort that can miss on a rounding difference.
        const rec_split: ?[]const u8 = if (u.optField(record, "split")) |s| (if (s == .string and s.string.len > 0) s.string else null) else null;
        const split = rec_split orelse (if (extra) |e| e.split else "unknown");
        const target: []const u8 = if (std.mem.eql(u8, split, "train")) "train" else "test";
        const g = try groups.getOrPut(gpa, target);
        if (!g.found_existing) g.value_ptr.* = .{};
        const group = g.value_ptr;

        if (!group.images.contains(stem)) {
            const jpg = try std.fmt.allocPrint(gpa, "{s}/images/{s}.jpg", .{ data, stem });
            const size = u.jpegSizeOfFile(gpa, io, jpg) orelse {
                dropped += 1;
                continue;
            };
            try group.images.put(gpa, stem, .{ .id = group.images.count() + 1, .name = stem, .width = size[0], .height = size[1] });
        }
        const image = group.images.get(stem).?;
        const iw: f64 = @floatFromInt(image.width);
        const ih: f64 = @floatFromInt(image.height);

        const skipped = u.truthy(u.optField(record, "skipped"));
        const has_eyes = u.optField(record, "viewer_left") != null and !skipped;
        const steep = u.truthy(u.optField(record, "steep_roll"));
        const upside = u.truthy(u.optField(record, "upside_down"));
        var kp: [15]Num = @splat(.{ .v = 0, .int = false });
        var num_keypoints: u8 = 0;
        var unordered: ?[2][2]Num = null;
        if (has_eyes) {
            const left = u.field(record, "viewer_left").array.items;
            const right = u.field(record, "viewer_right").array.items;
            if (steep) {
                unordered = .{ .{ Num.of(left[0]), Num.of(left[1]) }, .{ Num.of(right[0]), Num.of(right[1]) } };
            } else {
                const subject_right, const subject_left = if (upside) .{ right, left } else .{ left, right };
                for ([_][]json.Value{ subject_right, subject_left }, 0..) |p, index| {
                    kp[index * 3] = Num.of(p[0]).scale(iw);
                    kp[index * 3 + 1] = Num.of(p[1]).scale(ih);
                    kp[index * 3 + 2] = .{ .v = 2.0, .int = false }; // COCO: labelled and visible
                }
                num_keypoints = 2;
            }
        }
        const bbox = [4]Num{
            nx0.scale(iw),
            ny0.scale(ih),
            (Num{ .v = nx1.v - nx0.v, .int = nx1.int and nx0.int }).scale(iw),
            (Num{ .v = ny1.v - ny0.v, .int = ny1.int and ny0.int }).scale(ih),
        };

        var a: std.Io.Writer.Allocating = .init(gpa);
        const w = &a.writer;
        group.count += 1;
        try w.print("{s}{{\"id\": {d}, \"image_id\": {d}, \"category_id\": 1, \"bbox\": [", .{ if (group.count > 1) ", " else "", group.count, image.id });
        for (bbox, 0..) |v, i| {
            if (i > 0) try w.writeAll(", ");
            try v.write(w);
        }
        try w.writeAll("], \"area\": ");
        try (Num{ .v = bbox[2].v * bbox[3].v, .int = bbox[2].int and bbox[3].int }).scale(1).write(w);
        try w.writeAll(", \"iscrowd\": 0, \"keypoints\": [");
        for (kp, 0..) |v, i| {
            if (i > 0) try w.writeAll(", ");
            try v.write(w);
        }
        try w.print("], \"num_keypoints\": {d}, \"fcs\": {{\"upside_down\": {s}, \"steep_roll\": {s}, \"eyes_skipped\": {s}, \"eyes_unordered\": ", .{ num_keypoints, bstr(upside), bstr(steep), bstr(skipped) });
        if (unordered) |pair| {
            try w.writeAll("[");
            for (pair, 0..) |p, i| {
                if (i > 0) try w.writeAll(", ");
                try w.writeAll("[");
                try p[0].scale(iw).write(w);
                try w.writeAll(", ");
                try p[1].scale(ih).write(w);
                try w.writeAll("]");
            }
            try w.writeAll("]");
        } else try w.writeAll("null");
        if (extra) |e| try w.print(", \"occluded\": {s}, \"truncated\": {s}", .{ bstr(e.occluded), bstr(e.truncated) });
        try w.print(", \"split\": \"{s}\"}}}}", .{split});
        try group.annotations.appendSlice(gpa, a.written());

        if (skipped) group.tally.skipped += 1 else if (unordered != null) group.tally.side_unknown += 1 else group.tally.with_eyes += 1;
    }

    var so: u.Stdout = undefined;
    so.init(io);
    const sw = so.w();
    const Order = struct {
        keys: []const []const u8,
        pub fn lessThan(ctx: @This(), a: usize, b: usize) bool {
            return std.mem.lessThan(u8, ctx.keys[a], ctx.keys[b]);
        }
    };
    groups.sort(Order{ .keys = groups.keys() });
    for (groups.keys(), groups.values()) |target, group| {
        var d: std.Io.Writer.Allocating = .init(gpa);
        const w = &d.writer;
        try w.print("{{\"info\": {{\"description\": \"Open Images face boxes with eye keypoints clicked for Face Crop Studio\", " ++
            "\"source\": \"Open Images V7, /m/0dzct Human face\", \"keypoint_order\": \"RetinaFace/YuNet, in the subject's own frame\", " ++
            "\"role\": \"{s}\"}}, \"licenses\": [{{\"id\": 1, \"name\": \"CC BY 2.0\", \"url\": \"https://creativecommons.org/licenses/by/2.0/\"}}], \"images\": [", .{if (std.mem.eql(u8, target, "train")) "training labels" else "held-out test set"});
        for (group.images.values(), 0..) |im, i| {
            try w.print("{s}{{\"id\": {d}, \"file_name\": \"{s}.jpg\", \"width\": {d}, \"height\": {d}, \"license\": 1, \"open_images_id\": \"{s}\"}}", .{ if (i > 0) ", " else "", im.id, im.name, im.width, im.height, im.name });
        }
        try w.print("], \"annotations\": [{s}], \"categories\": [{{\"id\": 1, \"name\": \"face\", \"supercategory\": \"person\", " ++
            "\"keypoints\": [\"right_eye\", \"left_eye\", \"nose\", \"right_mouth\", \"left_mouth\"], \"skeleton\": [[1, 2]]}}]}}", .{group.annotations.items});
        const path = try std.fmt.allocPrint(gpa, "{s}/face_keypoints_{s}.json", .{ out_dir, target });
        try u.writeFile(io, path, d.written());
        try sw.print("{s}: {d} images, {d} annotations -> {s}\n", .{ target, group.images.count(), group.count, path });
        try sw.print("   both eyes, side known: {d}\n   points kept but side unknown (steep_roll): {d}\n   box only, eyes not visible: {d}\n", .{ group.tally.with_eyes, group.tally.side_unknown, group.tally.skipped });
    }
    if (dropped > 0) try sw.print("dropped, image missing or unreadable: {d}\n", .{dropped});
    try sw.flush();
}

fn bstr(b: bool) []const u8 {
    return if (b) "true" else "false";
}
