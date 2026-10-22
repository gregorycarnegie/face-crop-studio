//! Turn the COCO keypoint files into SCRFD's labelv2.txt annotation format.
//!
//!   zig run tools/dataset/to_labelv2.zig -- <data_dir> [out_dir]
//!
//! Reads face_keypoints_{train,test}.json and writes labelv2_{train,test}.txt beside them, the
//! format `RetinaFaceDataset` in SCRFD's vendored mmdet parses. Point `img_prefix` at the image
//! directory and `ann_file` at one of these.
//!
//! Layout, one image at a time:
//!
//!     # <file name> <width> <height>
//!     x1 y1 x2 y2  kp1x kp1y w1  kp2x kp2y w2  ...  kp5x kp5y w5
//!
//! Boxes are corners here, not COCO's x/y/width/height. A keypoint written as `-1 -1 -1` is
//! ignored by the landmark loss while the box still trains, which is what makes partial labels
//! expressible: this corpus has eyes and never nose or mouth, and faces flagged `steep_roll` have
//! eyes whose left/right identity is unknown, so those are written as ignored too rather than
//! guessed at. `_parse_ann_line` rewrites any surviving weight to 1.0, so the value written for a
//! present point only has to be non-negative.
const std = @import("std");
const u = @import("util.zig");

const NK = 5; // SCRFD's landmark count: right eye, left eye, nose, right mouth, left mouth
const IGNORED = "-1 -1 -1";

const Counts = struct { images: usize = 0, faces: usize = 0, with_points: usize = 0, points_ignored: usize = 0 };

const Image = struct { name: []const u8, width: i64, height: i64, id: i64 };

fn convert(gpa: std.mem.Allocator, io: std.Io, coco_path: []const u8, out_path: []const u8) !Counts {
    const doc = try u.readJson(gpa, io, coco_path);
    var by_image: std.AutoArrayHashMapUnmanaged(i64, std.ArrayList(std.json.Value)) = .empty;
    for (u.field(doc, "annotations").array.items) |a| {
        const entry = try by_image.getOrPut(gpa, u.field(a, "image_id").integer);
        if (!entry.found_existing) entry.value_ptr.* = .empty;
        try entry.value_ptr.append(gpa, a);
    }
    var images: std.ArrayList(Image) = .empty;
    for (u.field(doc, "images").array.items) |im| try images.append(gpa, .{
        .name = u.str(u.field(im, "file_name")),
        .width = u.field(im, "width").integer,
        .height = u.field(im, "height").integer,
        .id = u.field(im, "id").integer,
    });
    std.mem.sort(Image, images.items, {}, struct {
        fn lt(_: void, a: Image, b: Image) bool {
            return std.mem.lessThan(u8, a.name, b.name);
        }
    }.lt);

    var counts: Counts = .{};
    var out: std.Io.Writer.Allocating = .init(gpa);
    const w = &out.writer;
    for (images.items) |im| {
        const faces = (by_image.get(im.id) orelse continue).items;
        if (faces.len == 0) continue;
        counts.images += 1;
        try w.print("# {s} {d} {d}\n", .{ im.name, im.width, im.height });
        for (faces) |a| {
            counts.faces += 1;
            const bbox = u.field(a, "bbox").array.items;
            const x, const y, const bw, const bh = .{ u.num(bbox[0]), u.num(bbox[1]), u.num(bbox[2]), u.num(bbox[3]) };
            for ([_]f64{ x, y, x + bw, y + bh }, 0..) |v, i| {
                if (i > 0) try w.writeAll(" ");
                try u.fixed(w, v, 2);
            }
            const kp = u.field(a, "keypoints").array.items;
            var labelled = false;
            for (0..NK) |i| {
                if (u.num(kp[i * 3 + 2]) > 0) {
                    try w.writeAll(" ");
                    try u.fixed(w, u.num(kp[i * 3]), 2);
                    try w.writeAll(" ");
                    try u.fixed(w, u.num(kp[i * 3 + 1]), 2);
                    try w.writeAll(" 1");
                    labelled = true;
                } else {
                    try w.print(" {s}", .{IGNORED});
                    counts.points_ignored += 1;
                }
            }
            try w.writeAll("\n");
            counts.with_points += @intFromBool(labelled);
        }
    }
    if (out.written().len == 0) try w.writeAll("\n");
    try u.writeFile(io, out_path, out.written());
    return counts;
}

pub fn main(init: std.process.Init) !void {
    const gpa = init.arena.allocator();
    const args = try init.minimal.args.toSlice(gpa);
    if (args.len < 2) return error.Usage;
    const data = args[1];
    const out_dir = if (args.len > 2) args[2] else data;
    var so: u.Stdout = undefined;
    so.init(init.io);
    const w = so.w();
    for ([_][]const u8{ "train", "test" }) |split| {
        const coco_path = try std.fmt.allocPrint(gpa, "{s}/face_keypoints_{s}.json", .{ data, split });
        std.Io.Dir.cwd().access(init.io, coco_path, .{}) catch {
            try w.print("no face_keypoints_{s}.json, skipping\n", .{split});
            continue;
        };
        const out_path = try std.fmt.allocPrint(gpa, "{s}/labelv2_{s}.txt", .{ out_dir, split });
        const c = try convert(gpa, init.io, coco_path, out_path);
        try w.print("{s}: {d} images, {d} faces ({d} with eye points, {d} individual keypoints ignored) -> {s}\n", .{ split, c.images, c.faces, c.with_points, c.points_ignored, out_path });
    }
    try w.flush();
}
