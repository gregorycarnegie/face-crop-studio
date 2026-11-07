//! Build `fixtures/images/`: a committed, licence-clean sample of the held-out benchmark set.
//!
//! The previous `fixtures/images/` was a local folder of photographs nobody here had a licence for,
//! so it could not be committed and CI never saw it. This replaces it with Open Images validation and
//! test photographs, every one CC BY 2.0, drawn from the same held-out set `docs/MODEL_COMPARISON.md`
//! scores (`labelv2_test_all.txt`: 1,859 images that SCRFD-80k never trained on).
//!
//! Selection is a rule, not a choice: images ordered by the SHA-256 of their Open Images ID, the eight
//! already in `fixtures/oracle/` skipped, the first `--count` taken. That is a uniform sample, so the
//! recall measured on it should sit near the full set's, and anyone can rebuild the same set.
//!
//! Each image is downscaled so its longer side is `--side` pixels and re-encoded, which keeps the set
//! small enough to commit. The detector letterboxes to 640 regardless, so it sees nearly the same
//! input. Boxes are scaled to match. Re-encoding also drops the EXIF block; the pixels are kept in
//! their stored orientation, which is what the boxes describe, and an image whose stored size does
//! not match its label is skipped rather than guessed at.
//!
//! Run from the repository root:
//!
//!   zig run -O ReleaseFast -lc tools/dataset/make_accuracy_fixtures.zig tools/dataset/vendor/stb.c -- \
//!       <Downloads/face-data/openimages> fixtures/images [--count 60] [--side 640] [--quality 85]
//!
//! The committed images were encoded by OpenCV (libjpeg); stb's encoder writes different bytes for
//! the same pixels, so a rebuild selects the same images and writes the same boxes and attribution,
//! but not byte-identical JPEGs.
const std = @import("std");
const Io = std.Io;
const json = std.json;
const u = @import("util.zig");
const img = @import("image.zig");

const Credit = struct { license: []const u8, title: []const u8, author: []const u8, profile: []const u8, landing: []const u8 };
const Entry = struct { width: usize, height: usize, faces: std.ArrayList([4]f64) = .empty, ignore: std.ArrayList([4]f64) = .empty };

/// labelv2 file -> {name: {size, faces, ignore}}. Five-value lines are ignore regions.
fn labels(gpa: std.mem.Allocator, text: []const u8) !std.StringArrayHashMapUnmanaged(Entry) {
    var out: std.StringArrayHashMapUnmanaged(Entry) = .empty;
    var current: ?*Entry = null;
    var it = std.mem.splitScalar(u8, text, '\n');
    while (it.next()) |raw| {
        const line = std.mem.trimEnd(u8, raw, "\r");
        var words = std.mem.tokenizeAny(u8, line, " \t");
        if (std.mem.startsWith(u8, line, "#")) {
            _ = words.next();
            const name = words.next() orelse return error.BadLabels;
            const w = try std.fmt.parseInt(usize, words.next() orelse return error.BadLabels, 10);
            const h = try std.fmt.parseInt(usize, words.next() orelse return error.BadLabels, 10);
            const gop = try out.getOrPut(gpa, name);
            gop.value_ptr.* = .{ .width = w, .height = h };
            current = gop.value_ptr;
        } else if (std.mem.trim(u8, line, " \t").len > 0) {
            var values: std.ArrayList(f64) = .empty;
            while (words.next()) |v| try values.append(gpa, try std.fmt.parseFloat(f64, v));
            if (values.items.len < 4) return error.BadLabels;
            const entry = current orelse return error.BadLabels;
            try (if (values.items.len == 5) &entry.ignore else &entry.faces).append(gpa, values.items[0..4].*);
        }
    }
    return out;
}

/// A 4x4 synthetic gradient that contains no face, made here so it needs no licence.
fn testPattern(gpa: std.mem.Allocator) !img.Image {
    const pattern = try img.Image.init(gpa, 4, 4);
    for (0..4) |y| for (0..4) |x| {
        // OpenCV's BGR (x*60, y*60, 255 - x*30), stored here as RGB.
        pattern.at(x, y).* = .{ @intCast(255 - x * 30), @intCast(y * 60), @intCast(x * 60) };
    };
    return pattern;
}

/// `json.dumps(v, indent=1)`.
fn dumpIndented(w: *Io.Writer, v: json.Value, depth: usize) !void {
    switch (v) {
        .array => |a| {
            if (a.items.len == 0) return w.writeAll("[]");
            try w.writeAll("[");
            for (a.items, 0..) |item, i| {
                try w.writeAll(if (i > 0) ",\n" else "\n");
                try w.splatByteAll(' ', depth + 1);
                try dumpIndented(w, item, depth + 1);
            }
            try w.writeByte('\n');
            try w.splatByteAll(' ', depth);
            try w.writeAll("]");
        },
        .object => |o| {
            if (o.count() == 0) return w.writeAll("{}");
            try w.writeAll("{");
            for (o.keys(), o.values(), 0..) |k, item, i| {
                try w.writeAll(if (i > 0) ",\n" else "\n");
                try w.splatByteAll(' ', depth + 1);
                try u.dumpString(w, k);
                try w.writeAll(": ");
                try dumpIndented(w, item, depth + 1);
            }
            try w.writeByte('\n');
            try w.splatByteAll(' ', depth);
            try w.writeAll("}");
        },
        else => try u.dump(w, v),
    }
}

fn scaled(gpa: std.mem.Allocator, boxes: []const [4]f64, scale: f64) !json.Value {
    var list: json.Array = .init(gpa);
    for (boxes) |box| {
        var row: json.Array = .init(gpa);
        for (box) |v| try row.append(.{ .float = u.pyRound(v * scale, 2) });
        try list.append(.{ .array = row });
    }
    return .{ .array = list };
}

fn hexDigest(name: []const u8) [64]u8 {
    var digest: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(name, &digest, .{});
    return std.fmt.bytesToHex(digest, .lower);
}

fn byDigest(_: void, a: []const u8, b: []const u8) bool {
    return std.mem.lessThan(u8, &hexDigest(a), &hexDigest(b));
}

pub fn main(init: std.process.Init) !void {
    const gpa = init.arena.allocator();
    const io = init.io;
    const args = try init.minimal.args.toSlice(gpa);
    var positional: std.ArrayList([]const u8) = .empty;
    var count: usize = 60;
    var side: usize = 640;
    var quality: u8 = 85;
    var i: usize = 1;
    while (i < args.len) : (i += 1) {
        const arg = args[i];
        if (std.mem.startsWith(u8, arg, "--") and i + 1 < args.len) {
            i += 1;
            const v = args[i];
            if (std.mem.eql(u8, arg, "--count")) count = try std.fmt.parseInt(usize, v, 10) else if (std.mem.eql(u8, arg, "--side")) side = try std.fmt.parseInt(usize, v, 10) else if (std.mem.eql(u8, arg, "--quality")) quality = try std.fmt.parseInt(u8, v, 10) else return error.UnknownOption;
        } else try positional.append(gpa, arg);
    }
    if (positional.items.len != 2) {
        std.debug.print("usage: make_accuracy_fixtures <openimages_dir> <out_dir> [--count 60] [--side 640] [--quality 85]\n", .{});
        return error.Usage;
    }
    const data = positional.items[0];
    const out_dir = positional.items[1];
    const cwd = Io.Dir.cwd();

    var truth = try labels(gpa, try u.readFile(gpa, io, try std.fs.path.join(gpa, &.{ data, "labelv2_test_all.txt" })));
    var credits: std.StringHashMapUnmanaged(Credit) = .empty;
    for ([_][]const u8{ "validation", "test" }) |split| {
        const csv = try u.Csv.load(gpa, io, try std.fmt.allocPrint(gpa, "{s}{c}{s}-images-with-rotation.csv", .{ data, std.fs.path.sep, split }));
        const c = [_]usize{ csv.col("License"), csv.col("Title"), csv.col("Author"), csv.col("AuthorProfileURL"), csv.col("OriginalLandingURL") };
        for (csv.rows) |row| try credits.put(gpa, row[csv.col("ImageID")], .{ .license = row[c[0]], .title = row[c[1]], .author = row[c[2]], .profile = row[c[3]], .landing = row[c[4]] });
    }

    var oracle: std.StringHashMapUnmanaged(void) = .empty;
    {
        var dir = try cwd.openDir(io, "fixtures/oracle/images", .{ .iterate = true });
        defer dir.close(io);
        var it = dir.iterate();
        while (try it.next(io)) |e| if (e.kind == .file and std.mem.endsWith(u8, e.name, ".jpg")) try oracle.put(gpa, try gpa.dupe(u8, e.name), {});
    }
    var order: std.ArrayList([]const u8) = .empty;
    for (truth.keys()) |name| if (!oracle.contains(name)) try order.append(gpa, name);
    std.mem.sort([]const u8, order.items, {}, byDigest);

    try cwd.createDirPath(io, out_dir);
    var chosen: json.ObjectMap = .empty;
    var chosen_names: std.ArrayList([]const u8) = .empty;
    var skipped: std.ArrayList([]const u8) = .empty;
    var faces: usize = 0;
    for (order.items) |name| {
        if (chosen_names.items.len == count) break;
        const entry = truth.get(name).?;
        var arena: std.heap.ArenaAllocator = .init(std.heap.smp_allocator);
        defer arena.deinit();
        const tmp = arena.allocator();
        var image = img.load(tmp, io, try std.fs.path.join(tmp, &.{ data, "images", name }), .as_stored) catch {
            try skipped.append(gpa, name);
            continue;
        };
        if (image.width != entry.width or image.height != entry.height) {
            try skipped.append(gpa, name);
            continue;
        }
        const scale = @min(1.0, @as(f64, @floatFromInt(side)) / @as(f64, @floatFromInt(@max(image.width, image.height))));
        if (scale < 1.0) {
            const w: usize = @intFromFloat(img.rint(@as(f64, @floatFromInt(image.width)) * scale));
            const h: usize = @intFromFloat(img.rint(@as(f64, @floatFromInt(image.height)) * scale));
            image = try img.resizeArea(tmp, image, w, h);
        }
        try u.writeFile(io, try std.fs.path.join(tmp, &.{ out_dir, name }), try img.encodeJpeg(tmp, image, quality));
        try chosen.put(gpa, name, try obj(gpa, &.{ "faces", "ignore" }, &.{ try scaled(gpa, entry.faces.items, scale), try scaled(gpa, entry.ignore.items, scale) }));
        try chosen_names.append(gpa, name);
        faces += entry.faces.items.len;
    }

    try u.writeFile(io, try std.fs.path.join(gpa, &.{ out_dir, "test_pattern.png" }), try img.encodePng(gpa, try testPattern(gpa)));

    var gt: Io.Writer.Allocating = .init(gpa);
    try dumpIndented(&gt.writer, try obj(gpa, &.{ "source", "rule", "images" }, &.{
        .{ .string = "labelv2_test_all.txt (Open Images validation+test, held out from SCRFD-80k training)" },
        .{ .string = try std.fmt.allocPrint(gpa, "first {d} by SHA-256 of the file name, excluding fixtures/oracle; longer side scaled to {d} px, boxes scaled to match", .{ count, side }) },
        .{ .object = chosen },
    }), 0);
    try gt.writer.writeByte('\n');
    try u.writeFile(io, try std.fs.path.join(gpa, &.{ out_dir, "ground_truth.json" }), gt.written());

    var md: Io.Writer.Allocating = .init(gpa);
    const w = &md.writer;
    try w.print(
        \\# Fixture images: attribution
        \\
        \\{d} Open Images photographs ({d} annotated faces), all licensed
        \\[CC BY 2.0](https://creativecommons.org/licenses/by/2.0/). Each was downscaled so its longer
        \\side is {d} px and re-encoded as JPEG (quality {d}), which also removed its
        \\metadata. `ground_truth.json` holds the Open Images face boxes, scaled to match.
        \\`test_pattern.png` is a synthetic gradient made by `tools/dataset/make_accuracy_fixtures.zig`.
        \\
        \\| File | Title | Author | Source |
        \\|---|---|---|---|
        \\
    , .{ chosen_names.items.len, faces, side, quality });
    for (chosen_names.items) |name| {
        const row = credits.get(name[0 .. name.len - ".jpg".len]) orelse return error.MissingCredit;
        if (!std.mem.eql(u8, row.license, "https://creativecommons.org/licenses/by/2.0/")) {
            std.debug.print("{s}: not CC BY 2.0\n", .{name});
            return error.LicenceNotCcBy;
        }
        const title = try std.mem.replaceOwned(u8, gpa, if (row.title.len > 0) row.title else "untitled", "|", "\\|");
        const author = try std.mem.replaceOwned(u8, gpa, if (row.author.len > 0) row.author else "unknown", "|", "\\|");
        try w.print("| `{s}` | {s} | [{s}]({s}) | [Flickr]({s}) |\n", .{ name, title, author, row.profile, row.landing });
    }
    try u.writeFile(io, try std.fs.path.join(gpa, &.{ out_dir, "ATTRIBUTION.md" }), md.written());

    var size: u64 = 0;
    {
        var dir = try cwd.openDir(io, out_dir, .{ .iterate = true });
        defer dir.close(io);
        var it = dir.iterate();
        while (try it.next(io)) |e| if (e.kind == .file and std.mem.endsWith(u8, e.name, ".jpg")) {
            size += (try dir.statFile(io, e.name, .{})).size;
        };
    }
    var so: u.Stdout = undefined;
    so.init(io);
    try so.w().print("{d} images, {d} faces, {s} MB; skipped {d}: [", .{ chosen_names.items.len, faces, try u.fixedAlloc(gpa, @as(f64, @floatFromInt(size)) / 1e6, 1), skipped.items.len });
    for (skipped.items, 0..) |s, k| try so.w().print("{s}'{s}'", .{ if (k > 0) ", " else "", s });
    try so.w().writeAll("]\n");
    try so.w().flush();
}

fn obj(gpa: std.mem.Allocator, keys: []const []const u8, values: []const json.Value) !json.Value {
    return .{ .object = try json.ObjectMap.init(gpa, keys, values) };
}

test "labelv2 parsing splits faces from five-value ignore regions" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const parsed = try labels(arena.allocator(), "# a.jpg 640 480\r\n1 2 3 4 0.0 1 1\n5 6 7 8 -1\n\n# b.jpg 10 20\n");
    const a = parsed.get("a.jpg").?;
    try std.testing.expectEqual(@as(usize, 480), a.height);
    try std.testing.expectEqual(@as(usize, 1), a.faces.items.len);
    try std.testing.expectEqual(@as(f64, 5), a.ignore.items[0][0]);
    try std.testing.expectEqual(@as(usize, 0), parsed.get("b.jpg").?.faces.items.len);
}

test "indent=1 output matches json.dumps" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const gpa = arena.allocator();
    var out: Io.Writer.Allocating = .init(gpa);
    const v = try obj(gpa, &.{ "a", "b", "c" }, &.{ try scaled(gpa, &.{.{ 1, 2.5, 3.125, 4 }}, 1), try scaled(gpa, &.{}, 1), .{ .string = "x" } });
    try dumpIndented(&out.writer, v, 0);
    try std.testing.expectEqualStrings("{\n \"a\": [\n  [\n   1.0,\n   2.5,\n   3.12,\n   4.0\n  ]\n ],\n \"b\": [],\n \"c\": \"x\"\n}", out.written());
}
