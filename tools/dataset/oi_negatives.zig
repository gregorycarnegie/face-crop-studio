//! Pick Open Images train photographs with no people in them, record their credits, download them.
//!
//!   zig run -O ReleaseSafe tools/dataset/oi_negatives.zig -- <openimages_dir> [--pool 100000] [--workers 8] [--select-only]
//!   zig run -O ReleaseSafe tools/dataset/oi_negatives.zig -- pick <openimages_dir> <scrfd.json> <retina.json> [--count 20000]
//!
//! Keep `--workers` low: Zig's HTTP client resolves the host itself for every connection, and 48
//! workers flooded a home router's DNS until name resolution failed for the whole network.
//!
//! `pick` chooses the training negatives from the downloaded pool, given SCRFD-80k's detections
//! over all of it (`scrfd_detect.zig --list`, threshold 0.2) and RetinaFace's over the images
//! SCRFD fired on (`detector_bench.zig retina`). An image where RetinaFace finds a face at 0.9 or
//! more is vetoed as probably holding a real person. Of the rest, every image SCRFD fires on at
//! 0.3 or more is a hard negative and goes in; easy ones, drawn with the same seed, fill up to
//! `--count`. Images listed in `negatives_rejected.txt` (a person's verdicts from the review page)
//! are left out like vetoed ones. It writes `labelv2_negatives.txt` (headers only, for `train_scrfd --negatives`),
//! `negatives_manifest.csv` (credits) and `negatives_review.json` with an empty partner
//! `negatives_review_none.json`, the highest-scoring SCRFD hits left in, for
//! `compare_detectors.zig` to show a person before training.
//!
//! SCRFD-80k trained on 80,000 photographs that all contain faces, and on COCO it fires 5-6x as
//! often as RetinaFace on photographs with nobody in them -- animal faces, clocks, plates, wheels
//! (MODEL_COMPARISON.md). Face-free photographs, trained as pure background (`train_scrfd
//! --negatives`), are the fix that costs no labelling. This picks the pool they are mined from.
//!
//! Inputs in `openimages_dir`, all official Open Images files:
//!   class-descriptions-boxable.csv, oidv6-train-annotations-bbox.csv,
//!   train-annotations-human-imagelabels-boxable.csv, train-images-boxable-with-rotation.csv
//!
//! An image is eligible when none of its boxes, and none of its human-verified *positive* image
//! labels, is a person class or a class that usually carries a depicted human face:
//!
//! * **People:** Person, Man, Woman, Boy, Girl and every "Human ..." part.
//! * **Depicted faces:** Doll, Sculpture, Poster, Billboard, Picture frame, Mirror, Television,
//!   Computer monitor, Tablet computer, Mobile phone. The 80k labels made depictions ignore
//!   regions; training them as background would contradict that.
//!
//! Animals, teddy bears and toys stay in on purpose: non-human faces are what the detector should
//! learn to reject.
//!
//! **And a human must have checked that nobody is there.** Lacking a person label is not enough:
//! in a random 1,000 of the images that pass the rules above, about half of SCRFD-80k's hits at
//! its shipped threshold were real people, never labelled because nobody asked about Person.
//! The image labels also record verified *absences* (confidence 0), and an image is eligible only
//! when Person, Human face or Human head was verified absent. In those, about 1 hit in 10 was a
//! real face, and the rest were the animals, masks and objects this is meant to catch. The mining
//! step removes most of that residue (another detector vetoes) and a person reviews the rest.
//! Same rules as the 80k set otherwise: train split, `Rotation` 0.0.
//!
//! Writes `negatives_pool_manifest.csv` (licence, author, author URL, landing page per image; CC BY
//! 2.0 needs the attribution) and downloads into `images/` beside the 80k set, skipping anything
//! already there. The shuffle is `random.Random(20261108)`, so the pool is reproducible.
const std = @import("std");
const Io = std.Io;
const u = @import("util.zig");
const fetch = @import("oi_train_fetch.zig");

const SEED = 20261108;
/// An eligible image has at least one of these verified absent by a human.
const VERIFIED_ABSENT = [_][]const u8{ "Person", "Human face", "Human head" };
const DEPICTIONS = [_][]const u8{ "Person", "Man", "Woman", "Boy", "Girl", "Doll", "Sculpture", "Poster", "Billboard", "Picture frame", "Mirror", "Television", "Computer monitor", "Tablet computer", "Mobile phone" };

fn excluded(name: []const u8) bool {
    if (std.mem.startsWith(u8, name, "Human ")) return true;
    for (DEPICTIONS) |d| if (std.mem.eql(u8, d, name)) return true;
    return false;
}

/// The MIDs (`/m/...`) of the classes `wanted` accepts. The file has no header: `MID,DisplayName`.
fn labelsWhere(gpa: std.mem.Allocator, io: Io, path: []const u8, comptime wanted: fn ([]const u8) bool) !fetch.StrSet {
    var out: fetch.StrSet = .empty;
    const csv = try u.Csv.parse(gpa, try u.readFile(gpa, io, path));
    // Csv takes the first row for a header; here it is a class like any other.
    if (csv.header.len >= 2 and wanted(csv.header[1])) try out.put(gpa, csv.header[0], {});
    for (csv.rows) |r| if (r.len >= 2 and wanted(r[1])) try out.put(gpa, r[0], {});
    return out;
}

fn checkedForPeople(name: []const u8) bool {
    for (VERIFIED_ABSENT) |v| if (std.mem.eql(u8, v, name)) return true;
    return false;
}

/// Images with a row naming one of `labels` in column 2 (LabelName). With `confidence`, only rows
/// whose column 3 equals it count: the human-verified image labels record presence as "1" and a
/// verified absence as "0". Scanned line by line: the box file is 2.3 GB, too big to split into rows.
fn imagesWith(gpa: std.mem.Allocator, io: Io, path: []const u8, labels: *const fetch.StrSet, confidence: ?[]const u8, into: *fetch.StrSet) !usize {
    const text = try u.readFile(gpa, io, path);
    var lines = std.mem.splitScalar(u8, text, '\n');
    _ = lines.next(); // header
    var rows: usize = 0;
    while (lines.next()) |raw| {
        const line = std.mem.trimEnd(u8, raw, "\r");
        if (line.len == 0) continue;
        rows += 1;
        var fields = std.mem.splitScalar(u8, line, ',');
        const image = fields.next() orelse return error.BadCsv;
        _ = fields.next() orelse return error.BadCsv; // Source
        const label = fields.next() orelse return error.BadCsv;
        if (!labels.contains(label)) continue;
        if (confidence) |want| if (!std.mem.eql(u8, fields.next() orelse return error.BadCsv, want)) continue;
        if (!into.contains(image)) try into.put(gpa, try gpa.dupe(u8, image), {});
    }
    return rows;
}

const VETO: f64 = 0.9; // RetinaFace score that marks a probable real face
const HARD: f64 = 0.3; // SCRFD-80k score that makes an image a hard negative
const REVIEW = 400; // highest-scoring hits shown for review

const Mined = struct { name: []const u8, width: i64, height: i64, best: f64, hits: std.json.Value };

fn pick(gpa: std.mem.Allocator, io: Io, w: *Io.Writer, dir: []const u8, scrfd_path: []const u8, retina_path: []const u8, count: usize) !void {
    // RetinaFace: {name: [{box, score, eyes}, ...]} over the images SCRFD fired on.
    const retina = (try u.readJson(gpa, io, retina_path)).object;
    var vetoed: fetch.StrSet = .empty;
    var it = retina.iterator();
    while (it.next()) |e| for (e.value_ptr.array.items) |d| {
        if (u.num(u.field(d, "score")) >= VETO) {
            try vetoed.put(gpa, e.key_ptr.*, {});
            break;
        }
    };

    // Images a person rejected on the review page: one file name per line, '#' starts a comment.
    if (u.readFile(gpa, io, try std.fs.path.join(gpa, &.{ dir, "negatives_rejected.txt" }))) |text| {
        var lines = std.mem.splitScalar(u8, text, '\n');
        while (lines.next()) |raw| {
            const line = std.mem.trim(u8, raw[0 .. std.mem.findScalar(u8, raw, '#') orelse raw.len], " \t\r");
            if (line.len > 0) try vetoed.put(gpa, line, {});
        }
    } else |_| {}

    // SCRFD-80k: [{image, width, height, detections: [{score, ...}]}] over the whole pool.
    var hard: std.ArrayList(Mined) = .empty;
    var easy: std.ArrayList(Mined) = .empty;
    var unchecked: usize = 0;
    for ((try u.readJson(gpa, io, scrfd_path)).array.items) |r| {
        const name = std.fs.path.basename(u.str(u.field(r, "image")));
        if (vetoed.contains(name)) continue;
        const dets = u.field(r, "detections");
        var best: f64 = 0;
        for (dets.array.items) |d| best = @max(best, u.num(u.field(d, "score")));
        const m: Mined = .{ .name = name, .width = u.field(r, "width").integer, .height = u.field(r, "height").integer, .best = best, .hits = dets };
        if (best >= HARD) {
            if (!retina.contains(name)) unchecked += 1; // a hard negative RetinaFace never saw
            try hard.append(gpa, m);
        } else try easy.append(gpa, m);
    }
    if (unchecked > 0) {
        std.debug.print("{d} hard negatives are missing from {s}; run RetinaFace over every image SCRFD fired on\n", .{ unchecked, retina_path });
        return error.UncheckedHardNegatives;
    }
    const byName = struct {
        fn less(_: void, a: Mined, b: Mined) bool {
            return std.mem.lessThan(u8, a.name, b.name);
        }
    }.less;
    std.mem.sort(Mined, hard.items, {}, byName);
    std.mem.sort(Mined, easy.items, {}, byName);
    var rng = u.PyRandom.init(SEED);
    rng.shuffle(Mined, hard.items);
    rng.shuffle(Mined, easy.items);
    const n_hard = @min(hard.items.len, count);
    const n_easy = @min(easy.items.len, count - n_hard);
    try w.print("vetoed by RetinaFace >= {d}: {d}; hard (SCRFD >= {d}): {d}; easy: {d}; picked {d} hard + {d} easy\n", .{ VETO, vetoed.count(), HARD, hard.items.len, easy.items.len, n_hard, n_easy });

    const chosen = try std.mem.concat(gpa, Mined, &.{ hard.items[0..n_hard], easy.items[0..n_easy] });
    std.mem.sort(Mined, chosen, {}, byName);
    var labels: Io.Writer.Allocating = .init(gpa);
    for (chosen) |m| try labels.writer.print("# {s} {d} {d}\n", .{ m.name, m.width, m.height });
    try u.writeFile(io, try std.fs.path.join(gpa, &.{ dir, "labelv2_negatives.txt" }), labels.written());

    var wanted: fetch.StrSet = .empty;
    for (chosen) |m| try wanted.put(gpa, m.name[0 .. std.mem.findScalarLast(u8, m.name, '.') orelse m.name.len], {});
    const pool = try u.Csv.load(gpa, io, try std.fs.path.join(gpa, &.{ dir, "negatives_pool_manifest.csv" }));
    var rows: std.ArrayList([]const []const u8) = .empty;
    for (pool.rows) |r| if (wanted.contains(r[0])) try rows.append(gpa, r);
    if (rows.items.len != chosen.len) return error.ManifestMismatch;
    try fetch.writeManifest(gpa, io, try std.fs.path.join(gpa, &.{ dir, "negatives_manifest.csv" }), rows.items);

    // The highest-scoring hits left in, as fcs-cli records, for compare_detectors to show.
    const Hit = struct { m: *const Mined, d: std.json.Value };
    var hits: std.ArrayList(Hit) = .empty;
    for (chosen) |*m| for (m.hits.array.items) |d| if (u.num(u.field(d, "score")) >= HARD) try hits.append(gpa, .{ .m = m, .d = d });
    std.mem.sort(Hit, hits.items, {}, struct {
        fn more(_: void, a: Hit, b: Hit) bool {
            return u.num(u.field(a.d, "score")) > u.num(u.field(b.d, "score"));
        }
    }.more);
    const shown = hits.items[0..@min(hits.items.len, REVIEW)];
    var review: Io.Writer.Allocating = .init(gpa);
    var none: Io.Writer.Allocating = .init(gpa);
    try review.writer.writeAll("[");
    try none.writer.writeAll("[");
    for (shown, 0..) |h, k| {
        if (k > 0) {
            try review.writer.writeAll(", ");
            try none.writer.writeAll(", ");
        }
        const path = try std.fs.path.join(gpa, &.{ dir, "images", h.m.name });
        try review.writer.writeAll("{\"image\": ");
        try u.dumpString(&review.writer, path);
        try review.writer.print(", \"detections\": [{f}]}}", .{std.json.fmt(h.d, .{})});
        try none.writer.writeAll("{\"image\": ");
        try u.dumpString(&none.writer, path);
        try none.writer.writeAll(", \"detections\": []}");
    }
    try review.writer.writeAll("]");
    try none.writer.writeAll("]");
    try u.writeFile(io, try std.fs.path.join(gpa, &.{ dir, "negatives_review.json" }), review.written());
    try u.writeFile(io, try std.fs.path.join(gpa, &.{ dir, "negatives_review_none.json" }), none.written());
    try w.print("wrote labelv2_negatives.txt and negatives_manifest.csv ({d} images); {d} of {d} hits >= {d} in negatives_review.json, lowest score {d:.3}\n", .{ chosen.len, shown.len, hits.items.len, HARD, if (shown.len > 0) u.num(u.field(shown[shown.len - 1].d, "score")) else 0 });
    try w.flush();
}

pub fn main(init: std.process.Init) !void {
    const gpa = init.arena.allocator();
    const io = init.io;
    const args = try init.minimal.args.toSlice(gpa);
    if (args.len >= 5 and std.mem.eql(u8, args[1], "pick")) {
        var count: usize = 20_000;
        if (args.len == 7 and std.mem.eql(u8, args[5], "--count")) count = try std.fmt.parseInt(usize, args[6], 10) else if (args.len != 5) return error.Usage;
        var so: u.Stdout = undefined;
        so.init(io);
        return pick(gpa, io, so.w(), args[2], args[3], args[4], count);
    }
    var data: ?[]const u8 = null;
    var pool_size: usize = 100_000;
    var workers: usize = 32;
    var select_only = false;
    var i: usize = 1;
    while (i < args.len) : (i += 1) {
        if (std.mem.eql(u8, args[i], "--pool") and i + 1 < args.len) {
            i += 1;
            pool_size = try std.fmt.parseInt(usize, args[i], 10);
        } else if (std.mem.eql(u8, args[i], "--workers") and i + 1 < args.len) {
            i += 1;
            workers = try std.fmt.parseInt(usize, args[i], 10);
        } else if (std.mem.eql(u8, args[i], "--select-only")) {
            select_only = true;
        } else data = args[i];
    }
    const dir = data orelse {
        std.debug.print("usage: oi_negatives <openimages_dir> [--pool 100000] [--workers 32] [--select-only]\n", .{});
        return error.Usage;
    };
    var so: u.Stdout = undefined;
    so.init(io);
    const w = so.w();

    const classes = try std.fs.path.join(gpa, &.{ dir, "class-descriptions-boxable.csv" });
    const labels = try labelsWhere(gpa, io, classes, excluded);
    const checked = try labelsWhere(gpa, io, classes, checkedForPeople);
    if (checked.count() != VERIFIED_ABSENT.len) return error.MissingPersonClasses;
    try w.print("excluded classes: {d}\n", .{labels.count()});
    try w.flush();
    const image_labels = try std.fs.path.join(gpa, &.{ dir, "train-annotations-human-imagelabels-boxable.csv" });
    var people: fetch.StrSet = .empty;
    const boxes = try imagesWith(gpa, io, try std.fs.path.join(gpa, &.{ dir, "oidv6-train-annotations-bbox.csv" }), &labels, null, &people);
    const by_boxes = people.count();
    const verified = try imagesWith(gpa, io, image_labels, &labels, "1", &people);
    var nobody: fetch.StrSet = .empty;
    _ = try imagesWith(gpa, io, image_labels, &checked, "0", &nobody);
    try w.print("{d} box rows and {d} verified labels read; {d} images excluded ({d} by a box, {d} more by a verified label); {d} with nobody verified\n", .{ boxes, verified, people.count(), by_boxes, people.count() - by_boxes, nobody.count() });
    try w.flush();

    var rows: std.StringArrayHashMapUnmanaged([]const []const u8) = .empty;
    const ic = try u.Csv.load(gpa, io, try std.fs.path.join(gpa, &.{ dir, "train-images-boxable-with-rotation.csv" }));
    const c_rot = ic.col("Rotation");
    var cols: [fetch.FIELDS.len]usize = undefined;
    for (fetch.FIELDS, 0..) |f, k| cols[k] = ic.col(f);
    var rotated: usize = 0;
    for (ic.rows) |r| {
        const image = r[cols[0]];
        if (people.contains(image) or !nobody.contains(image)) continue;
        if (!std.mem.eql(u8, r[c_rot], "0.0")) {
            rotated += 1;
            continue;
        }
        const picked = try gpa.alloc([]const u8, fetch.FIELDS.len);
        for (cols, 0..) |c, k| picked[k] = r[c];
        try rows.put(gpa, image, picked);
    }

    var pool: std.ArrayList([]const u8) = .empty;
    try pool.appendSlice(gpa, rows.keys());
    std.mem.sort([]const u8, pool.items, {}, u.lessStr);
    var rng = u.PyRandom.init(SEED);
    rng.shuffle([]const u8, pool.items);
    const chosen = pool.items[0..@min(pool.items.len, pool_size)];
    try w.print("{d} train images in the table; eligible {d} ({d} left out for rotation); pool {d}\n", .{ ic.rows.len, pool.items.len, rotated, chosen.len });

    const manifest = try std.fs.path.join(gpa, &.{ dir, "negatives_pool_manifest.csv" });
    var manifest_rows: std.ArrayList([]const []const u8) = .empty;
    for (chosen) |image| try manifest_rows.append(gpa, rows.get(image).?);
    try fetch.writeManifest(gpa, io, manifest, manifest_rows.items);
    try w.print("wrote {s}\n", .{manifest});
    try w.flush();
    if (select_only) return;

    const dest = try std.fs.path.join(gpa, &.{ dir, "images" });
    try Io.Dir.cwd().createDirPath(io, dest);
    var job: fetch.Job = .{ .io = io, .chosen = chosen, .dest = dest, .out = &so };
    const threads = try gpa.alloc(std.Thread, @min(workers, chosen.len));
    for (threads) |*t| t.* = try std.Thread.spawn(.{}, fetch.worker, .{&job});
    for (threads) |t| t.join();

    // Photos removed from the mirror are expected; they leave the manifest rather than fail the run.
    if (job.failed.items.len > 0) {
        var log: Io.Writer.Allocating = .init(gpa);
        var gone: fetch.StrSet = .empty;
        for (job.failed.items) |f| {
            try log.writer.print("{s}\t{s}\n", .{ f.image, f.err });
            try gone.put(gpa, f.image, {});
        }
        try u.writeFile(io, try std.fs.path.join(gpa, &.{ dir, "negatives_fetch_failed.txt" }), log.written());
        var kept: std.ArrayList([]const []const u8) = .empty;
        for (manifest_rows.items) |r| if (!gone.contains(r[0])) try kept.append(gpa, r);
        try fetch.writeManifest(gpa, io, manifest, kept.items);
    }
    try w.print("finished: {d} images on disk, {d} unavailable\n", .{ chosen.len - job.failed.items.len, job.failed.items.len });
    try w.flush();
}

test "person and depicted-face classes are excluded; animals and toys are not" {
    for ([_][]const u8{ "Person", "Woman", "Human face", "Human hand", "Doll", "Sculpture", "Television", "Mirror" }) |n| try std.testing.expect(excluded(n));
    for ([_][]const u8{ "Dog", "Clock", "Teddy bear", "Toy", "Mammal", "Animal", "Plate", "Humanoid" }) |n| try std.testing.expect(!excluded(n));
}
