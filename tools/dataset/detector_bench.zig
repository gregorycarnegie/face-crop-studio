//! Score face detectors on COCO 2017, a labelled corpus none of them was trained on.
//!
//!   zig run -O ReleaseFast -lc tools/dataset/detector_bench.zig tools/dataset/vendor/stb.c -- fetch <annotations_trainval2017.zip> <out_dir> [--negatives 2000]
//!   zig run ... -- oi <labelv2_test_all.txt> <images_dir> <models_dir> <cache_dir>
//!   zig run ... -- coco <out_dir from fetch> <models_dir> <cache_dir>
//!
//! docs/MODEL_COMPARISON.md compared SCRFD-80k, RetinaFace R34 and the public SCRFD-500M on held-out
//! Open Images -- the domain SCRFD-80k was trained on, which can favour it. COCO is the second
//! domain: SCRFD-80k trained only on Open Images, the other two on WIDER FACE, so here none of the
//! three is at home. `oi` re-runs the October comparison with this harness's decoders; matching its
//! recall at the same budgets is what makes the `coco` numbers trustworthy.
//!
//! COCO has no face boxes, but its person keypoints carry both eyes, placed by hand. A face counts
//! as found when a detection contains both of its eye points and is no wider than 8x their
//! separation (so one giant box cannot claim a crowd); matching is one-to-one in score order. False
//! positives come from two places: images with no person in them at all, where any detection is
//! wrong, and detections in person images landing outside every person box. A detection on a
//! person but on no labelled face -- a profile, a back of a head, a hand -- is neither, and is left
//! out. Only commercially usable image licences are fetched (NonCommercial excluded, as in
//! DATA_CARD.md); evaluation makes no derivative, so ShareAlike and NoDerivs images are usable.
//!
//! Thresholds are the October ones, frozen before COCO was scored. The mismatch-budget table that
//! follows them selects thresholds on COCO itself and is descriptive only.
//!
//! Models (models_dir): scrfd80k_500m_640.onnx, eye_refiner.onnx, retinaface_r34.onnx and
//! scrfd_500m.onnx, the last two from huggingface.co/yakhyo/uniface-weights; their SHA-256 are in
//! MODEL_COMPARISON.md and are checked here. ONNX Runtime comes from ORT_DYLIB_PATH (`--ort` too).
//!
//! The Python original decoded with OpenCV; this decodes with stb (`image.zig`), whose JPEG pixels
//! differ by a level or two, so fresh detections move slightly. Re-scoring from an existing cache
//! reproduces the Python tables exactly.
const std = @import("std");
const Io = std.Io;
const json = std.json;
const u = @import("util.zig");
const img = @import("image.zig");
const ort = @import("ort.zig");
const detector = @import("scrfd_detect.zig");

const SIZE = 640;
const FLOOR: f32 = 0.02; // proposal floor, as in October
const BUDGETS = [_]f64{ 0.05, 0.1, 0.2, 0.5 };
// COCO's no-person images hold few detections at the frozen thresholds (0.005-0.03 each), so
// its budgets start lower than October's.
const COCO_BUDGETS = [_]f64{ 0.01, 0.02, 0.05, 0.1 };
const BUCKETS = [_][]const u8{ "0-8", "8-16", "16-32", "32+" };
const BUCKET_LO = [_]f64{ 0, 8, 16, 32 };

const Kind = enum { scrfd, retinaface };
const Spec = struct { label: []const u8, file: []const u8, kind: Kind, refined: bool, frozen: f64 };
/// FROZEN, in its order; `refined` marks the two detectors the crop pipeline was measured with.
const MODELS = [_]Spec{
    .{ .label = "SCRFD-80k", .file = "scrfd80k_500m_640.onnx", .kind = .scrfd, .refined = true, .frozen = 0.4851 },
    .{ .label = "RetinaFace R34", .file = "retinaface_r34.onnx", .kind = .retinaface, .refined = true, .frozen = 0.9807 },
    .{ .label = "Public SCRFD-500M", .file = "scrfd_500m.onnx", .kind = .scrfd, .refined = false, .frozen = 0.6415 },
};
const SHA256 = [_][2][]const u8{
    .{ "scrfd80k_500m_640.onnx", "2437def9510f2af6431a5b2090ae9f357bc189cb0223be1b6c4d4ff085798e64" },
    .{ "retinaface_r34.onnx", "bd0263dc2a465d32859555cb1741f2d98991eb0053696e8ee33fec583d30e630" },
    .{ "scrfd_500m.onnx", "5e4447f50245bbd7966bd6c0fa52938c61474a04ec7def48753668a9d8b4ea3a" },
    .{ "eye_refiner.onnx", "16b4ed6e42889c0eb3d6ce5068e399c19f12ad686bd121bcf959db34f8cef976" },
};

const Det = struct { box: [4]f64, score: f64, eyes: [4]f64, refined: ?[4]f64 = null };
const ByImage = std.StringArrayHashMapUnmanaged([]const Det);
const Row = detector.Row; // box, score, five points
const nms = detector.nms;

fn fail(comptime fmt: []const u8, args: anytype) noreturn {
    std.debug.print(fmt ++ "\n", args);
    std.process.exit(1);
}

// ── Models ────────────────────────────────────────────────────────────────────

const Model = struct {
    oracle: ort.Oracle,
    input: [:0]const u8,
    outputs: []const [:0]const u8,
    priors: []const [4]f32 = &.{},

    fn open(gpa: std.mem.Allocator, io: Io, runtime: []const u8, dir: []const u8, file: []const u8) !Model {
        const data = try u.readFile(gpa, io, try std.fs.path.join(gpa, &.{ dir, file }));
        var digest: [32]u8 = undefined;
        std.crypto.hash.sha2.Sha256.hash(data, &digest, .{});
        const hex = std.fmt.bytesToHex(digest, .lower);
        const want = for (SHA256) |e| {
            if (std.mem.eql(u8, e[0], file)) break e[1];
        } else unreachable;
        if (!std.mem.eql(u8, &hex, want)) fail("{s}: sha256 {s} is not the model MODEL_COMPARISON.md measured", .{ file, &hex });
        const names = try ort.graphNames(gpa, data);
        return .{ .oracle = try ort.Oracle.open(gpa, runtime, data), .input = names.inputs[0], .outputs = names.outputs };
    }
};

/// Top-left into a 640 square, aspect kept -- SCRFD.detect's preprocessing, used for all.
fn letterbox(gpa: std.mem.Allocator, image: img.Image) !struct { img.Image, f64 } {
    const scale = @as(f64, SIZE) / @as(f64, @floatFromInt(@max(image.width, image.height)));
    const width: usize = @intFromFloat(@trunc(@as(f64, @floatFromInt(image.width)) * scale));
    const height: usize = @intFromFloat(@trunc(@as(f64, @floatFromInt(image.height)) * scale));
    const resized = try img.resizeLinear(gpa, image, width, height);
    defer resized.deinit(gpa);
    const canvas = try img.Image.init(gpa, SIZE, SIZE);
    for (0..height) |y| @memcpy(canvas.pixels[y * SIZE * 3 ..][0 .. width * 3], resized.pixels[y * width * 3 ..][0 .. width * 3]);
    return .{ canvas, scale };
}

/// CHW floats, `(x - mean) * scale` per channel; `order` picks RGB (0,1,2) or BGR (2,1,0).
fn blob(gpa: std.mem.Allocator, image: img.Image, order: [3]usize, mean: [3]f32, scale: f32) ![]f32 {
    const plane = image.width * image.height;
    const out = try gpa.alloc(f32, 3 * plane);
    for (0..3) |c| for (0..plane) |p| {
        out[c * plane + p] = (@as(f32, @floatFromInt(image.pixels[p * 3 + order[c]])) - mean[c]) * scale;
    };
    return out;
}

/// Either SCRFD export: nine tensors, scores already sigmoided, distances in stride units.
fn scrfd(gpa: std.mem.Allocator, model: *Model, canvas: img.Image) ![]Row {
    const input = try blob(gpa, canvas, .{ 0, 1, 2 }, .{ 127.5, 127.5, 127.5 }, 1.0 / 128.0);
    defer gpa.free(input);
    const outputs = try model.oracle.runAll(gpa, input, &.{ 1, 3, SIZE, SIZE }, model.input, model.outputs);
    if (outputs.len != 9) return error.UnexpectedOutputs;
    var rows: std.ArrayList(Row) = .empty;
    for ([_]usize{ 8, 16, 32 }, 0..) |stride, level| {
        const side = SIZE / stride;
        const scores = outputs[level].data;
        const boxes = outputs[level + 3].data;
        const points = outputs[level + 6].data;
        if (scores.len != side * side * 2 or boxes.len != scores.len * 4 or points.len != scores.len * 10) return error.UnexpectedOutputs;
        const s: f32 = @floatFromInt(stride);
        for (scores, 0..) |score, j| {
            if (!(score >= FLOOR)) continue;
            const cell = j / 2; // two anchors per location
            const cx: f32 = @floatFromInt((cell % side) * stride);
            const cy: f32 = @floatFromInt((cell / side) * stride);
            const b = boxes[j * 4 ..][0..4];
            var row: Row = undefined;
            row[0] = cx - b[0] * s;
            row[1] = cy - b[1] * s;
            row[2] = cx + b[2] * s;
            row[3] = cy + b[3] * s;
            row[4] = score;
            for (0..5) |p| {
                row[5 + 2 * p] = cx + points[j * 10 + 2 * p] * s;
                row[6 + 2 * p] = cy + points[j * 10 + 2 * p + 1] * s;
            }
            try rows.append(gpa, row);
        }
    }
    return rows.toOwnedSlice(gpa);
}

fn retinaPriors(gpa: std.mem.Allocator) ![]const [4]f32 {
    var priors: std.ArrayList([4]f32) = .empty;
    for ([_]usize{ 8, 16, 32 }, [_][2]f64{ .{ 16, 32 }, .{ 64, 128 }, .{ 256, 512 } }) |step, sizes| {
        const side = SIZE / step;
        for (0..side) |y| for (0..side) |x| {
            const cx = (@as(f64, @floatFromInt(x)) + 0.5) * @as(f64, @floatFromInt(step)) / SIZE;
            const cy = (@as(f64, @floatFromInt(y)) + 0.5) * @as(f64, @floatFromInt(step)) / SIZE;
            for (sizes) |size| try priors.append(gpa, .{ @floatCast(cx), @floatCast(cy), @floatCast(size / SIZE), @floatCast(size / SIZE) });
        };
    }
    return priors.toOwnedSlice(gpa);
}

/// UniFace's RetinaFace: BGR minus the Caffe means, prior boxes, variances 0.1 and 0.2.
fn retinaface(gpa: std.mem.Allocator, model: *Model, canvas: img.Image) ![]Row {
    const input = try blob(gpa, canvas, .{ 2, 1, 0 }, .{ 104, 117, 123 }, 1);
    defer gpa.free(input);
    const outputs = try model.oracle.runAll(gpa, input, &.{ 1, 3, SIZE, SIZE }, model.input, model.outputs);
    if (outputs.len != 3) return error.UnexpectedOutputs;
    const loc, const conf, const marks = .{ outputs[0].data, outputs[1].data, outputs[2].data };
    const n = model.priors.len;
    if (loc.len != n * 4 or conf.len != n * 2 or marks.len != n * 10) return error.UnexpectedOutputs;
    var rows: std.ArrayList(Row) = .empty;
    for (model.priors, 0..) |p, k| {
        if (!(conf[k * 2 + 1] >= FLOOR)) continue;
        const l = loc[k * 4 ..][0..4];
        const cx = p[0] + l[0] * @as(f32, 0.1) * p[2];
        const cy = p[1] + l[1] * @as(f32, 0.1) * p[3];
        const w = p[2] * @exp(l[2] * @as(f32, 0.2));
        const h = p[3] * @exp(l[3] * @as(f32, 0.2));
        var row: Row = undefined;
        row[0] = (cx - w / 2) * SIZE;
        row[1] = (cy - h / 2) * SIZE;
        row[2] = (cx + w / 2) * SIZE;
        row[3] = (cy + h / 2) * SIZE;
        row[4] = conf[k * 2 + 1];
        for (0..5) |q| {
            row[5 + 2 * q] = (p[0] + marks[k * 10 + 2 * q] * @as(f32, 0.1) * p[2]) * SIZE;
            row[6 + 2 * q] = (p[1] + marks[k * 10 + 2 * q + 1] * @as(f32, 0.1) * p[3]) * SIZE;
        }
        try rows.append(gpa, row);
    }
    return rows.toOwnedSlice(gpa);
}

/// fcs-core::eye_refiner: a 112 crop of the box grown 1.25x, eyes as fractions. NumPy's scalar
/// promotion is followed: float32 box values, float64 wherever a Python number joins in.
fn refine(gpa: std.mem.Allocator, model: *Model, image: img.Image, box: [4]f32) ![4]f64 {
    const cx = @as(f64, box[0] + box[2]) / 2;
    const cy = @as(f64, box[1] + box[3]) / 2;
    const scale = 112 / (@as(f64, @max(box[2] - box[0], box[3] - box[1])) * 1.25);
    const f = struct {
        fn f32of(v: f64) f64 {
            return @as(f32, @floatCast(v));
        }
    }.f32of; // the matrix is np.float32
    const crop = try img.warpAffine(gpa, image, .{ f(scale), 0, f(56 - cx * scale), 0, f(scale), f(56 - cy * scale) }, 112, 112);
    defer crop.deinit(gpa);
    const input = try blob(gpa, crop, .{ 0, 1, 2 }, .{ 127.5, 127.5, 127.5 }, 1.0 / 128.0);
    defer gpa.free(input);
    const eyes = try model.oracle.run(gpa, input, &.{ 1, 3, 112, 112 }, model.input, model.outputs[0]);
    if (eyes.data.len != 4) return error.UnexpectedOutputs;
    var out: [4]f64 = undefined;
    for (&out, 0..) |*o, i| o.* = (@as(f64, eyes.data[i] * 112) - 56) / scale + (if (i % 2 == 0) cx else cy);
    return out;
}

fn writeDets(w: *Io.Writer, out: ByImage) !void {
    try w.writeAll("{");
    for (out.keys(), out.values(), 0..) |name, dets, n| {
        if (n > 0) try w.writeAll(", ");
        try u.dumpString(w, name);
        try w.writeAll(": [");
        for (dets, 0..) |d, k| {
            if (k > 0) try w.writeAll(", ");
            try w.writeAll("{\"box\": ");
            try floats(w, &d.box);
            try w.writeAll(", \"score\": ");
            try u.pyFloat(w, d.score);
            try w.writeAll(", \"eyes\": ");
            try floats(w, &d.eyes);
            if (d.refined) |r| {
                try w.writeAll(", \"refined\": ");
                try floats(w, &r);
            }
            try w.writeAll("}");
        }
        try w.writeAll("]");
    }
    try w.writeAll("}");
}

fn floats(w: *Io.Writer, values: []const f64) !void {
    try w.writeAll("[");
    for (values, 0..) |v, i| {
        if (i > 0) try w.writeAll(", ");
        try u.pyFloat(w, v);
    }
    try w.writeAll("]");
}

fn four(v: json.Value) [4]f64 {
    const a = v.array.items;
    return .{ u.num(a[0]), u.num(a[1]), u.num(a[2]), u.num(a[3]) };
}

fn readDets(gpa: std.mem.Allocator, io: Io, path: []const u8) !ByImage {
    const doc = try u.readJson(gpa, io, path);
    var out: ByImage = .empty;
    for (doc.object.keys(), doc.object.values()) |name, list| {
        const dets = try gpa.alloc(Det, list.array.items.len);
        for (dets, list.array.items) |*d, v| d.* = .{
            .box = four(u.field(v, "box")),
            .score = u.num(u.field(v, "score")),
            .eyes = four(u.field(v, "eyes")),
            .refined = if (u.optField(v, "refined")) |r| four(r) else null,
        };
        try out.put(gpa, name, dets);
    }
    return out;
}

const Ctx = struct { gpa: std.mem.Allocator, io: Io, runtime: []const u8, out: *u.Stdout };

/// Every model over every image at the proposal floor, cached; refined eyes for the two
/// detectors the crop pipeline was measured with, above their frozen thresholds.
fn detectAll(ctx: Ctx, names: []const []const u8, images_dir: []const u8, models_dir: []const u8, cache: []const u8) ![MODELS.len]ByImage {
    const gpa = ctx.gpa;
    try Io.Dir.cwd().createDirPath(ctx.io, cache);
    var results: [MODELS.len]ByImage = undefined;
    for (MODELS, &results) |spec, *result| {
        const label = try gpa.dupe(u8, spec.label);
        std.mem.replaceScalar(u8, label, ' ', '_');
        const path = try std.fmt.allocPrint(gpa, "{s}{c}{s}.json", .{ cache, std.fs.path.sep, label });
        if (Io.Dir.cwd().access(ctx.io, path, .{})) |_| {
            result.* = try readDets(gpa, ctx.io, path);
            continue;
        } else |_| {}
        var model = try Model.open(gpa, ctx.io, ctx.runtime, models_dir, spec.file);
        if (spec.kind == .retinaface) model.priors = try retinaPriors(gpa);
        var refiner = try Model.open(gpa, ctx.io, ctx.runtime, models_dir, "eye_refiner.onnx");
        var out: ByImage = .empty;
        for (names, 1..) |name, index| {
            var arena: std.heap.ArenaAllocator = .init(std.heap.smp_allocator);
            defer arena.deinit();
            const tmp = arena.allocator();
            const image = img.load(tmp, ctx.io, try std.fs.path.join(tmp, &.{ images_dir, name }), .apply_exif) catch continue;
            const canvas, const scale = try letterbox(tmp, image);
            const rows = try switch (spec.kind) {
                .scrfd => scrfd(tmp, &model, canvas),
                .retinaface => retinaface(tmp, &model, canvas),
            };
            const s: f32 = @floatCast(scale);
            for (rows) |*r| for (0..15) |c| if (c != 4) {
                r[c] /= s;
            };
            const kept = try nms(tmp, rows);
            const dets = try gpa.alloc(Det, kept.len);
            for (dets, kept) |*d, k| {
                const r = rows[k];
                d.* = .{ .box = .{ r[0], r[1], r[2], r[3] }, .score = r[4], .eyes = .{ r[5], r[6], r[7], r[8] } }; // viewer's left first, as fcs-core
                if (spec.refined and @as(f64, r[4]) >= spec.frozen) d.refined = try refine(tmp, &refiner, image, r[0..4].*);
            }
            try out.put(gpa, name, dets);
            if (index % 500 == 0 or index == names.len) {
                try ctx.out.w().print("  {s}: {d}/{d}\n", .{ spec.label, index, names.len });
                try ctx.out.w().flush();
            }
        }
        var text: Io.Writer.Allocating = .init(gpa);
        try writeDets(&text.writer, out);
        try u.writeFile(ctx.io, path, text.written());
        result.* = out;
    }
    return results;
}

// ── Scoring ───────────────────────────────────────────────────────────────────

fn iou(a: [4]f64, b: [4]f64) f64 {
    const ix = @max(0.0, @min(a[2], b[2]) - @max(a[0], b[0]));
    const iy = @max(0.0, @min(a[3], b[3]) - @max(a[1], b[1]));
    const inter = ix * iy;
    const union_ = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter;
    return if (union_ > 0) inter / union_ else 0.0;
}

/// The lowest threshold leaving at most `budget` unmatched detections per image.
fn budgetThreshold(gpa: std.mem.Allocator, unmatched: []const f64, images: usize, budget: f64) !f64 {
    const allowed: usize = @intFromFloat(@trunc(budget * @as(f64, @floatFromInt(images)) + 1e-9));
    const scores = try gpa.dupe(f64, unmatched);
    std.mem.sort(f64, scores, {}, std.sort.desc(f64));
    if (scores.len <= allowed) return 0.02;
    return std.math.nextAfter(f32, @floatCast(scores[allowed]), 1);
}

fn recallAt(matches: []const f64, total: usize, threshold: f64) f64 {
    if (total == 0) return 0;
    var hits: usize = 0;
    for (matches) |s| hits += @intFromBool(s >= threshold);
    return @as(f64, @floatFromInt(hits)) / @as(f64, @floatFromInt(total));
}

/// Eye-line tilt, subject's right eye to left: the angle face_cropper aligns.
fn eyeAngle(right: [2]f64, left: [2]f64) f64 {
    return std.math.atan2(left[1] - right[1], left[0] - right[0]);
}

fn angleError(truth: [2][2]f64, pred: [2][2]f64) f64 {
    const difference = @mod(@abs((eyeAngle(truth[0], truth[1]) - eyeAngle(pred[0], pred[1])) * (180.0 / std.math.pi)), 360);
    return @min(difference, 360 - difference);
}

fn dist(a: [2]f64, b: [2]f64) f64 {
    return std.math.hypot(a[0] - b[0], a[1] - b[1]);
}

/// `f"{v:>{width}.{n}f}"` and `f"{v:>{width}.{n}%}"`.
fn col(gpa: std.mem.Allocator, w: *Io.Writer, v: f64, width: usize, comptime n: u8, comptime percent: bool) !void {
    const s = try u.fixedAlloc(gpa, if (percent) v * 100 else v, n);
    try u.padLeft(w, if (percent) try std.fmt.allocPrint(gpa, "{s}%", .{s}) else s, width);
}

fn reprCol(gpa: std.mem.Allocator, w: *Io.Writer, v: f64, width: usize) !void {
    var text: Io.Writer.Allocating = .init(gpa);
    try u.pyFloat(&text.writer, v);
    try u.padLeft(w, text.written(), width);
}

fn lines(text: []const u8) std.mem.SplitIterator(u8, .scalar) {
    return std.mem.splitScalar(u8, text, '\n');
}

fn scoreOi(ctx: Ctx, labels_path: []const u8, images_dir: []const u8, models_dir: []const u8, cache: []const u8) !void {
    const gpa = ctx.gpa;
    var images: std.ArrayList([]const u8) = .empty;
    var faces: std.StringHashMapUnmanaged(std.ArrayList([4]f64)) = .empty;
    var ignore: std.StringHashMapUnmanaged(std.ArrayList([4]f64)) = .empty;
    var name: []const u8 = "";
    var it = lines(try u.readFile(gpa, ctx.io, labels_path));
    while (it.next()) |raw| {
        const line = std.mem.trimEnd(u8, raw, "\r");
        if (std.mem.startsWith(u8, line, "#")) {
            var words = std.mem.tokenizeAny(u8, line, " \t");
            _ = words.next();
            name = words.next() orelse return error.BadLabels;
            try images.append(gpa, name);
            try faces.put(gpa, name, .empty);
            try ignore.put(gpa, name, .empty);
        } else if (std.mem.trim(u8, line, " \t").len > 0) {
            var values: std.ArrayList(f64) = .empty;
            var words = std.mem.tokenizeAny(u8, line, " \t");
            while (words.next()) |v| try values.append(gpa, try std.fmt.parseFloat(f64, v));
            if (values.items.len < 4) return error.BadLabels;
            const box = values.items[0..4].*;
            try (if (values.items.len > 5) faces else ignore).getPtr(name).?.append(gpa, box);
        }
    }

    const results = try detectAll(ctx, images.items, images_dir, models_dir, cache);
    var total: usize = 0;
    var regions: usize = 0;
    for (images.items) |n| {
        total += faces.get(n).?.items.len;
        regions += ignore.get(n).?.items.len;
    }
    const w = ctx.out.w();
    try w.print("\nOpen Images held-out: {d} images, {d} faces, {d} ignore regions\n", .{ images.items.len, total, regions });
    try u.padRight(w, "model", 20);
    try w.writeAll(" ");
    try u.padLeft(w, "thresh@0.1", 10);
    try w.writeAll(" ");
    for (BUDGETS, 0..) |b, i| {
        if (i > 0) try w.writeAll(" ");
        try reprCol(gpa, w, b, 8);
    }
    try w.writeAll("\n");
    for (MODELS, results) |spec, by_image| {
        var matched: std.ArrayList(f64) = .empty;
        var unmatched: std.ArrayList(f64) = .empty;
        for (images.items) |n| {
            const truths = faces.get(n).?.items;
            const taken = try gpa.alloc(bool, truths.len);
            @memset(taken, false);
            for (by_image.get(n) orelse &.{}) |det| {
                var best: ?usize = null;
                var best_iou: f64 = 0.5;
                for (truths, 0..) |truth, i| {
                    const overlap = iou(det.box, truth);
                    if (!taken[i] and overlap >= best_iou) {
                        best = i;
                        best_iou = overlap;
                    }
                }
                if (best) |b| {
                    taken[b] = true;
                    try matched.append(gpa, det.score);
                } else {
                    const hit = for (ignore.get(n).?.items) |region| {
                        if (iou(det.box, region) > 0.3) break true;
                    } else false;
                    if (!hit) try unmatched.append(gpa, det.score);
                }
            }
        }
        try u.padRight(w, spec.label, 20);
        try w.writeAll(" ");
        try col(gpa, w, try budgetThreshold(gpa, unmatched.items, images.items.len, 0.1), 10, 4, false);
        try w.writeAll(" ");
        for (BUDGETS, 0..) |b, i| {
            if (i > 0) try w.writeAll(" ");
            try col(gpa, w, recallAt(matched.items, total, try budgetThreshold(gpa, unmatched.items, images.items.len, b)), 8, 1, true);
        }
        try w.writeAll("\n");
    }
    try w.writeAll("  recall at each unmatched-detections-per-image budget; October: SCRFD-80k 0.4851, 76.9% at 0.1\n");
    try w.flush();
}

const Face = struct { file: []const u8, index: usize, right: [2]f64, left: [2]f64, visible: bool, iod: f64 };
const Key = struct { file: []const u8, index: usize };
const KeyContext = struct {
    pub fn hash(_: KeyContext, k: Key) u64 {
        return std.hash.Wyhash.hash(k.index, k.file);
    }
    pub fn eql(_: KeyContext, a: Key, b: Key) bool {
        return a.index == b.index and std.mem.eql(u8, a.file, b.file);
    }
};
const Found = std.HashMapUnmanaged(Key, Det, KeyContext, 80);

fn bucket(iod: f64) usize {
    var b: usize = 0;
    while (b + 1 < BUCKET_LO.len and iod >= BUCKET_LO[b + 1]) b += 1;
    return b;
}

fn point(v: json.Value) [2]f64 {
    return .{ u.num(v.array.items[0]), u.num(v.array.items[1]) };
}

fn scoreCoco(ctx: Ctx, data: []const u8, models_dir: []const u8, cache: []const u8) !void {
    const gpa = ctx.gpa;
    const manifest = (try u.readJson(gpa, ctx.io, try std.fs.path.join(gpa, &.{ data, "manifest.json" }))).array.items;
    const names = try gpa.alloc([]const u8, manifest.len);
    for (names, manifest) |*n, r| n.* = u.str(u.field(r, "file"));
    const results = try detectAll(ctx, names, try std.fs.path.join(gpa, &.{ data, "images" }), models_dir, cache);

    var negatives: usize = 0;
    var people: usize = 0;
    var faces: std.ArrayList(Face) = .empty;
    for (manifest) |r| {
        if (u.field(r, "people").array.items.len == 0) {
            negatives += 1;
            continue;
        }
        people += 1;
        for (u.field(r, "faces").array.items, 0..) |f, i| try faces.append(gpa, .{
            .file = u.str(u.field(r, "file")),
            .index = i,
            .right = point(u.field(f, "right")),
            .left = point(u.field(f, "left")),
            .visible = u.field(f, "visible").bool,
            .iod = u.num(u.field(f, "iod")),
        });
    }

    var found: [MODELS.len]Found = @splat(.empty);
    var false_neg: [MODELS.len]std.ArrayList(f64) = @splat(.empty);
    var false_out: [MODELS.len]std.ArrayList(f64) = @splat(.empty);
    for (results, 0..) |by_image, m| {
        for (manifest) |record| {
            const file = u.str(u.field(record, "file"));
            const record_faces = u.field(record, "faces").array.items;
            const record_people = u.field(record, "people").array.items;
            const taken = try gpa.alloc(bool, record_faces.len);
            @memset(taken, false);
            for (by_image.get(file) orelse &.{}) |det| {
                const x1, const y1, const x2, const y2 = det.box;
                const centre = [2]f64{ (x1 + x2) / 2, (y1 + y2) / 2 };
                var best: ?struct { f64, usize } = null;
                for (record_faces, 0..) |f, i| {
                    const right = point(u.field(f, "right"));
                    const left = point(u.field(f, "left"));
                    const inside_r = x1 <= right[0] and right[0] <= x2 and y1 <= right[1] and right[1] <= y2;
                    const inside_l = x1 <= left[0] and left[0] <= x2 and y1 <= left[1] and left[1] <= y2;
                    if (taken[i] or !inside_r or !inside_l or !(x2 - x1 <= 8 * @max(u.num(u.field(f, "iod")), 1.0))) continue;
                    const d = dist(centre, .{ (right[0] + left[0]) / 2, (right[1] + left[1]) / 2 });
                    if (best == null or d < best.?[0]) best = .{ d, i }; // min((distance, index))
                }
                if (best) |b| {
                    taken[b[1]] = true;
                    try found[m].put(gpa, .{ .file = file, .index = b[1] }, det);
                } else if (record_people.len == 0) {
                    try false_neg[m].append(gpa, det.score);
                } else {
                    const on_person = for (record_people) |pv| {
                        const p = pv.array.items;
                        if (u.num(p[0]) <= centre[0] and centre[0] <= u.num(p[2]) and u.num(p[1]) <= centre[1] and centre[1] <= u.num(p[3])) break true;
                    } else false;
                    if (!on_person) try false_out[m].append(gpa, det.score);
                }
            }
        }
    }

    var counts: [BUCKETS.len]usize = @splat(0);
    for (faces.items) |f| counts[bucket(f.iod)] += 1;
    const w = ctx.out.w();
    try w.print("\nCOCO 2017, licence-usable: {d} images with {d} eye-labelled faces, {d} images with no person\n", .{ people, faces.items.len, negatives });
    try w.writeAll("faces by eye separation (px): ");
    for (BUCKETS, counts, 0..) |b, c, i| try w.print("{s}{s}: {d}", .{ if (i > 0) ", " else "", b, c });
    try w.writeAll("\n");

    const Recall = struct {
        fn of(faces_: []const Face, found_: *const Found, threshold: f64, wanted: ?[]const usize) f64 {
            var pool: usize = 0;
            var hits: usize = 0;
            for (faces_) |f| {
                if (wanted) |list| if (std.mem.findScalar(usize, list, bucket(f.iod)) == null) continue;
                pool += 1;
                if (found_.get(.{ .file = f.file, .index = f.index })) |d| hits += @intFromBool(d.score >= threshold);
            }
            return if (pool > 0) @as(f64, @floatFromInt(hits)) / @as(f64, @floatFromInt(pool)) else 0;
        }
    };
    const usable = [_]usize{ 1, 2, 3 }; // every bucket but 0-8

    try w.writeAll("\nAt the October thresholds, frozen before scoring:\n");
    try u.padRight(w, "model", 20);
    for ([_][]const u8{ "thresh", "recall", "iod>=8" } ++ BUCKETS ++ [_][]const u8{ "FP/neg", "FP/out" }) |h| {
        try w.writeAll(" ");
        try u.padLeft(w, h, 7);
    }
    try w.writeAll("\n");
    for (MODELS, 0..) |spec, m| {
        const t = spec.frozen;
        var neg: usize = 0;
        for (false_neg[m].items) |s| neg += @intFromBool(s >= t);
        var out: usize = 0;
        for (false_out[m].items) |s| out += @intFromBool(s >= t);
        try u.padRight(w, spec.label, 20);
        try w.writeAll(" ");
        try col(gpa, w, t, 7, 4, false);
        try w.writeAll(" ");
        try col(gpa, w, Recall.of(faces.items, &found[m], t, null), 7, 1, true);
        try w.writeAll(" ");
        try col(gpa, w, Recall.of(faces.items, &found[m], t, &usable), 7, 1, true);
        for (0..BUCKETS.len) |b| {
            try w.writeAll(" ");
            try col(gpa, w, Recall.of(faces.items, &found[m], t, &.{b}), 7, 1, true);
        }
        try w.writeAll(" ");
        try col(gpa, w, @as(f64, @floatFromInt(neg)) / @as(f64, @floatFromInt(negatives)), 7, 3, false);
        try w.writeAll(" ");
        try col(gpa, w, @as(f64, @floatFromInt(out)) / @as(f64, @floatFromInt(people)), 7, 3, false);
        try w.writeAll("\n");
    }
    try w.writeAll("  FP/neg: detections per no-person image. FP/out: per person image, outside every person box.\n");

    try w.writeAll("\nRecall (eyes >= 8 px apart) at a no-person-image FP budget, selected on COCO:\n");
    try u.padRight(w, "model", 20);
    for (COCO_BUDGETS) |b| {
        try w.writeAll(" ");
        try reprCol(gpa, w, b, 8);
    }
    try w.writeAll("\n");
    for (MODELS, 0..) |spec, m| {
        try u.padRight(w, spec.label, 20);
        for (COCO_BUDGETS) |b| {
            try w.writeAll(" ");
            try col(gpa, w, Recall.of(faces.items, &found[m], try budgetThreshold(gpa, false_neg[m].items, negatives, b), &usable), 8, 1, true);
        }
        try w.writeAll("\n");
    }

    // Eye line on faces every detector found at its frozen threshold, eyes clearly visible.
    var paired: std.ArrayList(Face) = .empty;
    for (faces.items) |f| {
        if (!f.visible or f.iod < 16) continue;
        const all = for (MODELS, 0..) |spec, m| {
            const d = found[m].get(.{ .file = f.file, .index = f.index }) orelse break false;
            if (!(d.score >= spec.frozen)) break false;
        } else true;
        if (all) try paired.append(gpa, f);
    }
    try w.print("\nEye-line error on {d} faces found by all three, both eyes visible, >= 16 px apart:\n", .{paired.items.len});
    try u.padRight(w, "pipeline", 36);
    for ([_][]const u8{ "median", "p90" }) |h| {
        try w.writeAll(" ");
        try u.padLeft(w, h, 7);
    }
    try w.writeAll(" ");
    try u.padLeft(w, "<=5 deg", 8);
    try w.writeAll(" ");
    try u.padLeft(w, "dist/iod", 9);
    try w.writeAll("\n");
    const Pipeline = struct { name: []const u8, model: usize, refined: bool };
    var pipelines: std.ArrayList(Pipeline) = .empty;
    for (MODELS, 0..) |spec, m| try pipelines.append(gpa, .{ .name = try std.fmt.allocPrint(gpa, "{s} raw landmarks", .{spec.label}), .model = m, .refined = false });
    try pipelines.append(gpa, .{ .name = "SCRFD-80k + eye refiner", .model = 0, .refined = true });
    try pipelines.append(gpa, .{ .name = "RetinaFace + eye refiner", .model = 1, .refined = true });
    for (pipelines.items) |p| {
        const errors = try gpa.alloc(f64, paired.items.len);
        const distances = try gpa.alloc(f64, paired.items.len);
        var within: usize = 0;
        for (paired.items, errors, distances) |f, *e, *d| {
            const det = found[p.model].get(.{ .file = f.file, .index = f.index }).?;
            const v = if (p.refined) det.refined orelse return error.MissingRefinedEyes else det.eyes;
            const pred = [2][2]f64{ .{ v[0], v[1] }, .{ v[2], v[3] } }; // viewer's left = subject's right
            e.* = angleError(.{ f.right, f.left }, pred);
            d.* = (dist(pred[0], f.right) + dist(pred[1], f.left)) / 2 / f.iod;
            within += @intFromBool(e.* <= 5);
        }
        std.mem.sort(f64, errors, {}, std.sort.asc(f64));
        try u.padRight(w, p.name, 36);
        try w.writeAll(" ");
        try col(gpa, w, try u.median(gpa, errors), 6, 2, false);
        try w.writeAll("d ");
        try col(gpa, w, errors[@intFromFloat(@trunc(0.9 * @as(f64, @floatFromInt(errors.len))))], 6, 2, false);
        try w.writeAll("d ");
        try col(gpa, w, @as(f64, @floatFromInt(within)) / @as(f64, @floatFromInt(errors.len)), 8, 1, true);
        try w.writeAll(" ");
        try col(gpa, w, try u.median(gpa, distances), 9, 3, false);
        try w.writeAll("\n");
    }
    try w.flush();
}

// ── Fetch ─────────────────────────────────────────────────────────────────────

/// `a + b` with Python's int/float distinction, so json.dumps prints what Python printed.
fn add(a: json.Value, b: json.Value) json.Value {
    if (a == .integer and b == .integer) return .{ .integer = a.integer + b.integer };
    return .{ .float = u.num(a) + u.num(b) };
}

fn obj(gpa: std.mem.Allocator, keys: []const []const u8, values: []const json.Value) !json.Value {
    return .{ .object = try json.ObjectMap.init(gpa, keys, values) };
}

fn arr(gpa: std.mem.Allocator, values: []const json.Value) !json.Value {
    var a: json.Array = .init(gpa);
    try a.appendSlice(values);
    return .{ .array = a };
}

const Download = struct { file: []const u8, url: []const u8 };
const Job = struct {
    io: Io,
    downloads: []const Download,
    dest: []const u8,
    next: std.atomic.Value(usize) = .init(0),
    done: std.atomic.Value(usize) = .init(0),
    mutex: Io.Mutex = .init,
    failed: std.ArrayList([]const u8) = .empty,
    out: *u.Stdout,
};

fn getOne(job: *Job, client: *std.http.Client, d: Download) ?[]const u8 {
    const gpa = std.heap.smp_allocator;
    const path = std.fs.path.join(gpa, &.{ job.dest, d.file }) catch return "OutOfMemory";
    defer gpa.free(path);
    if (Io.Dir.cwd().access(job.io, path, .{})) |_| return null else |_| {}
    var body: Io.Writer.Allocating = .init(gpa);
    defer body.deinit();
    const res = client.fetch(.{ .location = .{ .url = d.url }, .response_writer = &body.writer }) catch |e| return @errorName(e);
    if (res.status != .ok) return "HTTP error";
    const tmp = std.fmt.allocPrint(gpa, "{s}.part", .{path}) catch return "OutOfMemory";
    defer gpa.free(tmp);
    u.writeFile(job.io, tmp, body.written()) catch |e| return @errorName(e);
    Io.Dir.rename(Io.Dir.cwd(), tmp, Io.Dir.cwd(), path, job.io) catch |e| return @errorName(e); // never a truncated .jpg
    return null;
}

fn worker(job: *Job) void {
    var client: std.http.Client = .{ .allocator = std.heap.smp_allocator, .io = job.io };
    defer client.deinit();
    while (true) {
        const i = job.next.fetchAdd(1, .monotonic);
        if (i >= job.downloads.len) return;
        const err = getOne(job, &client, job.downloads[i]);
        job.mutex.lockUncancelable(job.io);
        defer job.mutex.unlock(job.io);
        if (err) |e| job.failed.append(std.heap.smp_allocator, std.fmt.allocPrint(std.heap.smp_allocator, "{s}: {s}", .{ job.downloads[i].file, e }) catch "OutOfMemory") catch {};
        const done = job.done.fetchAdd(1, .monotonic) + 1;
        if (done % 1000 == 0 or done == job.downloads.len) {
            job.out.w().print("{d}/{d}\n", .{ done, job.downloads.len }) catch {};
            job.out.w().flush() catch {};
        }
    }
}

fn fetch(ctx: Ctx, annotations: []const u8, out_dir: []const u8, negatives_wanted: usize) !void {
    const gpa = ctx.gpa;
    const images_dir = try std.fs.path.join(gpa, &.{ out_dir, "images" });
    try Io.Dir.cwd().createDirPath(ctx.io, images_dir);
    var rng: u.PyRandom = .init(2017);
    var manifest: json.Array = .init(gpa);
    var downloads: std.ArrayList(Download) = .empty;
    for ([_][]const u8{ "train2017", "val2017" }) |split| {
        const entry = try std.fmt.allocPrint(gpa, "annotations/person_keypoints_{s}.json", .{split});
        const data = try json.parseFromSliceLeaky(json.Value, gpa, try u.zipEntry(gpa, ctx.io, annotations, entry), .{});
        var usable: std.AutoHashMapUnmanaged(i64, void) = .empty;
        var licences: std.AutoHashMapUnmanaged(i64, json.Value) = .empty;
        for (u.field(data, "licenses").array.items) |l| {
            const id = u.field(l, "id").integer;
            if (std.mem.find(u8, u.str(u.field(l, "name")), "NonCommercial") == null) try usable.put(gpa, id, {});
            try licences.put(gpa, id, u.field(l, "name"));
        }
        var by_image: std.AutoArrayHashMapUnmanaged(i64, std.ArrayList(json.Value)) = .empty;
        for (u.field(data, "annotations").array.items) |ann| {
            const gop = try by_image.getOrPut(gpa, u.field(ann, "image_id").integer);
            if (!gop.found_existing) gop.value_ptr.* = .empty;
            try gop.value_ptr.append(gpa, ann);
        }
        var negatives: std.ArrayList(json.Value) = .empty;
        for (u.field(data, "images").array.items) |image| {
            if (!usable.contains(u.field(image, "license").integer)) continue;
            const anns = if (by_image.get(u.field(image, "id").integer)) |list| list.items else &.{};
            if (anns.len == 0) {
                try negatives.append(gpa, image);
                continue;
            }
            var record_faces: json.Array = .init(gpa);
            var record_people: json.Array = .init(gpa);
            for (anns) |ann| {
                const bbox = u.field(ann, "bbox").array.items;
                const person = try arr(gpa, &.{ bbox[0], bbox[1], add(bbox[0], bbox[2]), add(bbox[1], bbox[3]) });
                const kp = u.field(ann, "keypoints").array.items;
                const v1 = u.num(kp[5]);
                const v2 = u.num(kp[8]);
                if (!u.truthy(u.field(ann, "iscrowd")) and v1 != 0 and v2 != 0) {
                    // COCO's sides are the subject's own; index 2 is their right eye.
                    try record_faces.append(try obj(gpa, &.{ "right", "left", "visible", "iod" }, &.{
                        try arr(gpa, kp[6..8]),
                        try arr(gpa, kp[3..5]),
                        .{ .bool = v1 == 2 and v2 == 2 },
                        .{ .float = dist(.{ u.num(kp[3]), u.num(kp[4]) }, .{ u.num(kp[6]), u.num(kp[7]) }) },
                    }));
                }
                try record_people.append(person);
            }
            if (record_faces.items.len > 0) {
                try manifest.append(try obj(gpa, &.{ "file", "faces", "people", "licence", "flickr" }, &.{
                    u.field(image, "file_name"),
                    .{ .array = record_faces },
                    .{ .array = record_people },
                    licences.get(u.field(image, "license").integer).?,
                    u.optField(image, "flickr_url") orelse .null,
                }));
                try downloads.append(gpa, .{ .file = u.str(u.field(image, "file_name")), .url = u.str(u.field(image, "coco_url")) });
            }
        }
        const sample = try rng.sample(gpa, json.Value, negatives.items, @min(negatives.items.len, negatives_wanted / 2));
        for (sample) |image| {
            try manifest.append(try obj(gpa, &.{ "file", "faces", "people", "licence", "flickr" }, &.{
                u.field(image, "file_name"),
                try arr(gpa, &.{}),
                try arr(gpa, &.{}),
                licences.get(u.field(image, "license").integer).?,
                u.optField(image, "flickr_url") orelse .null,
            }));
            try downloads.append(gpa, .{ .file = u.str(u.field(image, "file_name")), .url = u.str(u.field(image, "coco_url")) });
        }
    }

    var job: Job = .{ .io = ctx.io, .downloads = downloads.items, .dest = images_dir, .out = ctx.out };
    const threads = try gpa.alloc(std.Thread, @min(16, downloads.items.len));
    for (threads) |*t| t.* = try std.Thread.spawn(.{}, worker, .{&job});
    for (threads) |t| t.join();
    if (job.failed.items.len > 0) {
        for (job.failed.items) |f| std.debug.print("{s}\n", .{f});
        fail("{d} downloads failed; manifest not written", .{job.failed.items.len});
    }
    var text: Io.Writer.Allocating = .init(gpa);
    try u.dump(&text.writer, .{ .array = manifest });
    const manifest_path = try std.fs.path.join(gpa, &.{ out_dir, "manifest.json" });
    try u.writeFile(ctx.io, manifest_path, text.written());
    var faces: usize = 0;
    var negatives: usize = 0;
    for (manifest.items) |r| {
        faces += u.field(r, "faces").array.items.len;
        negatives += @intFromBool(u.field(r, "people").array.items.len == 0);
    }
    try ctx.out.w().print("wrote {s}: {d} images, {d} faces, {d} negatives\n", .{ manifest_path, manifest.items.len, faces, negatives });
    try ctx.out.w().flush();
}

pub fn main(init: std.process.Init) !void {
    const gpa = init.arena.allocator();
    const args = try init.minimal.args.toSlice(gpa);
    var positional: std.ArrayList([]const u8) = .empty;
    var runtime = ort.runtimePath(init.environ_map);
    var negatives: usize = 2000;
    var i: usize = 1;
    while (i < args.len) : (i += 1) {
        if (std.mem.eql(u8, args[i], "--ort") and i + 1 < args.len) {
            i += 1;
            runtime = args[i];
        } else if (std.mem.eql(u8, args[i], "--negatives") and i + 1 < args.len) {
            i += 1;
            negatives = try std.fmt.parseInt(usize, args[i], 10);
        } else try positional.append(gpa, args[i]);
    }
    const p = positional.items;
    var out: u.Stdout = undefined;
    out.init(init.io);
    const ctx: Ctx = .{ .gpa = gpa, .io = init.io, .runtime = runtime, .out = &out };
    const command = if (p.len > 0) p[0] else "";
    if (std.mem.eql(u8, command, "fetch") and p.len == 3) return fetch(ctx, p[1], p[2], negatives);
    if (std.mem.eql(u8, command, "oi") and p.len == 5) return scoreOi(ctx, p[1], p[2], p[3], p[4]);
    if (std.mem.eql(u8, command, "coco") and p.len == 4) return scoreCoco(ctx, p[1], p[2], p[3]);
    fail("usage: detector_bench fetch <annotations.zip> <out_dir> [--negatives 2000] | oi <labels> <images> <models> <cache> | coco <data> <models> <cache> [--ort path]", .{});
}

test "budget thresholds sit one float32 step above the first score over budget" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const gpa = arena.allocator();
    // 10 images at 0.2/image allow 2 unmatched detections: the third-highest score is excluded.
    const t = try budgetThreshold(gpa, &.{ 0.9, 0.5, 0.8, 0.3 }, 10, 0.2);
    try std.testing.expectEqual(@as(f64, std.math.nextAfter(f32, 0.5, 1)), t);
    try std.testing.expectEqual(@as(f64, 0.02), try budgetThreshold(gpa, &.{0.9}, 10, 0.2));
    try std.testing.expectEqual(@as(f64, 2.0 / 3.0), recallAt(&.{ 0.9, 0.6, 0.4 }, 3, t));
}

test "angle error wraps and buckets follow eye separation" {
    try std.testing.expectApproxEqAbs(@as(f64, 2), angleError(.{ .{ 0, 0 }, .{ 10, 0 } }, .{ .{ 0, 0 }, .{ 10, std.math.tan(std.math.degreesToRadians(2.0)) * 10 } }), 1e-9);
    try std.testing.expectApproxEqAbs(@as(f64, 0), angleError(.{ .{ 0, 0 }, .{ 10, 0 } }, .{ .{ 0, 0 }, .{ 20, 0 } }), 1e-12);
    try std.testing.expectEqual(@as(usize, 0), bucket(7.99));
    try std.testing.expectEqual(@as(usize, 1), bucket(8));
    try std.testing.expectEqual(@as(usize, 3), bucket(400));
}

test "coordinate sums keep Python's int/float distinction" {
    try std.testing.expectEqual(@as(i64, 5), add(.{ .integer = 2 }, .{ .integer = 3 }).integer);
    try std.testing.expectEqual(@as(f64, 5.5), add(.{ .integer = 2 }, .{ .float = 3.5 }).float);
}
