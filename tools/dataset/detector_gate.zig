//! The detector release gate: one PASS/FAIL verdict for a candidate face detector, its operating
//! thresholds chosen on one half of each held-out set and its results read on the other.
//!
//!   zig run -O ReleaseFast -lc tools/dataset/detector_gate.zig tools/dataset/vendor/stb.c -- \
//!       <labelv2_test_all.txt> <oi_images_dir> <coco_dir> <models_dir> <cache_dir> <candidate.onnx>
//!
//! An optional 7th argument, `$D/zig_runs/diagnostics/labelv2_large_heldout.txt`
//! (`oi_to_labelv2 test --large 0.45`: 1,058 more Open Images val+test photos with a face >= 45%
//! of the width, never trained on), joins the large-face gate: its test half is pooled with the
//! held-out set's (94 + 528 = 622 faces) and the tolerance drops to TOL_LARGE_HELD. Use it.
//!
//! Run from the repo root (it reads samples/). `coco_dir` is detector_bench's `fetch` output,
//! `models_dir` holds the four SHA-checked models detector_bench names. ORT_DYLIB_PATH as there.
//! Detections are cached per model file as `<cache_dir>/<sha256[0..16]>-{oi,coco}.json`, so a
//! retrained file can never be read from a stale cache. Exit status 1 when the verdict is FAIL.
//! On this machine:
//!
//!   D=C:/Users/grego/Downloads/face-data E=//wsl$/Ubuntu/home/grego/work/coco-eval
//!   ... -- $D/openimages/labelv2_test_all.txt $D/openimages/images $E/coco $E/models \
//!          $D/zig_runs/gate_cache <candidate.onnx>
//!
//! **Split.** Every held-out image is in exactly one half, by the SHA-256 of its image ID (the
//! file name without extension: Open Images' 16-hex ImageID, COCO's 12-digit image id): low bit
//! of the first digest byte 0 = tuning half, 1 = final-test half. No list files; reproduce with
//! `printf %s 000132c20b84269b | sha256sum`. Open Images held-out (1,859 photos) splits 941 / 918,
//! COCO's 13,747 (incl. 1,687 no-person photos) 6,817 / 6,930 (no-person 807 / 880). Every
//! threshold is picked on the tuning half; the gates read only the test half. The tuning columns
//! are in-sample and printed for reference.
//!
//! **COCO stays out of all mining and training** -- no negatives, no positives, no augmentation
//! sources, no threshold or recipe chosen by looking at its test half. It is the one corpus none
//! of the compared detectors was trained on; using it for anything but this gate spends that.
//! Open Images held-out is likewise never trained on (only mined data from Open Images *train*).
//!
//! **Gates** (test half; baseline = shipped models/scrfd80k_500m_640.onnx unless named):
//!   1. OI recall at 0.1 unmatched detections/image: >= shipped - TOL_OI.
//!   2. OI recall on faces >= 45% of the image width, same threshold: >= shipped - TOL_LARGE.
//!   3. COCO recall (eyes >= 8 px apart) at 0.01/0.02/0.05/0.1 detections per no-person photo:
//!      strictly above public SCRFD-500M at each rate (user's target, journal:SCRFD-2:0006).
//!      Stretch, not part of the verdict: above RetinaFace R34 at each rate.
//!   4. samples/sample_0*.jpg: every photo's best detection scores >= 0.5.
//!   5. Eye-line median error, eye refiner on the model's own boxes at its OI threshold, on faces
//!      both candidate and shipped find (OI: clicked eyes; COCO: both eyes visible, >= 16 px):
//!      candidate <= shipped + TOL_EYE (OI and COCO separately).
//!   6. CPU ms per photo (letterbox, inference, decode, NMS; 4 ORT threads, median of two passes
//!      over 48 OI test photos after 5 warm-ups): reported only.
//!
//! **Tolerances** are image-sampling noise on the test half: 1.645 x the bootstrap SE (1000
//! resamples, seed 2017; whole images for recall, faces for the eye line) of the *shipped*
//! model's own figure -- how precisely the test half pins the baseline, a one-sided 95%
//! allowance. Measured 2026-11-17 (the run prints them again): OI recall SE 1.21 pt -> TOL_OI
//! 2.0 pt; large faces SE 3.63 pt -> TOL_LARGE 6.0 pt; eye line OI SE 0.064 deg -> 0.10, COCO
//! 0.044 deg -> 0.07. The large-face tolerance is loose because only 94 test faces are that big:
//! it catches attempt 1 (-27.7) but would pass RetinaFace (-5.3) or attempt 2's A3-hi (-2.3 on
//! the full set). Read the printed paired SE(d) beside it: two models on the same images differ
//! by less noise than that (2.8-4.9 pt here), and a bigger large-face test set is the real fix.
//! COCO's "beat" has no tolerance, as ruled; its paired SE is 0.25-0.42 pt.
//! With the large-face file (2026-11-17): shipped SE 1.45 pt on 622 faces -> TOL_LARGE_HELD 2.4 pt;
//! pooled large-face recall test half: shipped 86.5, RetinaFace 80.1 (-6.4, SE 1.14: now FAILS),
//! attempt 1 62.1 (-24.4), public 55.0. About a fifth of OI's >= 45% boxes are not detectable real
//! faces (eye close-ups, unflagged depictions and groups; HARD_NEGATIVES.md "Diagnostics"), which
//! a detector trained on OI labels learns and RetinaFace does not: read the gap to RetinaFace so.
//!
//! **Baselines** (2026-11-17, Windows, Ryzen 9 7950X, ORT 1.24.4, stb decode; this gate's runs in
//! Downloads/face-data/zig_runs/gate_runs). Recall %, tune/test; COCO and eye line test half;
//! eye line is each model's own median (the gate compares paired faces):
//!
//!   model              OI thr  OI@0.1     >=45% width  COCO @.01/.02/.05/.1  eye OI/COCO  ms   samples  verdict
//!   shipped SCRFD-80k  0.5026  73.4/78.2  73.4/85.1    81.9/84.2/86.9/88.9   1.23/2.14   11.7  0.764   FAIL: COCO x4
//!   public SCRFD-500M  0.6382  67.9/70.2  54.8/48.9    87.1/89.3/91.5/93.0   1.13/2.01   13.0  0.084   FAIL: 7 of 9
//!   RetinaFace R34     0.9763  73.4/77.3  74.2/79.8    87.7/92.9/94.5/95.2   1.17/2.14  204.3  1.000   PASS
//!   attempt 1 (neg)    0.4876  73.8/78.3  57.3/57.4    88.6/90.1/91.9/92.5   1.21/2.15   11.1  0.077   FAIL: large, samples, COCO@.1
//!   fine-tune A0       0.5022  73.8/78.8  74.2/86.2    82.0/84.3/86.7/88.4   1.20/2.16   11.2  0.756   FAIL: COCO x4
//!
//! COCO tuning-half recall (in-sample): shipped 83.8/86.2/88.5/90.2, public 88.7/90.1/92.3/93.5,
//! RetinaFace 89.6/94.0/95.3/95.9. Thresholds tuned on 807 no-person photos overshoot on the test
//! half's 880 (shipped 0.025 at the 0.01 rate): at 0.01 the budget is 8 detections, so the low
//! rates are coarse. A0 (3-epoch fine-tune of shipped, no negatives) is the "same model" control:
//! it passes every shipped-relative gate. CPU ms is one run's report and moves ~10% between runs.
const std = @import("std");
const Io = std.Io;
const json = std.json;
const u = @import("util.zig");
const img = @import("image.zig");
const ort = @import("ort.zig");
const bench = @import("detector_bench.zig");
const Det = bench.Det;

const OI_BUDGET: f64 = 0.1;
const LARGE: f64 = 0.45;
const RATES = bench.COCO_BUDGETS;
const SAMPLE_MIN: f64 = 0.5;
/// Refined eyes are cached for every detection from here up; a threshold below it fails loudly.
const REFINE_FLOOR: f64 = 0.2;
const TOL_OI: f64 = 0.02;
const TOL_LARGE: f64 = 0.06;
/// With the large-face held-out file: 1.645 x shipped's 1.45 pt SE on the pooled 622 test faces.
const TOL_LARGE_HELD: f64 = 0.024;
const TOL_EYE = [2]f64{ 0.10, 0.07 }; // OI, COCO
const BOOT = 1000;
const TIMED = 48;

const BASELINES = [_][2][]const u8{
    .{ "shipped SCRFD-80k", "scrfd80k_500m_640.onnx" },
    .{ "public SCRFD-500M", "scrfd_500m.onnx" },
    .{ "RetinaFace R34", "retinaface_r34.onnx" },
};
const SHIPPED = 1;
const PUBLIC = 2;
const RETINA = 3;

/// The half an image belongs to: true for the final-test half.
pub fn isTest(name: []const u8) bool {
    var digest: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(u.stemOf(name), &digest, .{});
    return digest[0] & 1 == 1;
}

const Gate = struct { gpa: std.mem.Allocator, io: Io, runtime: []const u8, out: *u.Stdout, cache: []const u8, refiner: bench.Model };

const Loaded = struct { model: bench.Model, kind: bench.Kind, key: []const u8 };

fn load(g: *Gate, path: []const u8, want: ?[]const u8) !Loaded {
    var digest: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(try u.readFile(g.gpa, g.io, path), &digest, .{});
    const hex = std.fmt.bytesToHex(digest, .lower);
    if (want) |w| if (!std.mem.eql(u8, w, &hex)) bench.fail("{s}: sha256 {s} is not the baseline model", .{ path, &hex });
    var model = try bench.Model.unchecked(g.gpa, g.io, g.runtime, path);
    const kind: bench.Kind = switch (model.outputs.len) {
        9 => .scrfd,
        3 => .retinaface,
        else => bench.fail("{s}: {d} outputs, neither SCRFD (9) nor RetinaFace (3)", .{ path, model.outputs.len }),
    };
    if (kind == .retinaface) model.priors = try bench.retinaPriors(g.gpa);
    return .{ .model = model, .kind = kind, .key = try g.gpa.dupe(u8, hex[0..16]) };
}

/// Every image at the proposal floor, refined eyes from REFINE_FLOOR up; cached by model hash.
fn detections(g: *Gate, m: *Loaded, label: []const u8, corpus: []const u8, names: []const []const u8, images_dir: []const u8) !bench.ByImage {
    const gpa = g.gpa;
    const path = try std.fmt.allocPrint(gpa, "{s}{c}{s}-{s}.json", .{ g.cache, std.fs.path.sep, m.key, corpus });
    if (Io.Dir.cwd().access(g.io, path, .{})) |_| return bench.readDets(gpa, g.io, path) else |_| {}
    var out: bench.ByImage = .empty;
    for (names, 1..) |name, index| {
        var arena: std.heap.ArenaAllocator = .init(std.heap.smp_allocator);
        defer arena.deinit();
        const tmp = arena.allocator();
        const image = img.load(tmp, g.io, try std.fs.path.join(tmp, &.{ images_dir, name }), .apply_exif) catch continue;
        const rows = try bench.detect(tmp, &m.model, m.kind, image);
        const dets = try gpa.alloc(Det, rows.len);
        for (dets, rows) |*d, r| {
            d.* = .{ .box = .{ r[0], r[1], r[2], r[3] }, .score = r[4], .eyes = .{ r[5], r[6], r[7], r[8] } };
            if (r[4] >= REFINE_FLOOR) d.refined = try bench.refine(tmp, &g.refiner, image, r[0..4].*);
        }
        try out.put(gpa, name, dets);
        if (index % 1000 == 0 or index == names.len) {
            try g.out.w().print("  {s} {s}: {d}/{d}\n", .{ label, corpus, index, names.len });
            try g.out.w().flush();
        }
    }
    var text: Io.Writer.Allocating = .init(gpa);
    try bench.writeDets(&text.writer, out);
    try u.writeFile(g.io, path, text.written());
    return out;
}

// ── Corpora, flattened to faces ───────────────────────────────────────────────

const Corpus = struct {
    images: []const []const u8,
    image_test: []const bool,
    /// The image's false detections count toward the budget: OI every image, COCO no-person ones.
    counts: []const bool,
    face_image: []const usize,
    face_test: []const bool,
    /// In the recall pool: OI every face, COCO eyes >= 8 px apart.
    pool: []const bool,
    large: []const bool,
    /// Reference eye line, subject's right eye first, where the eye-line gate uses the face.
    eyes: []const ?[2][2]f64,
};

/// One model on one corpus: the detection each face was matched to, and per image the scores
/// of detections that count as false (OI: unmatched outside ignore regions; COCO: on no-person
/// photos).
const Run = struct { hit: []const ?Det, false_scores: []const []const f64 };

const Faces = struct {
    image: std.ArrayList(usize) = .empty,
    is_test: std.ArrayList(bool) = .empty,
    pool: std.ArrayList(bool) = .empty,
    large: std.ArrayList(bool) = .empty,
    eyes: std.ArrayList(?[2][2]f64) = .empty,

    fn add(f: *Faces, gpa: std.mem.Allocator, image: usize, is_test: bool, pool: bool, large: bool, eyes: ?[2][2]f64) !void {
        try f.image.append(gpa, image);
        try f.is_test.append(gpa, is_test);
        try f.pool.append(gpa, pool);
        try f.large.append(gpa, large);
        try f.eyes.append(gpa, eyes);
    }
};

fn oiCorpus(gpa: std.mem.Allocator, labels: bench.OiLabels) !Corpus {
    const names = labels.images.items;
    const image_test = try gpa.alloc(bool, names.len);
    const counts = try gpa.alloc(bool, names.len);
    @memset(counts, true);
    var f: Faces = .{};
    for (names, 0..) |name, i| {
        image_test[i] = isTest(name);
        const width = labels.widths.get(name).?;
        for (labels.faces.get(name).?.items, labels.eyes.get(name).?.items) |box, clicked|
            try f.add(gpa, i, image_test[i], true, (box[2] - box[0]) / width >= LARGE, clicked);
    }
    return .{ .images = names, .image_test = image_test, .counts = counts, .face_image = f.image.items, .face_test = f.is_test.items, .pool = f.pool.items, .large = f.large.items, .eyes = f.eyes.items };
}

fn oiRun(gpa: std.mem.Allocator, labels: bench.OiLabels, by_image: bench.ByImage) !Run {
    var hit: std.ArrayList(?Det) = .empty;
    const false_scores = try gpa.alloc([]const f64, labels.images.items.len);
    for (labels.images.items, false_scores) |name, *fs| {
        var unmatched: std.ArrayList(f64) = .empty;
        try hit.appendSlice(gpa, try bench.matchOi(gpa, by_image.get(name) orelse &.{}, labels.faces.get(name).?.items, labels.ignore.get(name).?.items, &unmatched));
        fs.* = unmatched.items;
    }
    return .{ .hit = hit.items, .false_scores = false_scores };
}

fn pt(v: json.Value) [2]f64 {
    return .{ u.num(v.array.items[0]), u.num(v.array.items[1]) };
}

fn cocoCorpus(gpa: std.mem.Allocator, manifest: []const json.Value) !Corpus {
    const names = try gpa.alloc([]const u8, manifest.len);
    const image_test = try gpa.alloc(bool, manifest.len);
    const counts = try gpa.alloc(bool, manifest.len);
    var f: Faces = .{};
    for (manifest, 0..) |record, i| {
        names[i] = u.str(u.field(record, "file"));
        image_test[i] = isTest(names[i]);
        counts[i] = u.field(record, "people").array.items.len == 0;
        for (u.field(record, "faces").array.items) |face| {
            const iod = u.num(u.field(face, "iod"));
            const clear = u.field(face, "visible").bool and iod >= 16;
            try f.add(gpa, i, image_test[i], iod >= 8, false, if (clear) .{ pt(u.field(face, "right")), pt(u.field(face, "left")) } else null);
        }
    }
    return .{ .images = names, .image_test = image_test, .counts = counts, .face_image = f.image.items, .face_test = f.is_test.items, .pool = f.pool.items, .large = f.large.items, .eyes = f.eyes.items };
}

fn cocoRun(gpa: std.mem.Allocator, manifest: []const json.Value, by_image: bench.ByImage) !Run {
    var hit: std.ArrayList(?Det) = .empty;
    const false_scores = try gpa.alloc([]const f64, manifest.len);
    var matched: bench.Found = .empty;
    for (manifest, false_scores) |record, *fs| {
        const file = u.str(u.field(record, "file"));
        matched.clearRetainingCapacity();
        var negative: std.ArrayList(f64) = .empty;
        var outside: std.ArrayList(f64) = .empty;
        try bench.matchCoco(gpa, by_image.get(file) orelse &.{}, record, &matched, &negative, &outside);
        for (0..u.field(record, "faces").array.items.len) |k| try hit.append(gpa, matched.get(.{ .file = file, .index = k }));
        fs.* = negative.items;
    }
    return .{ .hit = hit.items, .false_scores = false_scores };
}

// ── Measures ──────────────────────────────────────────────────────────────────

/// The lowest threshold keeping the tuning half within `budget` false detections per counted image.
fn tune(gpa: std.mem.Allocator, c: Corpus, r: Run, budget: f64) !f64 {
    var scores: std.ArrayList(f64) = .empty;
    var images: usize = 0;
    for (r.false_scores, c.counts, c.image_test) |s, counted, t| if (counted and !t) {
        images += 1;
        try scores.appendSlice(gpa, s);
    };
    return bench.budgetThreshold(gpa, scores.items, images, budget);
}

fn falseRate(c: Corpus, r: Run, t: f64, half: bool) f64 {
    var n: usize = 0;
    var images: usize = 0;
    for (r.false_scores, c.counts, c.image_test) |s, counted, is_test| if (counted and is_test == half) {
        images += 1;
        for (s) |x| n += @intFromBool(x >= t);
    };
    return @as(f64, @floatFromInt(n)) / @as(f64, @floatFromInt(@max(images, 1)));
}

fn found(r: Run, face: usize, t: f64) bool {
    return if (r.hit[face]) |d| d.score >= t else false;
}

/// Per image of `half`: [model a's hits, model b's hits, pool size] over faces in `mask`.
fn units(gpa: std.mem.Allocator, c: Corpus, mask: []const bool, half: bool, a: Run, ta: f64, b: Run, tb: f64) ![][3]f64 {
    const per = try gpa.alloc([3]f64, c.images.len);
    @memset(per, .{ 0, 0, 0 });
    for (c.face_image, c.face_test, mask, 0..) |i, is_test, keep, f| if (keep and is_test == half) {
        per[i][0] += @floatFromInt(@intFromBool(found(a, f, ta)));
        per[i][1] += @floatFromInt(@intFromBool(found(b, f, tb)));
        per[i][2] += 1;
    };
    var kept: std.ArrayList([3]f64) = .empty;
    for (per, c.image_test) |p, is_test| if (is_test == half) try kept.append(gpa, p);
    return kept.items;
}

fn recall(gpa: std.mem.Allocator, c: Corpus, mask: []const bool, half: bool, r: Run, t: f64) !f64 {
    var hits: f64 = 0;
    var n: f64 = 0;
    for (try units(gpa, c, mask, half, r, t, r, t)) |x| {
        hits += x[0];
        n += x[2];
    }
    return if (n > 0) hits / n else 0;
}

/// Bootstrap SE of (sum a - sum b) / sum pool, images resampled with replacement.
fn seRecall(xs: []const [3]f64) f64 {
    var prng: std.Random.DefaultPrng = .init(2017);
    const r = prng.random();
    var sum: f64 = 0;
    var sq: f64 = 0;
    for (0..BOOT) |_| {
        var a: f64 = 0;
        var b: f64 = 0;
        var n: f64 = 0;
        for (0..xs.len) |_| {
            const x = xs[r.uintLessThan(usize, xs.len)];
            a += x[0];
            b += x[1];
            n += x[2];
        }
        const d = if (n > 0) (a - b) / n else 0;
        sum += d;
        sq += d * d;
    }
    const mean = sum / BOOT;
    return @sqrt(@max(sq / BOOT - mean * mean, 0));
}

/// Per face: eye-line error of the refined eyes where the face has a reference and is found at `t`.
fn eyeErrors(gpa: std.mem.Allocator, c: Corpus, r: Run, t: f64) ![]?f64 {
    if (t < REFINE_FLOOR) bench.fail("threshold {d} is below REFINE_FLOOR {d}: no refined eyes cached there", .{ t, REFINE_FLOOR });
    const out = try gpa.alloc(?f64, c.eyes.len);
    for (out, c.eyes, 0..) |*o, truth, f| {
        o.* = null;
        const reference = truth orelse continue;
        if (!found(r, f, t)) continue;
        const v = r.hit[f].?.refined orelse return error.MissingRefinedEyes;
        o.* = bench.angleError(reference, .{ .{ v[0], v[1] }, .{ v[2], v[3] } }); // viewer's left = subject's right
    }
    return out;
}

fn medianOf(gpa: std.mem.Allocator, c: Corpus, errs: []const ?f64, half: bool) !f64 {
    var xs: std.ArrayList(f64) = .empty;
    for (errs, c.face_test) |e, is_test| if (is_test == half) if (e) |x| try xs.append(gpa, x);
    return if (xs.items.len > 0) u.median(gpa, xs.items) else std.math.nan(f64);
}

/// Test-half faces both models found with an eye reference: [a's error, b's error].
fn eyePairs(gpa: std.mem.Allocator, c: Corpus, a: []const ?f64, b: []const ?f64) ![][2]f64 {
    var out: std.ArrayList([2]f64) = .empty;
    for (a, b, c.face_test) |x, y, is_test| if (is_test) if (x) |ex| if (y) |ey| try out.append(gpa, .{ ex, ey });
    return out.items;
}

fn medianCol(gpa: std.mem.Allocator, pairs: []const [2]f64, k: usize) !f64 {
    const xs = try gpa.alloc(f64, pairs.len);
    for (xs, pairs) |*x, p| x.* = p[k];
    return u.median(gpa, xs);
}

/// Bootstrap SE of median(a) - median(b) (`b_only`: of median(b) alone), faces resampled.
fn seMedian(gpa: std.mem.Allocator, pairs: []const [2]f64, b_only: bool) !f64 {
    var prng: std.Random.DefaultPrng = .init(2017);
    const r = prng.random();
    const sample = try gpa.alloc([2]f64, pairs.len);
    var sum: f64 = 0;
    var sq: f64 = 0;
    for (0..BOOT) |_| {
        for (sample) |*s| s.* = pairs[r.uintLessThan(usize, pairs.len)];
        const b = try medianCol(gpa, sample, 1);
        const d = if (b_only) b else try medianCol(gpa, sample, 0) - b;
        sum += d;
        sq += d * d;
    }
    const mean = sum / BOOT;
    return @sqrt(@max(sq / BOOT - mean * mean, 0));
}

// ── Per-model figures ─────────────────────────────────────────────────────────

const Model = struct {
    label: []const u8,
    path: []const u8,
    key: []const u8 = "",
    oi: Run = undefined,
    coco: Run = undefined,
    held: Run = undefined,
    oi_t: f64 = 0,
    coco_t: [RATES.len]f64 = undefined,
    oi_eye: []?f64 = &.{},
    coco_eye: []?f64 = &.{},
    ms: f64 = 0,
    sample_min: f64 = 0,
    samples: []f64 = &.{},
};

fn sampleScores(g: *Gate, m: *Loaded, paths: []const []const u8) ![]f64 {
    const out = try g.gpa.alloc(f64, paths.len);
    for (paths, out) |p, *o| {
        var arena: std.heap.ArenaAllocator = .init(std.heap.smp_allocator);
        defer arena.deinit();
        const rows = try bench.detect(arena.allocator(), &m.model, m.kind, try img.load(arena.allocator(), g.io, p, .apply_exif));
        o.* = if (rows.len > 0) rows[0][4] else 0; // best first
    }
    return out;
}

fn cpuMs(g: *Gate, m: *Loaded, images: []const img.Image) !f64 {
    for (images[0..5]) |image| {
        var arena: std.heap.ArenaAllocator = .init(std.heap.smp_allocator);
        defer arena.deinit();
        _ = try bench.detect(arena.allocator(), &m.model, m.kind, image);
    }
    var times: std.ArrayList(f64) = .empty;
    for (0..2) |_| for (images) |image| {
        var arena: std.heap.ArenaAllocator = .init(std.heap.smp_allocator);
        defer arena.deinit();
        const start = Io.Timestamp.now(g.io, .awake);
        _ = try bench.detect(arena.allocator(), &m.model, m.kind, image);
        try times.append(g.gpa, @as(f64, @floatFromInt(start.durationTo(Io.Timestamp.now(g.io, .awake)).nanoseconds)) / std.time.ns_per_ms);
    };
    return u.median(g.gpa, times.items);
}

// ── Report ────────────────────────────────────────────────────────────────────

const Verdict = struct { pass: bool = true, failed: usize = 0, gates: usize = 0 };

fn gateRow(gpa: std.mem.Allocator, w: *Io.Writer, v: *Verdict, name: []const u8, cand: f64, base: f64, se: ?f64, rule: []const u8, ok: bool, stretch: bool, comptime percent: bool) !void {
    try u.padRight(w, name, 40);
    try w.writeAll(" ");
    try bench.col(gpa, w, cand, 9, if (percent) 1 else 2, false);
    try w.writeAll(" ");
    try bench.col(gpa, w, base, 9, if (percent) 1 else 2, false);
    try w.writeAll(" ");
    try u.padLeft(w, try std.fmt.allocPrint(gpa, "{s}{s}", .{ if (cand - base >= 0) "+" else "", try u.fixedAlloc(gpa, cand - base, 2) }), 7);
    try w.writeAll(" ");
    try u.padLeft(w, if (se) |s| try u.fixedAlloc(gpa, s, 2) else "", 6);
    try w.writeAll("  ");
    try u.padRight(w, rule, 22);
    try w.writeAll(if (ok) " PASS" else " FAIL");
    if (stretch) try w.writeAll(" (stretch)");
    try w.writeAll("\n");
    if (!stretch) {
        v.gates += 1;
        if (!ok) {
            v.pass = false;
            v.failed += 1;
        }
    }
}

fn pct(x: f64) f64 {
    return 100 * x;
}

fn tableRow(gpa: std.mem.Allocator, w: *Io.Writer, name: []const u8, models: []const Model, comptime f: anytype, ctx: anytype, digits: u8) !void {
    try u.padRight(w, name, 34);
    for (models) |m| for ([_]bool{ false, true }) |half| {
        try w.writeAll(" ");
        const v: ?f64 = try f(gpa, ctx, m, half);
        if (v) |x| switch (digits) {
            1 => try bench.col(gpa, w, x, 8, 1, false),
            2 => try bench.col(gpa, w, x, 8, 2, false),
            3 => try bench.col(gpa, w, x, 8, 3, false),
            else => try bench.col(gpa, w, x, 8, 4, false),
        } else try u.padLeft(w, "", 8);
    };
    try w.writeAll("\n");
}

const Both = struct { oi: Corpus, coco: Corpus, held: ?Corpus = null, rate: usize = 0 };

/// Large-face units of `half`: the OI held-out faces >= LARGE, plus the large-face set's if given.
fn largeUnits(gpa: std.mem.Allocator, b: Both, half: bool, x: Model, tx: f64, y: Model, ty: f64) ![][3]f64 {
    const own = try units(gpa, b.oi, b.oi.large, half, x.oi, tx, y.oi, ty);
    const held = b.held orelse return own;
    return std.mem.concat(gpa, [3]f64, &.{ own, try units(gpa, held, held.large, half, x.held, tx, y.held, ty) });
}

/// Pooled recall of column k (0 = first model, 1 = second) over per-image units.
fn pooled(xs: []const [3]f64, k: usize) f64 {
    var hits: f64 = 0;
    var n: f64 = 0;
    for (xs) |x| {
        hits += x[k];
        n += x[2];
    }
    return if (n > 0) hits / n else 0;
}

const Rows = struct {
    fn oiT(_: std.mem.Allocator, _: Both, m: Model, half: bool) !?f64 {
        return if (half) null else m.oi_t;
    }
    fn oiRecall(gpa: std.mem.Allocator, b: Both, m: Model, half: bool) !?f64 {
        return pct(try recall(gpa, b.oi, b.oi.pool, half, m.oi, m.oi_t));
    }
    fn oiFalse(_: std.mem.Allocator, b: Both, m: Model, half: bool) !?f64 {
        return falseRate(b.oi, m.oi, m.oi_t, half);
    }
    fn oiLarge(gpa: std.mem.Allocator, b: Both, m: Model, half: bool) !?f64 {
        return pct(try recall(gpa, b.oi, b.oi.large, half, m.oi, m.oi_t));
    }
    fn heldLarge(gpa: std.mem.Allocator, b: Both, m: Model, half: bool) !?f64 {
        const held = b.held orelse return null;
        return pct(try recall(gpa, held, held.large, half, m.held, m.oi_t));
    }
    fn allLarge(gpa: std.mem.Allocator, b: Both, m: Model, half: bool) !?f64 {
        return pct(pooled(try largeUnits(gpa, b, half, m, m.oi_t, m, m.oi_t), 0));
    }
    fn cocoT(_: std.mem.Allocator, b: Both, m: Model, half: bool) !?f64 {
        return if (half) null else m.coco_t[b.rate];
    }
    fn cocoRecall(gpa: std.mem.Allocator, b: Both, m: Model, half: bool) !?f64 {
        return pct(try recall(gpa, b.coco, b.coco.pool, half, m.coco, m.coco_t[b.rate]));
    }
    fn cocoFalse(_: std.mem.Allocator, b: Both, m: Model, half: bool) !?f64 {
        return falseRate(b.coco, m.coco, m.coco_t[b.rate], half);
    }
    fn oiEye(gpa: std.mem.Allocator, b: Both, m: Model, half: bool) !?f64 {
        return try medianOf(gpa, b.oi, m.oi_eye, half);
    }
    fn cocoEye(gpa: std.mem.Allocator, b: Both, m: Model, half: bool) !?f64 {
        return try medianOf(gpa, b.coco, m.coco_eye, half);
    }
    fn ms(_: std.mem.Allocator, _: Both, m: Model, half: bool) !?f64 {
        return if (half) null else m.ms;
    }
    fn samples(_: std.mem.Allocator, _: Both, m: Model, half: bool) !?f64 {
        return if (half) null else m.sample_min;
    }
};

pub fn main(init: std.process.Init) !void {
    const gpa = init.arena.allocator();
    const args = try init.minimal.args.toSlice(gpa);
    var positional: std.ArrayList([]const u8) = .empty;
    var runtime = ort.runtimePath(init.environ_map);
    var i: usize = 1;
    while (i < args.len) : (i += 1) {
        if (std.mem.eql(u8, args[i], "--ort") and i + 1 < args.len) {
            i += 1;
            runtime = args[i];
        } else try positional.append(gpa, args[i]);
    }
    const p = positional.items;
    if (p.len != 6 and p.len != 7) bench.fail("usage: detector_gate <labelv2_test_all.txt> <oi_images> <coco_dir> <models_dir> <cache_dir> <candidate.onnx> [labelv2_large_heldout.txt] [--ort path]", .{});
    const labels_path, const oi_images, const coco_dir, const models_dir, const cache, const candidate = .{ p[0], p[1], p[2], p[3], p[4], p[5] };
    var out: u.Stdout = undefined;
    out.init(init.io);
    const w = out.w();
    try Io.Dir.cwd().createDirPath(init.io, cache);
    var g: Gate = .{ .gpa = gpa, .io = init.io, .runtime = runtime, .out = &out, .cache = cache, .refiner = try bench.Model.open(gpa, init.io, runtime, models_dir, "eye_refiner.onnx") };

    const labels = try bench.readOiLabels(gpa, init.io, labels_path);
    const manifest = (try u.readJson(gpa, init.io, try std.fs.path.join(gpa, &.{ coco_dir, "manifest.json" }))).array.items;
    const held_labels: ?bench.OiLabels = if (p.len == 7) try bench.readOiLabels(gpa, init.io, p[6]) else null;
    const both: Both = .{ .oi = try oiCorpus(gpa, labels), .coco = try cocoCorpus(gpa, manifest), .held = if (held_labels) |l| try oiCorpus(gpa, l) else null };
    const coco_images = try std.fs.path.join(gpa, &.{ coco_dir, "images" });

    // samples/sample_0*.jpg, sorted
    var sample_paths: std.ArrayList([]const u8) = .empty;
    {
        var dir = try Io.Dir.cwd().openDir(init.io, "samples", .{ .iterate = true });
        defer dir.close(init.io);
        var it = dir.iterate();
        while (try it.next(init.io)) |e| if (std.mem.startsWith(u8, e.name, "sample_0") and std.mem.endsWith(u8, e.name, ".jpg"))
            try sample_paths.append(gpa, try std.fs.path.join(gpa, &.{ "samples", e.name }));
        std.mem.sort([]const u8, sample_paths.items, {}, u.lessStr);
        if (sample_paths.items.len == 0) bench.fail("no samples/sample_0*.jpg: run from the repo root", .{});
    }
    // The timed photos: the first TIMED of the OI test half, decoded once.
    var timed: std.ArrayList(img.Image) = .empty;
    for (both.oi.images, both.oi.image_test) |name, is_test| if (is_test and timed.items.len < TIMED)
        try timed.append(gpa, try img.load(gpa, init.io, try std.fs.path.join(gpa, &.{ oi_images, name }), .apply_exif));

    var models: [4]Model = undefined;
    models[0] = .{ .label = "candidate", .path = candidate };
    for (BASELINES, 1..) |b, k| models[k] = .{ .label = b[0], .path = try std.fs.path.join(gpa, &.{ models_dir, b[1] }) };
    for (&models, 0..) |*m, k| {
        const want: ?[]const u8 = if (k == 0) null else for (bench.SHA256) |e| {
            if (std.mem.eql(u8, e[0], BASELINES[k - 1][1])) break e[1];
        } else unreachable;
        var loaded = try load(&g, m.path, want);
        m.key = loaded.key;
        m.oi = try oiRun(gpa, labels, try detections(&g, &loaded, m.label, "oi", both.oi.images, oi_images));
        m.coco = try cocoRun(gpa, manifest, try detections(&g, &loaded, m.label, "coco", both.coco.images, coco_images));
        // shortcut: the cache is keyed by model only, so delete *-oilarge.json if the large-face file changes.
        if (held_labels) |l| m.held = try oiRun(gpa, l, try detections(&g, &loaded, m.label, "oilarge", both.held.?.images, oi_images));
        m.oi_t = try tune(gpa, both.oi, m.oi, OI_BUDGET);
        for (&m.coco_t, RATES) |*t, rate| t.* = try tune(gpa, both.coco, m.coco, rate);
        m.oi_eye = try eyeErrors(gpa, both.oi, m.oi, m.oi_t);
        m.coco_eye = try eyeErrors(gpa, both.coco, m.coco, m.oi_t);
        m.samples = try sampleScores(&g, &loaded, sample_paths.items);
        m.sample_min = std.mem.min(f64, m.samples);
        m.ms = try cpuMs(&g, &loaded, timed.items);
    }

    var n_tune: [2]usize = .{ 0, 0 };
    var n_test: [2]usize = .{ 0, 0 };
    var neg: [2]usize = .{ 0, 0 };
    for ([_]Corpus{ both.oi, both.coco }, 0..) |c, k| for (c.image_test, c.counts) |t, counted| {
        if (t) n_test[k] += 1 else n_tune[k] += 1;
        if (k == 1 and counted) neg[@intFromBool(t)] += 1;
    };
    try w.print("\nDetector gate: {s} (sha256 {s}...)\n", .{ candidate, models[0].key });
    try w.print("Split by SHA-256 of image ID: Open Images {d} tune / {d} test; COCO {d} tune / {d} test (no-person {d} / {d})\n", .{ n_tune[0], n_test[0], n_tune[1], n_test[1], neg[0], neg[1] });
    try w.writeAll("\nThresholds chosen on the tuning half; tune columns are in-sample.\n");
    try u.padRight(w, "", 34);
    for (models) |m| {
        try w.writeAll(" ");
        try u.padLeft(w, if (m.label.len > 17) m.label[0..17] else m.label, 17);
    }
    try w.writeAll("\n");
    try u.padRight(w, "", 34);
    for (models) |_| try w.writeAll("     tune     test");
    try w.writeAll("\n");
    try tableRow(gpa, w, "OI threshold @0.1 unmatched/img", &models, Rows.oiT, both, 4);
    try tableRow(gpa, w, "OI recall %", &models, Rows.oiRecall, both, 1);
    try tableRow(gpa, w, "OI unmatched/image", &models, Rows.oiFalse, both, 3);
    try tableRow(gpa, w, "OI recall %, faces >= 45% width", &models, Rows.oiLarge, both, 1);
    if (both.held != null) {
        try tableRow(gpa, w, "large-face set recall %, >= 45%", &models, Rows.heldLarge, both, 1);
        try tableRow(gpa, w, "both pooled, >= 45% (the gate)", &models, Rows.allLarge, both, 1);
    }
    for (RATES, 0..) |rate, k| {
        var b = both;
        b.rate = k;
        try tableRow(gpa, w, try std.fmt.allocPrint(gpa, "COCO threshold @{d} FP/neg", .{rate}), &models, Rows.cocoT, b, 4);
        try tableRow(gpa, w, try std.fmt.allocPrint(gpa, "COCO recall % (eyes>=8px) @{d}", .{rate}), &models, Rows.cocoRecall, b, 1);
        try tableRow(gpa, w, try std.fmt.allocPrint(gpa, "COCO FP/no-person photo @{d}", .{rate}), &models, Rows.cocoFalse, b, 3);
    }
    try tableRow(gpa, w, "OI eye-line median deg (refiner)", &models, Rows.oiEye, both, 2);
    try tableRow(gpa, w, "COCO eye-line median deg (refiner)", &models, Rows.cocoEye, both, 2);
    try tableRow(gpa, w, "CPU ms/photo (median)", &models, Rows.ms, both, 1);
    try tableRow(gpa, w, "samples: lowest best score", &models, Rows.samples, both, 3);

    // Gates, on the final-test half only.
    const c = models[0];
    const s = models[SHIPPED];
    var v: Verdict = .{};
    try w.writeAll("\nGates, final-test half:\n");
    try u.padRight(w, "gate", 40);
    try w.writeAll("       cand  baseline   delta  SE(d)  rule\n");
    {
        const xs = try units(gpa, both.oi, both.oi.pool, true, c.oi, c.oi_t, s.oi, s.oi_t);
        const a, const b = .{ pct(try recall(gpa, both.oi, both.oi.pool, true, c.oi, c.oi_t)), pct(try recall(gpa, both.oi, both.oi.pool, true, s.oi, s.oi_t)) };
        try gateRow(gpa, w, &v, "OI recall % @0.1 unmatched/img", a, b, pct(seRecall(xs)), try std.fmt.allocPrint(gpa, "vs shipped, >= -{s}", .{try u.fixedAlloc(gpa, pct(TOL_OI), 1)}), a - b >= -pct(TOL_OI) - 1e-9, false, true);
        const ys = try largeUnits(gpa, both, true, c, c.oi_t, s, s.oi_t);
        const la, const lb = .{ pct(pooled(ys, 0)), pct(pooled(ys, 1)) };
        const tol = if (both.held != null) TOL_LARGE_HELD else TOL_LARGE;
        try gateRow(gpa, w, &v, if (both.held != null) "OI + large set recall %, >= 45% width" else "OI recall %, faces >= 45% width", la, lb, pct(seRecall(ys)), try std.fmt.allocPrint(gpa, "vs shipped, >= -{s}", .{try u.fixedAlloc(gpa, pct(tol), 1)}), la - lb >= -pct(tol) - 1e-9, false, true);
    }
    for ([_]usize{ PUBLIC, RETINA }) |k| for (RATES, 0..) |rate, j| {
        const o = models[k];
        const xs = try units(gpa, both.coco, both.coco.pool, true, c.coco, c.coco_t[j], o.coco, o.coco_t[j]);
        const a, const b = .{ pct(try recall(gpa, both.coco, both.coco.pool, true, c.coco, c.coco_t[j])), pct(try recall(gpa, both.coco, both.coco.pool, true, o.coco, o.coco_t[j])) };
        const name = try std.fmt.allocPrint(gpa, "COCO recall % @{d} FP/neg", .{rate});
        try gateRow(gpa, w, &v, name, a, b, pct(seRecall(xs)), if (k == PUBLIC) "> public SCRFD-500M" else "> RetinaFace R34", a > b, k == RETINA, true);
    };
    {
        try u.padRight(w, "samples/sample_0*.jpg best scores", 40);
        for (c.samples) |x| try w.print(" {s}", .{try u.fixedAlloc(gpa, x, 3)});
        try w.writeAll("\n");
        try gateRow(gpa, w, &v, "samples: lowest best score", c.sample_min, s.sample_min, null, "every one >= 0.5", c.sample_min >= SAMPLE_MIN, false, false);
    }
    for ([_]Corpus{ both.oi, both.coco }, [_][]const u8{ "OI", "COCO" }, 0..) |corpus, name, k| {
        const pairs = try eyePairs(gpa, corpus, if (k == 0) c.oi_eye else c.coco_eye, if (k == 0) s.oi_eye else s.coco_eye);
        const a, const b = .{ try medianCol(gpa, pairs, 0), try medianCol(gpa, pairs, 1) };
        try gateRow(gpa, w, &v, try std.fmt.allocPrint(gpa, "{s} eye-line median deg ({d} paired)", .{ name, pairs.len }), a, b, try seMedian(gpa, pairs, false), try std.fmt.allocPrint(gpa, "vs shipped, <= +{s}", .{try u.fixedAlloc(gpa, TOL_EYE[k], 2)}), a - b <= TOL_EYE[k] + 1e-9, false, false);
    }
    try u.padRight(w, "CPU ms/photo (report only)", 40);
    try w.writeAll(" ");
    try bench.col(gpa, w, c.ms, 9, 1, false);
    try w.writeAll(" ");
    try bench.col(gpa, w, s.ms, 9, 1, false);
    try w.writeAll("\n");

    // The noise the tolerances were set from: the shipped model's own bootstrap SE on the test half.
    {
        const xs = try units(gpa, both.oi, both.oi.pool, true, s.oi, s.oi_t, s.oi, std.math.inf(f64));
        const ys = try largeUnits(gpa, both, true, s, s.oi_t, s, std.math.inf(f64));
        var large_faces: usize = 0;
        for (both.oi.large, both.oi.face_test) |l, t| large_faces += @intFromBool(l and t);
        if (both.held) |h| for (h.large, h.face_test) |l, t| {
            large_faces += @intFromBool(l and t);
        };
        const oi_pairs = try eyePairs(gpa, both.oi, s.oi_eye, s.oi_eye);
        const coco_pairs = try eyePairs(gpa, both.coco, s.coco_eye, s.coco_eye);
        try w.print("\nShipped's own bootstrap SE on the test half (sets the tolerances): OI recall {s} pt, large faces {s} pt ({d} faces), eye line OI {s} deg, COCO {s} deg\n", .{
            try u.fixedAlloc(gpa, pct(seRecall(xs)), 2),
            try u.fixedAlloc(gpa, pct(seRecall(ys)), 2),
            large_faces,
            try u.fixedAlloc(gpa, try seMedian(gpa, oi_pairs, true), 3),
            try u.fixedAlloc(gpa, try seMedian(gpa, coco_pairs, true), 3),
        });
    }
    try w.print("\nVERDICT: {s} ({d} of {d} gates failed)\n", .{ if (v.pass) "PASS" else "FAIL", v.failed, v.gates });
    try w.flush();
    if (!v.pass) std.process.exit(1);
}

test "the split is a stable hash of the image ID" {
    // `printf %s ID | sha256sum`: 07f17f91... (odd: test) and 6afc83aa... (even: tuning).
    try std.testing.expect(isTest("000132c20b84269b.jpg") and isTest("000132c20b84269b"));
    try std.testing.expect(!isTest("dir/000000000139.jpg") and !isTest("000000000139"));
    var n: usize = 0;
    var buf: [16]u8 = undefined;
    for (0..1000) |k| n += @intFromBool(isTest(try std.fmt.bufPrint(&buf, "{d:0>12}", .{k})));
    try std.testing.expect(n > 450 and n < 550);
}

test "bootstrap SE of a paired recall difference" {
    try std.testing.expectEqual(@as(f64, 0), seRecall(&.{ .{ 1, 1, 2 }, .{ 2, 2, 3 } }));
    // 400 images, one face each, a finds 80%, b none: SE ~ sqrt(.8 * .2 / 400) = 0.02.
    var xs: [400][3]f64 = undefined;
    for (&xs, 0..) |*x, k| x.* = .{ if (k % 5 == 0) 0 else 1, 0, 1 };
    try std.testing.expectApproxEqAbs(@as(f64, 0.02), seRecall(&xs), 0.003);
}
