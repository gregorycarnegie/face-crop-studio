//! Train SCRFD-500M natively on an NVIDIA GPU, reproducing the MMDetection recipes.
//!
//!   zig run -O ReleaseFast -lc tools/dataset/train_scrfd.zig tools/dataset/vendor/stb.c -- //!       <openimages_dir> <labelv2.txt> [--recipe curve|80k] [--out dir] [--resume checkpoint.pth]
//!
//! `curve` is `scrfd_fcs_500m.py` (the label-count curve: batch 8, 640 epochs, LR steps at
//! 440/544); `80k` is `scrfd_fcs80k_500m.py` (batch 64, 100 epochs, steps at 69/85, ignore
//! regions at IoF 0.5). Both: SGD 0.01, momentum 0.9, weight decay 5e-4, 1,500 iterations of
//! linear warm-up from 0.1%, MMDetection's initialisation. Data goes through `scrfd_augment.zig`
//! and `scrfd_targets.zig` on CPU threads, prefetched while the GPU trains (`scrfd_net.zig`).
//! Checkpoints are MMDetection-shaped (`state_dict`, `meta`) plus the SGD momentum, so
//! `export_scrfd.zig`, `eval_eye_error.zig` and `--resume` all read them.
//!
//! Checks, each against an independent implementation:
//!   --check <checkpoint.pth>                  eval forward vs scrfd.zig's CPU reference
//!   --dump-grads <checkpoint.pth> <dir>       a backward pass for torch autograd to compare
//!   --dump-loss <checkpoint.pth> <data> <dir> a real batch's losses and gradients for mmdet
//! Finite differences cannot check a network this deep: its small-batch BatchNorms make them
//! non-linear at any usable step.
const std = @import("std");
const u = @import("util.zig");
const ck = @import("checkpoint.zig");
const cuda = @import("cuda.zig");
const net = @import("scrfd.zig");
const gpu = @import("scrfd_net.zig");
const img = @import("image.zig");
const aug = @import("scrfd_augment.zig");
const targets = @import("scrfd_targets.zig");

fn selfCheck(gpa: std.mem.Allocator, io: std.Io, dev: *const cuda.Device, checkpoint: []const u8) !void {
    const model = try net.Scrfd.fromCheckpoint(gpa, try ck.load(gpa, io, checkpoint));
    var prng: std.Random.DefaultPrng = .init(5);
    const r = prng.random();
    const size = 640;
    var m = try gpu.Net.init(gpa, dev, model, 2, size);
    const input = try gpa.alloc(f32, 2 * 3 * size * size);
    for (input) |*x| x.* = r.float(f32) * 2 - 1;
    try dev.upload(f32, m.out[0], input);
    try m.forward(2, false);
    var worst: f64 = 0;
    for (0..2) |b| {
        const expected = try model.forward(gpa, input[b * 3 * size * size ..][0 .. 3 * size * size], size);
        for (0..3) |kind| for (0..3) |level| {
            const v = model.heads[level][kind];
            const side = size / model.scale[v];
            const per_image = model.channels[v] * side * side;
            const raw = try gpa.alloc(f32, 2 * per_image);
            try dev.download(f32, raw, m.out[v]);
            const got = try net.deploy(gpa, raw[b * per_image ..][0..per_image], side, net.per_anchor[kind], kind == 0);
            for (expected[kind * 3 + level], got) |a, g| worst = @max(worst, @abs(@as(f64, a) - g));
        };
    }
    std.debug.print("eval forward at 640 vs scrfd.zig: worst difference {e}\n", .{worst});
    if (worst > 1e-3) return error.ForwardMismatch;
    std.debug.print("FORWARD CHECK PASSED on {s}\n", .{dev.deviceName()});
}

/// loss = sum over the nine raw head maps of map * w, at 128x128 with batch 2: d(loss)/d(map) = w.
fn dumpGrads(gpa: std.mem.Allocator, io: std.Io, dev: *const cuda.Device, checkpoint: []const u8, dir: []const u8) !void {
    const model = try net.Scrfd.fromCheckpoint(gpa, try ck.load(gpa, io, checkpoint));
    const size = 128;
    const n = 2;
    var prng: std.Random.DefaultPrng = .init(6);
    const r = prng.random();
    var m = try gpu.Net.init(gpa, dev, model, n, size);
    const input = try gpa.alloc(f32, n * 3 * size * size);
    for (input) |*x| x.* = r.float(f32) * 2 - 1;
    try dev.upload(f32, m.out[0], input);
    try u.writeFile(io, try std.fmt.allocPrint(gpa, "{s}/input.bin", .{dir}), std.mem.sliceAsBytes(input));
    try m.forward(n, true);
    try m.zeroGrads();
    for (0..3) |kind| for (0..3) |level| {
        const v = model.heads[level][kind];
        const w = try gpa.alloc(f32, m.out[v].len);
        for (w) |*x| x.* = r.float(f32) * 2 - 1;
        try dev.upload(f32, m.grad[v], w);
        try u.writeFile(io, try std.fmt.allocPrint(gpa, "{s}/w{d}.bin", .{ dir, kind * 3 + level }), std.mem.sliceAsBytes(w));
    };
    try m.backward(n);
    var index: std.Io.Writer.Allocating = .init(gpa);
    var blob: std.ArrayList(u8) = .empty;
    try index.writer.writeAll("{");
    for (m.params.items, 0..) |p, i| {
        const host = try gpa.alloc(f32, p.grad.len);
        try dev.download(f32, host, p.grad);
        if (i > 0) try index.writer.writeAll(",");
        try index.writer.print("\"{s}\":[{d},{d}]", .{ p.name, blob.items.len / 4, host.len });
        try blob.appendSlice(gpa, std.mem.sliceAsBytes(host));
    }
    try index.writer.writeAll("}");
    try u.writeFile(io, try std.fmt.allocPrint(gpa, "{s}/grads.json", .{dir}), index.written());
    try u.writeFile(io, try std.fmt.allocPrint(gpa, "{s}/grads.bin", .{dir}), blob.items);
    std.debug.print("wrote {d} parameter gradients to {s}\n", .{ m.params.items.len, dir });
}

/// Convert one image's targets into the GPU's positives.
pub fn gpuPositives(gpa: std.mem.Allocator, out: *std.ArrayList(gpu.Positive), image: u32, positives: []const targets.Positive) !void {
    for (positives) |p| {
        var level: u32 = 0;
        var offset: u32 = 0;
        while (level < 2 and p.anchor >= offset + targets.level_counts[level]) : (level += 1) offset += @intCast(targets.level_counts[level]);
        const local = p.anchor - offset;
        try out.append(gpa, .{ .image = image, .level = level, .cell = local / 2, .anchor = local % 2, .box = p.box, .kps = p.kps, .kps_weight = p.kps_weight });
    }
}

fn withEyes(entries: []const aug.Entry, skip: usize) aug.Entry {
    var seen: usize = 0;
    for (entries) |e| for (e.faces) |f| if (f.kps[0][2] > 0) {
        if (seen == skip) return e;
        seen += 1;
        break;
    };
    return entries[0];
}

/// A real batch (augmented images, ATSS targets), its losses and every gradient, for comparison
/// with mmdet's `forward_train` on the same tensor and annotations.
fn dumpLoss(gpa: std.mem.Allocator, io: std.Io, dev: *const cuda.Device, checkpoint: []const u8, data: []const u8, dir: []const u8) !void {
    const model = try net.Scrfd.fromCheckpoint(gpa, try ck.load(gpa, io, checkpoint));
    const n = 2;
    var m = try gpu.Net.init(gpa, dev, model, n, targets.size);
    var lb = try gpu.LossBuffers.init(dev, n);
    const entries = try aug.parseLabels(gpa, try u.readFile(gpa, io, try std.fs.path.join(gpa, &.{ data, "labelv2_train_all.txt" })));
    const all = comptime targets.anchors();
    var prng: std.Random.DefaultPrng = .init(8);
    const plane = targets.size * targets.size;
    const input = try gpa.alloc(f32, n * 3 * plane);
    const label_weight = try gpa.alloc(u8, n * targets.anchor_count);
    var positives: std.ArrayList(gpu.Positive) = .empty;
    var samples: f32 = 0;
    var ann_json: std.Io.Writer.Allocating = .init(gpa);
    try ann_json.writer.writeAll("[");
    for (0..n) |b| {
        const e = withEyes(entries, b * 7); // a spread of images that carry eye labels
        const src = try aug.fromRgb8(gpa, try img.load(gpa, io, try std.fs.path.join(gpa, &.{ data, "images", e.name }), .apply_exif));
        const faces = try gpa.dupe(targets.Face, e.faces);
        const im, const ann = try aug.apply(gpa, src, .{ .faces = faces, .ignores = e.ignores }, aug.draw(prng.random(), e));
        aug.normalise(im, input[b * 3 * plane ..][0 .. 3 * plane]);
        const tg = try targets.targets(gpa, &all, ann.faces, ann.ignores, 0.5);
        @memcpy(label_weight[b * targets.anchor_count ..][0..targets.anchor_count], tg.label_weight);
        try gpuPositives(gpa, &positives, @intCast(b), tg.positives);
        samples += @floatFromInt(@max(tg.positives.len, 1));
        if (b > 0) try ann_json.writer.writeAll(",");
        try ann_json.writer.writeAll("{\"faces\":[");
        for (ann.faces, 0..) |f, k| {
            if (k > 0) try ann_json.writer.writeAll(",");
            try ann_json.writer.print("[{d},{d},{d},{d}", .{ f.box[0], f.box[1], f.box[2], f.box[3] });
            for (f.kps) |kp| try ann_json.writer.print(",{d},{d},{d}", .{ kp[0], kp[1], kp[2] });
            try ann_json.writer.writeAll("]");
        }
        try ann_json.writer.writeAll("],\"ignores\":[");
        for (ann.ignores, 0..) |g, k| {
            if (k > 0) try ann_json.writer.writeAll(",");
            try ann_json.writer.print("[{d},{d},{d},{d}]", .{ g[0], g[1], g[2], g[3] });
        }
        try ann_json.writer.print("],\"positives\":{d}}}", .{tg.positives.len});
    }
    try ann_json.writer.writeAll("]");
    try u.writeFile(io, try std.fmt.allocPrint(gpa, "{s}/annotations.json", .{dir}), ann_json.written());
    try u.writeFile(io, try std.fmt.allocPrint(gpa, "{s}/input.bin", .{dir}), std.mem.sliceAsBytes(input));
    try dev.upload(f32, m.out[0], input);
    try m.forward(n, true);
    try m.zeroGrads();
    const l = try gpu.loss(&m, &lb, n, label_weight, positives.items, samples);
    try m.backward(n);
    try u.writeFile(io, try std.fmt.allocPrint(gpa, "{s}/losses.json", .{dir}), try std.fmt.allocPrint(gpa, "{{\"cls\":{d},\"box\":{d},\"kps\":{d},\"positives\":{d}}}", .{ l.cls, l.box, l.kps, positives.items.len }));
    var index: std.Io.Writer.Allocating = .init(gpa);
    var blob: std.ArrayList(u8) = .empty;
    try index.writer.writeAll("{");
    for (m.params.items, 0..) |p, i| {
        const host = try gpa.alloc(f32, p.grad.len);
        try dev.download(f32, host, p.grad);
        if (i > 0) try index.writer.writeAll(",");
        try index.writer.print("\"{s}\":[{d},{d}]", .{ p.name, blob.items.len / 4, host.len });
        try blob.appendSlice(gpa, std.mem.sliceAsBytes(host));
    }
    try index.writer.writeAll("}");
    try u.writeFile(io, try std.fmt.allocPrint(gpa, "{s}/grads.json", .{dir}), index.written());
    try u.writeFile(io, try std.fmt.allocPrint(gpa, "{s}/grads.bin", .{dir}), blob.items);
    std.debug.print("losses: cls {d:.6} box {d:.6} kps {d:.6} over {d} positives\n", .{ l.cls, l.box, l.kps, positives.items.len });
}

// ── Initial weights ───────────────────────────────────────────────────────────

const stage_blocks = [4]usize{ 2, 3, 2, 6 };
const stage_planes = [6]usize{ 16, 16, 40, 72, 152, 288 };
const neck_channels = 16;
const head_channels = 64;

const Init = enum { kaiming, xavier, head };

/// An SCRFD-500M (bnkps) state_dict initialised as MMDetection initialises it: backbone
/// convolutions kaiming-normal (fan_out, ReLU), neck convolutions xavier-uniform, head
/// convolutions normal(0, 0.01) with the class bias at -4.595, BatchNorm at 1 and 0.
fn initialState(gpa: std.mem.Allocator, seed: u64) ![]const ck.OutEntry {
    var prng: std.Random.DefaultPrng = .init(seed);
    const r = prng.random();
    var entries: std.ArrayList(ck.OutEntry) = .empty;
    const S = struct {
        fn tensor(g: std.mem.Allocator, shape: []const usize, values: []const f32) !ck.Out {
            return .{ .tensor = .{ .dtype = .float32, .shape = try g.dupe(usize, shape), .data = std.mem.sliceAsBytes(values) } };
        }
        fn filled(g: std.mem.Allocator, n: usize, v: f32) ![]f32 {
            const out = try g.alloc(f32, n);
            @memset(out, v);
            return out;
        }
    };
    const zero = try gpa.alloc(i64, 1);
    zero[0] = 0;
    const Builder = struct {
        g: std.mem.Allocator,
        r: std.Random,
        list: *std.ArrayList(ck.OutEntry),
        counter: []i64,

        fn conv(b: @This(), name: []const u8, shape: [4]usize, how: Init, bias: ?f32) !void {
            const count = shape[0] * shape[1] * shape[2] * shape[3];
            const w = try b.g.alloc(f32, count);
            const receptive: f32 = @floatFromInt(shape[2] * shape[3]);
            const fan_in = @as(f32, @floatFromInt(shape[1])) * receptive;
            const fan_out = @as(f32, @floatFromInt(shape[0])) * receptive;
            for (w) |*v| v.* = switch (how) {
                .kaiming => b.r.floatNorm(f32) * @sqrt(2 / fan_out),
                .xavier => (b.r.float(f32) * 2 - 1) * @sqrt(6 / (fan_in + fan_out)),
                .head => b.r.floatNorm(f32) * 0.01,
            };
            try b.list.append(b.g, .{ .key = try std.fmt.allocPrint(b.g, "{s}.weight", .{name}), .value = try S.tensor(b.g, &shape, w) });
            if (bias) |v| try b.list.append(b.g, .{ .key = try std.fmt.allocPrint(b.g, "{s}.bias", .{name}), .value = try S.tensor(b.g, &.{shape[0]}, try S.filled(b.g, shape[0], v)) });
        }

        fn bn(b: @This(), name: []const u8, c: usize) !void {
            for ([_][]const u8{ "weight", "bias", "running_mean", "running_var" }, [_]f32{ 1, 0, 0, 1 }) |field, v| {
                try b.list.append(b.g, .{ .key = try std.fmt.allocPrint(b.g, "{s}.{s}", .{ name, field }), .value = try S.tensor(b.g, &.{c}, try S.filled(b.g, c, v)) });
            }
            try b.list.append(b.g, .{ .key = try std.fmt.allocPrint(b.g, "{s}.num_batches_tracked", .{name}), .value = .{ .tensor = .{ .dtype = .int64, .shape = &.{}, .data = std.mem.sliceAsBytes(b.counter) } } });
        }

        /// MobileNetV1's depthwise-separable block.
        fn block(b: @This(), prefix: []const u8, cin: usize, cout: usize) !void {
            try b.conv(try std.fmt.allocPrint(b.g, "{s}.0", .{prefix}), .{ cin, 1, 3, 3 }, .kaiming, null);
            try b.bn(try std.fmt.allocPrint(b.g, "{s}.1", .{prefix}), cin);
            try b.conv(try std.fmt.allocPrint(b.g, "{s}.3", .{prefix}), .{ cout, cin, 1, 1 }, .kaiming, null);
            try b.bn(try std.fmt.allocPrint(b.g, "{s}.4", .{prefix}), cout);
        }
    };
    const b: Builder = .{ .g = gpa, .r = r, .list = &entries, .counter = zero };
    try b.conv("backbone.stem.0.0", .{ stage_planes[0], 3, 3, 3 }, .kaiming, null);
    try b.bn("backbone.stem.0.1", stage_planes[0]);
    try b.block("backbone.stem.1", stage_planes[0], stage_planes[1]);
    for (stage_blocks, 0..) |blocks, s| for (0..blocks) |k| {
        const cin = if (k == 0) stage_planes[s + 1] else stage_planes[s + 2];
        try b.block(try std.fmt.allocPrint(gpa, "backbone.layer{d}.{d}", .{ s + 1, k }), cin, stage_planes[s + 2]);
    };
    for (stage_planes[3..], 0..) |cin, i| try b.conv(try std.fmt.allocPrint(gpa, "neck.lateral_convs.{d}.conv", .{i}), .{ neck_channels, cin, 1, 1 }, .xavier, 0);
    for (0..3) |i| try b.conv(try std.fmt.allocPrint(gpa, "neck.fpn_convs.{d}.conv", .{i}), .{ neck_channels, neck_channels, 3, 3 }, .xavier, 0);
    for ([_][]const u8{ "downsample_convs", "pafpn_convs" }) |kind| for (0..2) |i| {
        try b.conv(try std.fmt.allocPrint(gpa, "neck.{s}.{d}.conv", .{ kind, i }), .{ neck_channels, neck_channels, 3, 3 }, .xavier, 0);
    };
    for (net.strides) |s| {
        for (0..2) |k| {
            const cin: usize = if (k == 0) neck_channels else head_channels;
            const p = try std.fmt.allocPrint(gpa, "bbox_head.cls_stride_convs.({d}, {d}).{d}", .{ s, s, k });
            try b.conv(try std.fmt.allocPrint(gpa, "{s}.depthwise_conv.conv", .{p}), .{ cin, 1, 3, 3 }, .head, null);
            try b.bn(try std.fmt.allocPrint(gpa, "{s}.depthwise_conv.bn", .{p}), cin);
            try b.conv(try std.fmt.allocPrint(gpa, "{s}.pointwise_conv.conv", .{p}), .{ head_channels, cin, 1, 1 }, .head, null);
            try b.bn(try std.fmt.allocPrint(gpa, "{s}.pointwise_conv.bn", .{p}), head_channels);
        }
        for ([_][]const u8{ "cls", "reg", "kps" }, net.per_anchor, [_]f32{ -4.595, 0, 0 }) |kind, per, bias| {
            try b.conv(try std.fmt.allocPrint(gpa, "bbox_head.stride_{s}.({d}, {d})", .{ kind, s, s }), .{ net.anchors * per, head_channels, 3, 3 }, .head, bias);
        }
    }
    const project = try gpa.alloc(f32, 9);
    for (project, 0..) |*v, i| v.* = @floatFromInt(i);
    try entries.append(gpa, .{ .key = "bbox_head.integral.project", .value = try S.tensor(gpa, &.{9}, project) });
    return entries.toOwnedSlice(gpa);
}

// ── Training ──────────────────────────────────────────────────────────────────

const Recipe = struct { batch: usize, epochs: usize, steps: [2]usize, ignore_iof: f32, interval: usize };
/// `scrfd_fcs_500m.py` (the label-count curve) and `scrfd_fcs80k_500m.py`.
const recipes = .{
    .curve = Recipe{ .batch = 8, .epochs = 640, .steps = .{ 440, 544 }, .ignore_iof = -1, .interval = 80 },
    .@"80k" = Recipe{ .batch = 64, .epochs = 100, .steps = .{ 69, 85 }, .ignore_iof = 0.5, .interval = 10 },
};
const base_lr = 0.01;
const warmup_iters = 1500;
const warmup_ratio = 0.001;

/// mmcv's step policy with linear warm-up: x0.1 at each step epoch, ramped over the first
/// 1,500 iterations from 0.1% of the rate.
fn learningRate(epoch: usize, iter: usize, steps: [2]usize) f64 {
    var lr: f64 = base_lr;
    for (steps) |s| if (epoch >= s) {
        lr *= 0.1;
    };
    if (iter < warmup_iters) lr *= 1 - (1 - @as(f64, @floatFromInt(iter)) / warmup_iters) * (1 - warmup_ratio);
    return lr;
}

const Batch = struct {
    input: []f32,
    label_weight: []u8,
    positives: [][]gpu.Positive,
    counts: []usize,
};

const Prep = struct {
    io: std.Io,
    images: []const u8,
    entries: []const aug.Entry,
    order: []const usize,
    seed: u64,
    ignore_iof: f32,
    batch: *Batch,
    next: std.atomic.Value(usize) = .init(0),
    failed: std.atomic.Value(bool) = .init(false),
};

fn prepWorker(job: *Prep) void {
    const all = comptime targets.anchors();
    const plane = targets.size * targets.size;
    while (true) {
        const slot = job.next.fetchAdd(1, .monotonic);
        if (slot >= job.order.len) return;
        var arena: std.heap.ArenaAllocator = .init(std.heap.smp_allocator);
        defer arena.deinit();
        const g = arena.allocator();
        const e = job.entries[job.order[slot]];
        var prng: std.Random.DefaultPrng = .init(job.seed ^ (@as(u64, slot) *% 0x9E3779B97F4A7C15));
        const done = blk: {
            const decoded = img.load(g, job.io, std.fs.path.join(g, &.{ job.images, e.name }) catch break :blk false, .apply_exif) catch |err| {
                std.debug.print("cannot read {s}: {s}\n", .{ e.name, @errorName(err) });
                break :blk false;
            };
            const src = aug.fromRgb8(g, decoded) catch break :blk false;
            const faces = g.dupe(targets.Face, e.faces) catch break :blk false;
            const im, const ann = aug.apply(g, src, .{ .faces = faces, .ignores = e.ignores }, aug.draw(prng.random(), e)) catch break :blk false;
            aug.normalise(im, job.batch.input[slot * 3 * plane ..][0 .. 3 * plane]);
            const tg = targets.targets(g, &all, ann.faces, ann.ignores, job.ignore_iof) catch break :blk false;
            @memcpy(job.batch.label_weight[slot * targets.anchor_count ..][0..targets.anchor_count], tg.label_weight);
            var list: std.ArrayList(gpu.Positive) = .empty;
            gpuPositives(std.heap.smp_allocator, &list, @intCast(slot), tg.positives) catch break :blk false;
            std.heap.smp_allocator.free(job.batch.positives[slot]);
            job.batch.positives[slot] = list.toOwnedSlice(std.heap.smp_allocator) catch break :blk false;
            job.batch.counts[slot] = tg.positives.len;
            break :blk true;
        };
        if (!done) {
            job.failed.store(true, .monotonic);
            return;
        }
    }
}

fn prepare(job: *Prep, workers: usize) !void {
    var threads: [64]std.Thread = undefined;
    const count = @min(threads.len, workers);
    for (threads[0..count]) |*t| t.* = try std.Thread.spawn(.{}, prepWorker, .{job});
    for (threads[0..count]) |t| t.join();
    if (job.failed.load(.monotonic)) return error.BatchFailed;
}

fn allocBatch(gpa: std.mem.Allocator, n: usize) !Batch {
    const b: Batch = .{ .input = try gpa.alloc(f32, n * 3 * targets.size * targets.size), .label_weight = try gpa.alloc(u8, n * targets.anchor_count), .positives = try gpa.alloc([]gpu.Positive, n), .counts = try gpa.alloc(usize, n) };
    for (b.positives) |*p| p.* = &.{};
    return b;
}

const Options = struct {
    data: []const u8,
    labels: []const u8,
    out: []const u8,
    recipe: Recipe,
    resume_from: ?[]const u8 = null,
    seed: u64 = 0,
    workers: usize = 16,
    max_iters: ?usize = null,
};

fn saveCheckpoint(gpa: std.mem.Allocator, io: std.Io, path: []const u8, m: *gpu.Net, epoch: usize, iter: usize) !void {
    var momentum: std.ArrayList(ck.OutEntry) = .empty;
    for (m.params.items) |p| {
        const host = try gpa.alloc(f32, p.momentum.len);
        try m.dev.download(f32, host, p.momentum);
        try momentum.append(gpa, .{ .key = p.name, .value = .{ .tensor = .{ .dtype = .float32, .shape = p.shape, .data = std.mem.sliceAsBytes(host) } } });
    }
    try ck.save(gpa, io, path, &.{
        .{ .key = "meta", .value = .{ .dict = &.{
            .{ .key = "epoch", .value = .{ .int = @intCast(epoch) } },
            .{ .key = "iter", .value = .{ .int = @intCast(iter) } },
            .{ .key = "trained_by", .value = .{ .string = "tools/dataset/train_scrfd.zig" } },
        } } },
        .{ .key = "state_dict", .value = .{ .dict = try m.stateDict(gpa) } },
        // SGD momentum per parameter, so a run can resume where it stopped (not read by mmdet).
        .{ .key = "fcs_momentum", .value = .{ .dict = try momentum.toOwnedSlice(gpa) } },
    });
}

fn train(gpa: std.mem.Allocator, io: std.Io, dev: *const cuda.Device, o: Options) !void {
    const cwd = std.Io.Dir.cwd();
    try cwd.createDirPath(io, o.out);
    const entries = try aug.parseLabels(gpa, try u.readFile(gpa, io, o.labels));
    const images = try std.fs.path.join(gpa, &.{ o.data, "images" });
    var start_epoch: usize = 0;
    var iter: usize = 0;
    const source = if (o.resume_from) |path| path else blk: {
        const path = try std.fs.path.join(gpa, &.{ o.out, "init.pth" });
        try ck.save(gpa, io, path, &.{ .{ .key = "meta", .value = .{ .dict = &.{} } }, .{ .key = "state_dict", .value = .{ .dict = try initialState(gpa, o.seed) } } });
        break :blk path;
    };
    const checkpoint = try ck.load(gpa, io, source);
    const model = try net.Scrfd.fromCheckpoint(gpa, checkpoint);
    const n = o.recipe.batch;
    var m = try gpu.Net.init(gpa, dev, model, n, targets.size);
    var lb = try gpu.LossBuffers.init(dev, n);
    if (o.resume_from != null) {
        const meta = checkpoint.dict.get("meta").?.dict;
        start_epoch = @intCast(meta.get("epoch").?.integer);
        iter = @intCast(meta.get("iter").?.integer);
        m.batches = @intCast(iter);
        const momentum = (checkpoint.dict.get("fcs_momentum") orelse return error.CheckpointHasNoMomentum).dict;
        for (m.params.items) |p| try dev.upload(f32, p.momentum, try momentum.get(p.name).?.tensor.floats(gpa));
    }
    const per_epoch = (entries.len + n - 1) / n;
    var out: u.Stdout = undefined;
    out.init(io);
    const w = out.w();
    try w.print("{d} training images, batch {d}, {d} iterations per epoch, epochs {d}..{d}, on {s}\n", .{ entries.len, n, per_epoch, start_epoch + 1, o.recipe.epochs, dev.deviceName() });
    try w.flush();

    var batches = [2]Batch{ try allocBatch(gpa, n), try allocBatch(gpa, n) };
    const order = try gpa.alloc(usize, per_epoch * n);
    var shuffler: std.Random.DefaultPrng = .init(o.seed +% 17);
    const started = std.Io.Timestamp.now(io, .awake);
    var sums = [3]f64{ 0, 0, 0 };
    var logged: usize = 0;
    var first_step = o.resume_from == null;
    for (start_epoch..o.recipe.epochs) |epoch| {
        // Shuffle, padding the last batch with random repeats (as mmdet's GroupSampler pads).
        for (order[0..entries.len], 0..) |*v, i| v.* = i;
        shuffler.random().shuffle(usize, order[0..entries.len]);
        for (order[entries.len..]) |*v| v.* = shuffler.random().uintLessThan(usize, entries.len);
        var pending: ?std.Thread = null;
        var jobs: [2]Prep = undefined;
        jobs[0] = .{ .io = io, .images = images, .entries = entries, .order = order[0..n], .seed = o.seed *% 1000003 +% epoch *% 7919, .ignore_iof = o.recipe.ignore_iof, .batch = &batches[0] };
        try prepare(&jobs[0], o.workers);
        for (0..per_epoch) |k| {
            const cur = k % 2;
            if (k + 1 < per_epoch) {
                const nxt = (k + 1) % 2;
                jobs[nxt] = .{ .io = io, .images = images, .entries = entries, .order = order[(k + 1) * n ..][0..n], .seed = (o.seed *% 1000003 +% epoch *% 7919) ^ ((k + 1) *% 104729), .ignore_iof = o.recipe.ignore_iof, .batch = &batches[nxt] };
                pending = try std.Thread.spawn(.{}, prepare, .{ &jobs[nxt], o.workers });
            }
            const b = &batches[cur];
            var positives: std.ArrayList(gpu.Positive) = .empty;
            defer positives.deinit(gpa);
            var samples: f32 = 0;
            for (b.positives, b.counts) |p, c| {
                try positives.appendSlice(gpa, p);
                samples += @floatFromInt(@max(c, 1));
            }
            try dev.upload(f32, m.out[0], b.input);
            try m.forward(@intCast(n), true);
            try m.zeroGrads();
            const l = try gpu.loss(&m, &lb, @intCast(n), b.label_weight, positives.items, samples);
            try m.backward(@intCast(n));
            try m.sgd(learningRate(epoch, iter, o.recipe.steps), 0.9, 5e-4, first_step);
            first_step = false;
            iter += 1;
            sums[0] += l.cls;
            sums[1] += l.box;
            sums[2] += l.kps;
            logged += 1;
            if (!std.math.isFinite(l.cls + l.box + l.kps)) {
                std.debug.print("loss became non-finite at iteration {d}\n", .{iter});
                return error.Diverged;
            }
            if (pending) |t| {
                t.join();
                pending = null;
                if (jobs[(k + 1) % 2].failed.load(.monotonic)) return error.BatchFailed;
            }
            if (iter % 50 == 0 or k + 1 == per_epoch) {
                const secs = @as(f64, @floatFromInt(started.durationTo(std.Io.Timestamp.now(io, .awake)).nanoseconds)) / std.time.ns_per_s;
                const f: f64 = @floatFromInt(logged);
                try w.print("epoch {d} [{d}/{d}] iter {d}  lr {e:.3}  loss_cls {d:.4}  loss_bbox {d:.4}  loss_kps {d:.4}  {d:.1} img/s\n", .{ epoch + 1, k + 1, per_epoch, iter, learningRate(epoch, iter, o.recipe.steps), sums[0] / f, sums[1] / f, sums[2] / f, @as(f64, @floatFromInt((iter - start_epoch * per_epoch) * n)) / secs });
                try w.flush();
                sums = .{ 0, 0, 0 };
                logged = 0;
            }
            if (o.max_iters) |limit| if (iter >= limit) {
                if (pending) |t| t.join();
                try saveCheckpoint(gpa, io, try std.fs.path.join(gpa, &.{ o.out, "latest.pth" }), &m, epoch, iter);
                return;
            };
        }
        if ((epoch + 1) % o.recipe.interval == 0 or epoch + 1 == o.recipe.epochs) {
            const path = try std.fmt.allocPrint(gpa, "{s}{c}epoch_{d}.pth", .{ o.out, std.fs.path.sep, epoch + 1 });
            try saveCheckpoint(gpa, io, path, &m, epoch + 1, iter);
            try saveCheckpoint(gpa, io, try std.fs.path.join(gpa, &.{ o.out, "latest.pth" }), &m, epoch + 1, iter);
            try w.print("saved {s}\n", .{path});
            try w.flush();
        }
    }
}

pub fn main(init: std.process.Init) !void {
    const gpa = init.arena.allocator();
    const args = try init.minimal.args.toSlice(gpa);
    var dev = try cuda.Device.open(gpa, gpu.ptx);
    if (args.len == 3 and std.mem.eql(u8, args[1], "--check")) return selfCheck(gpa, init.io, &dev, args[2]);
    if (args.len == 4 and std.mem.eql(u8, args[1], "--dump-grads")) return dumpGrads(gpa, init.io, &dev, args[2], args[3]);
    if (args.len == 5 and std.mem.eql(u8, args[1], "--dump-loss")) return dumpLoss(gpa, init.io, &dev, args[2], args[3], args[4]);
    var positional: std.ArrayList([]const u8) = .empty;
    var o: Options = .{ .data = "", .labels = "", .out = "work_dirs/scrfd", .recipe = recipes.curve };
    var epochs: ?usize = null;
    var batch: ?usize = null;
    var i: usize = 1;
    while (i < args.len) : (i += 1) {
        const arg = args[i];
        if (std.mem.startsWith(u8, arg, "--") and i + 1 < args.len) {
            i += 1;
            const v = args[i];
            if (std.mem.eql(u8, arg, "--recipe")) {
                o.recipe = if (std.mem.eql(u8, v, "80k")) recipes.@"80k" else if (std.mem.eql(u8, v, "curve")) recipes.curve else return error.UnknownRecipe;
            } else if (std.mem.eql(u8, arg, "--out")) o.out = v else if (std.mem.eql(u8, arg, "--resume")) o.resume_from = v else if (std.mem.eql(u8, arg, "--seed")) o.seed = try std.fmt.parseInt(u64, v, 10) else if (std.mem.eql(u8, arg, "--workers")) o.workers = try std.fmt.parseInt(usize, v, 10) else if (std.mem.eql(u8, arg, "--epochs")) epochs = try std.fmt.parseInt(usize, v, 10) else if (std.mem.eql(u8, arg, "--batch")) batch = try std.fmt.parseInt(usize, v, 10) else if (std.mem.eql(u8, arg, "--max-iters")) o.max_iters = try std.fmt.parseInt(usize, v, 10) else return error.UnknownOption;
        } else try positional.append(gpa, arg);
    }
    if (positional.items.len != 2) {
        std.debug.print(
            \\usage: train_scrfd <openimages_dir> <labelv2.txt> [--recipe curve|80k] [--out dir] [--epochs N] [--batch N]
            \\                   [--resume checkpoint.pth] [--seed 0] [--workers 16] [--max-iters N]
            \\       train_scrfd --check <checkpoint.pth> | --dump-grads <checkpoint.pth> <dir> | --dump-loss <checkpoint.pth> <openimages_dir> <dir>
            \\
        , .{});
        return error.Usage;
    }
    o.data = positional.items[0];
    o.labels = positional.items[1];
    if (epochs) |e| o.recipe.epochs = e;
    if (batch) |b| o.recipe.batch = b;
    return train(gpa, init.io, &dev, o);
}

test "the learning rate warms up linearly from 0.1% and steps down tenfold" {
    try std.testing.expectApproxEqRel(@as(f64, 1e-5), learningRate(0, 0, .{ 440, 544 }), 1e-12);
    try std.testing.expectApproxEqRel(@as(f64, 0.01 * (1 - 0.5 * 0.999)), learningRate(0, 750, .{ 440, 544 }), 1e-12);
    try std.testing.expectEqual(@as(f64, 0.01), learningRate(3, 5000, .{ 440, 544 }));
    try std.testing.expectApproxEqRel(@as(f64, 1e-3), learningRate(440, 99999, .{ 440, 544 }), 1e-12);
    try std.testing.expectApproxEqRel(@as(f64, 1e-4), learningRate(600, 99999, .{ 440, 544 }), 1e-12);
}

test "the initial state builds the exporter's architecture" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const gpa = arena.allocator();
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const path = try std.fmt.allocPrint(gpa, ".zig-cache/tmp/{s}/init.pth", .{&tmp.sub_path});
    try ck.save(gpa, std.testing.io, path, &.{.{ .key = "state_dict", .value = .{ .dict = try initialState(gpa, 0) } }});
    const model = try net.Scrfd.fromCheckpoint(gpa, try ck.load(gpa, std.testing.io, path));
    try std.testing.expectEqual(@as(usize, 66), model.steps.len);
    // The same names as the synthetic fixture shaped like the shipped model.
    const fixture = try net.Scrfd.fromCheckpoint(gpa, try ck.load(gpa, std.testing.io, "tools/dataset/testdata/scrfd.pt"));
    for (model.steps, fixture.steps) |a, b| if (a == .conv) {
        try std.testing.expectEqualStrings(b.conv.conv.name, a.conv.conv.name);
        try std.testing.expectEqual(b.conv.conv.weight.len, a.conv.conv.weight.len);
    };
}
