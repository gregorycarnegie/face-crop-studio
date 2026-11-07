//! Train SCRFD-500M natively on an NVIDIA GPU (work in progress: network checks only so far).
//!
//!   zig run -O ReleaseFast -lc tools/dataset/train_scrfd.zig -- --check <checkpoint.pth>
//!   zig run -O ReleaseFast -lc tools/dataset/train_scrfd.zig -- --dump-grads <checkpoint.pth> <dir>
//!
//! `--check` compares the GPU network's eval-mode outputs with `scrfd.zig`'s CPU reference.
//! `--dump-grads` writes a training-mode backward pass (input, loss weights, every parameter's
//! gradient) for comparison with torch autograd on MMDetection's own model: finite differences
//! cannot check a network this deep, whose small-batch BatchNorms make them non-linear.
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

pub fn main(init: std.process.Init) !void {
    const gpa = init.arena.allocator();
    const args = try init.minimal.args.toSlice(gpa);
    var dev = try cuda.Device.open(gpa, gpu.ptx);
    if (args.len == 3 and std.mem.eql(u8, args[1], "--check")) return selfCheck(gpa, init.io, &dev, args[2]);
    if (args.len == 4 and std.mem.eql(u8, args[1], "--dump-grads")) return dumpGrads(gpa, init.io, &dev, args[2], args[3]);
    if (args.len == 5 and std.mem.eql(u8, args[1], "--dump-loss")) return dumpLoss(gpa, init.io, &dev, args[2], args[3], args[4]);
    std.debug.print("usage: train_scrfd --check <checkpoint.pth> | --dump-grads <checkpoint.pth> <dir>\n", .{});
    return error.Usage;
}
