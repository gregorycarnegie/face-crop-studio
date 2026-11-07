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

pub fn main(init: std.process.Init) !void {
    const gpa = init.arena.allocator();
    const args = try init.minimal.args.toSlice(gpa);
    var dev = try cuda.Device.open(gpa, gpu.ptx);
    if (args.len == 3 and std.mem.eql(u8, args[1], "--check")) return selfCheck(gpa, init.io, &dev, args[2]);
    if (args.len == 4 and std.mem.eql(u8, args[1], "--dump-grads")) return dumpGrads(gpa, init.io, &dev, args[2], args[3]);
    std.debug.print("usage: train_scrfd --check <checkpoint.pth> | --dump-grads <checkpoint.pth> <dir>\n", .{});
    return error.Usage;
}
