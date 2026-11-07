//! Train a small model that takes a face crop and returns its two eye points, on an NVIDIA GPU.
//!
//!   zig run -O ReleaseFast -lc tools/dataset/train_eye_refiner.zig tools/dataset/vendor/stb.c -- <data_dir> [--epochs 300] [--out eye_refiner.pt]
//!       [--batch 64] [--angle-weight 0.5] [--val-fraction 0.15] [--seed 0] [--check]
//!
//! The label-count curve (CURVE_RESULTS.md) found that training a whole detector on 2,482 images
//! could not beat YuNet's eye points, and that more eye labels did not help: point distance
//! improved 39% from 400 to 1988 labels while the eye-*line* angle stayed flat at ~5.3 degrees.
//! Detection was never the problem. This attacks the narrow task instead -- given a face box,
//! where are the eyes -- which is small enough that 1,988 labelled pairs may actually suffice.
//!
//! Two choices follow directly from that finding:
//!
//! * **Rotation augmentation, up to 30 degrees.** Angle error stayed flat because 2,482 images
//!   contain little roll variety, so more labels on them taught eye appearance rather than eye
//!   geometry. Rotating crops manufactures the variety the corpus lacks.
//! * **An auxiliary angle loss.** The metric that matters is the tilt of the line between the
//!   points, and it was the metric that would not move, so it is optimised directly rather than
//!   left to follow from point regression.
//!
//! Trained from scratch rather than fine-tuned from an ImageNet backbone. A pretrained backbone
//! would likely need fewer labels, but this project has been careful about what it can ship, and
//! pretrained weights carry their own terms; raising that as a choice beats quietly depending on
//! it. If from-scratch stalls, that is the first thing to revisit.
//!
//! Only faces whose eyes were clicked and whose side is known are used: `steep_roll` faces carry
//! points but no reliable left/right identity, and training on them would teach the ambiguity.
//!
//! The recipe is the PyTorch original's: AdamW (lr 3e-3, weight decay 1e-4) under OneCycleLR with
//! its momentum cycling, smooth-L1 points plus 0.5 x (1 - cos) angle loss, BatchNorm in training
//! mode, PyTorch's default initialisation, and checkpoint selection on a validation slice of the
//! training *images* -- the same slice, since Python's `random.Random(1234).shuffle` is
//! reproduced. The network runs on the GPU through the NVIDIA driver alone (`cuda.zig`, kernels
//! in `gpu/refiner_kernels.zig`); crops are cut on the CPU with OpenCV's warpAffine arithmetic.
//! The checkpoint is a PyTorch ZIP that `export_eye_refiner.zig` and `torch.load` both read.
//! `--check` compares the GPU network against `refiner.zig` and finite differences, then exits.
const std = @import("std");
const u = @import("util.zig");
const img = @import("image.zig");
const ck = @import("checkpoint.zig");
const cuda = @import("cuda.zig");
const net = @import("refiner.zig");

const size = net.size;
const pixels = size * size * 3;
const ptx = @embedFile("gpu/refiner_kernels.ptx");
const lanes = 256; // gpu/refiner_kernels.zig
const Buf = cuda.Buffer(f32);
const trainer = "tools/dataset/train_eye_refiner.zig"; // recorded in the checkpoint as `trained_by`

// ── Data ──────────────────────────────────────────────────────────────────────

const Face = struct { name: []const u8, box: [4]f64, eyes: [2][2]f32 };

fn usable(a: std.json.Value) bool {
    return u.num(u.field(a, "num_keypoints")) == 2 and !u.truthy(u.field(u.field(a, "fcs"), "steep_roll"));
}

/// Faces of one split; with `hold_out`, the training images are divided by *image* -- two faces
/// from one photo share a camera, a scene and often a subject -- into "fit" and "val".
fn loadFaces(gpa: std.mem.Allocator, io: std.Io, data: []const u8, split: []const u8, hold_out: f64, val: bool) ![]Face {
    const doc = try u.readJson(gpa, io, try std.fmt.allocPrint(gpa, "{s}{c}face_keypoints_{s}.json", .{ data, std.fs.path.sep, split }));
    var names: std.AutoHashMapUnmanaged(i64, []const u8) = .empty;
    var unique: std.StringArrayHashMapUnmanaged(void) = .empty;
    for (u.field(doc, "images").array.items) |image| {
        const name = u.str(u.field(image, "file_name"));
        try names.put(gpa, u.field(image, "id").integer, name);
        try unique.put(gpa, name, {});
    }
    var held: std.StringHashMapUnmanaged(void) = .empty;
    if (hold_out > 0) {
        const sorted = try gpa.dupe([]const u8, unique.keys());
        std.mem.sort([]const u8, sorted, {}, u.lessStr);
        var rng: u.PyRandom = .init(1234);
        rng.shuffle([]const u8, sorted);
        for (sorted[0..@intFromFloat(@trunc(@as(f64, @floatFromInt(sorted.len)) * hold_out))]) |n| try held.put(gpa, n, {});
    }
    var faces: std.ArrayList(Face) = .empty;
    for (u.field(doc, "annotations").array.items) |a| {
        if (!usable(a)) continue;
        const name = names.get(u.field(a, "image_id").integer) orelse return error.UnknownImage;
        if (hold_out > 0 and val != held.contains(name)) continue;
        const b = u.field(a, "bbox").array.items;
        const k = u.field(a, "keypoints").array.items;
        try faces.append(gpa, .{
            .name = name,
            .box = .{ u.num(b[0]), u.num(b[1]), u.num(b[2]), u.num(b[3]) },
            .eyes = .{ .{ @floatCast(u.num(k[0])), @floatCast(u.num(k[1])) }, .{ @floatCast(u.num(k[3])), @floatCast(u.num(k[4])) } },
        });
    }
    return faces.toOwnedSlice(gpa);
}

fn uniform(r: std.Random, lo: f64, hi: f64) f64 {
    return lo + (hi - lo) * r.float(f64);
}

/// The crop and its target (eye points as fractions of the crop), augmented when training:
/// scale 1.10-1.55x the longer side, a 6% shift, up to 30 degrees of roll, a mirror that swaps
/// which eye is which, and brightness/contrast jitter.
fn cutCrop(gpa: std.mem.Allocator, frame: img.Image, face: Face, rng: ?std.Random, out: *[pixels]u8, target: *[4]f32) !void {
    const x, const y, const w, const h = face.box;
    var centre = [2]f32{ @floatCast(x + w / 2), @floatCast(y + h / 2) };
    var side = @max(w, h);
    var angle: f64 = 0;
    if (rng) |r| {
        side *= uniform(r, 1.10, 1.55);
        centre[0] += @floatCast(uniform(r, -0.06, 0.06) * w);
        centre[1] += @floatCast(uniform(r, -0.06, 0.06) * h);
        angle = uniform(r, -30, 30);
    } else side *= 1.25;
    // Rotate about the crop centre and scale to the network's input in one affine, so the eye
    // points can be carried through with the same matrix rather than re-derived.
    var m = img.rotationMatrix(centre[0], centre[1], angle, size / side);
    m[2] += size / 2.0 - @as(f64, centre[0]);
    m[5] += size / 2.0 - @as(f64, centre[1]);
    const crop = try img.warpAffine(gpa, frame, m, size, size);
    defer crop.deinit(gpa);
    var points: [2][2]f64 = undefined;
    for (face.eyes, &points) |e, *p| p.* = .{ m[0] * e[0] + m[1] * e[1] + m[2], m[3] * e[0] + m[4] * e[1] + m[5] };
    @memcpy(out, crop.pixels);
    if (rng) |r| {
        if (r.float(f64) < 0.5) {
            for (0..size) |row| std.mem.reverse([3]u8, std.mem.bytesAsSlice([3]u8, out[row * size * 3 ..][0 .. size * 3]));
            for (&points) |*p| p[0] = size - p[0];
            std.mem.swap([2]f64, &points[0], &points[1]); // mirroring swaps which eye is which
        }
        if (r.float(f64) < 0.5) {
            const contrast: f32 = @floatCast(uniform(r, 0.7, 1.3));
            const brightness: f32 = @floatCast(uniform(r, -20, 20));
            for (out) |*v| v.* = @intFromFloat(std.math.clamp(@as(f32, @floatFromInt(v.*)) * contrast + brightness, 0, 255));
        }
    }
    target.* = .{ @floatCast(points[0][0] / size), @floatCast(points[0][1] / size), @floatCast(points[1][0] / size), @floatCast(points[1][1] / size) };
}

const CropJob = struct {
    io: std.Io,
    images: []const u8,
    faces: []const Face,
    order: []const usize,
    seed: u64,
    augment: bool,
    crops: []u8,
    targets: [][4]f32,
    next: std.atomic.Value(usize) = .init(0),
    failed: std.atomic.Value(bool) = .init(false),
};

fn cropWorker(job: *CropJob) void {
    while (true) {
        const slot = job.next.fetchAdd(1, .monotonic);
        if (slot >= job.order.len) return;
        var arena: std.heap.ArenaAllocator = .init(std.heap.smp_allocator);
        defer arena.deinit();
        const gpa = arena.allocator();
        const face = job.faces[job.order[slot]];
        // A missing file must not silently become a black face.
        const frame = img.load(gpa, job.io, std.fs.path.join(gpa, &.{ job.images, face.name }) catch return fail(job), .apply_exif) catch |e| {
            std.debug.print("cannot read {s}: {s}\n", .{ face.name, @errorName(e) });
            return fail(job);
        };
        var prng: std.Random.DefaultPrng = .init(job.seed ^ (@as(u64, slot) *% 0x9E3779B97F4A7C15));
        cutCrop(gpa, frame, face, if (job.augment) prng.random() else null, job.crops[slot * pixels ..][0..pixels], &job.targets[slot]) catch return fail(job);
    }
}

fn fail(job: *CropJob) void {
    job.failed.store(true, .monotonic);
    job.next.store(job.order.len, .monotonic);
}

/// Cut `order`'s faces in parallel, slot i holding face order[i].
fn cutAll(io: std.Io, images: []const u8, faces: []const Face, order: []const usize, seed: u64, augment: bool, crops: []u8, targets: [][4]f32) !void {
    var job: CropJob = .{ .io = io, .images = images, .faces = faces, .order = order, .seed = seed, .augment = augment, .crops = crops, .targets = targets };
    var threads: [64]std.Thread = undefined;
    const count = @min(threads.len, @max(1, std.Thread.getCpuCount() catch 4));
    for (threads[0..count]) |*t| t.* = try std.Thread.spawn(.{}, cropWorker, .{&job});
    for (threads[0..count]) |t| t.join();
    if (job.failed.load(.monotonic)) return error.CropFailed;
}

// ── Network on the GPU ────────────────────────────────────────────────────────

const Param = struct {
    value: Buf,
    grad: Buf,
    m: Buf,
    v: Buf,
    shape: []const usize,

    fn init(dev: *const cuda.Device, gpa: std.mem.Allocator, shape: []const usize, host: []const f32) !Param {
        const p: Param = .{ .value = try dev.alloc(f32, host.len), .grad = try dev.alloc(f32, host.len), .m = try dev.alloc(f32, host.len), .v = try dev.alloc(f32, host.len), .shape = try gpa.dupe(usize, shape) };
        try dev.upload(f32, p.value, host);
        try dev.zero(p.m);
        try dev.zero(p.v);
        return p;
    }
};

const Layer = struct {
    cin: u32,
    cout: u32,
    stride: u32,
    side_in: u32,
    side_out: u32,
    weight: Param,
    gamma: Param,
    beta: Param,
    running_mean: Buf,
    running_var: Buf,
    mean: Buf,
    variance: Buf,
    sqdev: Buf,
    z: Buf, // convolution output
    y: Buf, // after BatchNorm and ReLU
};

const Kernels = struct {
    conv_fwd: cuda.Function,
    conv_bwd_data: cuda.Function,
    conv_bwd_weight: cuda.Function,
    channel_sum: cuda.Function,
    bn_stats: cuda.Function,
    scale: cuda.Function,
    bn_relu: cuda.Function,
    bn_bwd_reduce: cuda.Function,
    bn_bwd_dx: cuda.Function,
    avgpool_fwd: cuda.Function,
    avgpool_bwd: cuda.Function,
    linear_fwd: cuda.Function,
    relu_fwd: cuda.Function,
    relu_bwd: cuda.Function,
    linear_bwd_data: cuda.Function,
    linear_bwd_weight: cuda.Function,
    adamw: cuda.Function,
    crops_to_input: cuda.Function,
};

/// PyTorch's default initialisation: kaiming_uniform(a=sqrt(5)) for weights, i.e. U(+-1/sqrt(fan_in)),
/// the same bound for Linear biases, and BatchNorm at gamma 1, beta 0.
fn uniformInit(gpa: std.mem.Allocator, r: std.Random, count: usize, fan_in: usize) ![]f32 {
    const bound = 1 / @sqrt(@as(f32, @floatFromInt(fan_in)));
    const out = try gpa.alloc(f32, count);
    for (out) |*v| v.* = (r.float(f32) * 2 - 1) * bound;
    return out;
}

fn filled(gpa: std.mem.Allocator, count: usize, value: f32) ![]f32 {
    const out = try gpa.alloc(f32, count);
    @memset(out, value);
    return out;
}

const Net = struct {
    dev: *const cuda.Device,
    k: Kernels,
    batch: usize,
    layers: [10]Layer,
    fc1_w: Param,
    fc1_b: Param,
    fc2_w: Param,
    fc2_b: Param,
    crops: cuda.Buffer(u8),
    input: Buf,
    pooled: Buf,
    hidden: Buf,
    act: Buf,
    pred: Buf,
    g_a: Buf,
    g_b: Buf,
    g_pool: Buf,
    g_hidden: Buf,
    g_act: Buf,
    g_pred: Buf,
    /// Training forward passes so far: BatchNorm's num_batches_tracked.
    batches: i64 = 0,

    fn init(gpa: std.mem.Allocator, dev: *const cuda.Device, batch: usize, seed: u64) !Net {
        var k: Kernels = undefined;
        inline for (comptime std.meta.fieldNames(Kernels)) |name| @field(k, name) = try dev.function(name);
        var prng: std.Random.DefaultPrng = .init(seed);
        const r = prng.random();
        var self: Net = undefined;
        self.dev = dev;
        self.k = k;
        self.batch = batch;
        self.batches = 0;
        var side: u32 = size;
        var largest: usize = 0;
        for (&self.layers, 0..) |*l, i| {
            const cin: u32 = @intCast(net.channels[i]);
            const cout: u32 = @intCast(net.channels[i + 1]);
            const stride: u32 = if (i % 2 == 0) 2 else 1;
            const out_side = (side - 1) / stride + 1;
            const activation = batch * cout * out_side * out_side;
            largest = @max(largest, activation, batch * cin * side * side);
            l.* = .{
                .cin = cin,
                .cout = cout,
                .stride = stride,
                .side_in = side,
                .side_out = out_side,
                .weight = try Param.init(dev, gpa, &.{ cout, cin, 3, 3 }, try uniformInit(gpa, r, cout * cin * 9, cin * 9)),
                .gamma = try Param.init(dev, gpa, &.{cout}, try filled(gpa, cout, 1)),
                .beta = try Param.init(dev, gpa, &.{cout}, try filled(gpa, cout, 0)),
                .running_mean = try dev.alloc(f32, cout),
                .running_var = try dev.alloc(f32, cout),
                .mean = try dev.alloc(f32, cout),
                .variance = try dev.alloc(f32, cout),
                .sqdev = try dev.alloc(f32, cout),
                .z = try dev.alloc(f32, activation),
                .y = try dev.alloc(f32, activation),
            };
            try dev.zero(l.running_mean);
            try dev.upload(f32, l.running_var, try filled(gpa, cout, 1));
            side = out_side;
        }
        self.fc1_w = try Param.init(dev, gpa, &.{ 128, 256 }, try uniformInit(gpa, r, 128 * 256, 256));
        self.fc1_b = try Param.init(dev, gpa, &.{128}, try uniformInit(gpa, r, 128, 256));
        self.fc2_w = try Param.init(dev, gpa, &.{ 4, 128 }, try uniformInit(gpa, r, 4 * 128, 128));
        self.fc2_b = try Param.init(dev, gpa, &.{4}, try uniformInit(gpa, r, 4, 128));
        self.crops = try dev.alloc(u8, batch * pixels);
        self.input = try dev.alloc(f32, batch * pixels);
        self.pooled = try dev.alloc(f32, batch * 256);
        self.hidden = try dev.alloc(f32, batch * 128);
        self.act = try dev.alloc(f32, batch * 128);
        self.pred = try dev.alloc(f32, batch * 4);
        self.g_a = try dev.alloc(f32, largest);
        self.g_b = try dev.alloc(f32, largest);
        self.g_pool = try dev.alloc(f32, batch * 256);
        self.g_hidden = try dev.alloc(f32, batch * 128);
        self.g_act = try dev.alloc(f32, batch * 128);
        self.g_pred = try dev.alloc(f32, batch * 4);
        return self;
    }

    /// Every trainable parameter, in PyTorch's `model.parameters()` order.
    fn paramList(self: *Net) [34]*Param {
        var out: [34]*Param = undefined;
        for (&self.layers, 0..) |*l, i| out[i * 3 ..][0..3].* = .{ &l.weight, &l.gamma, &l.beta };
        out[30..34].* = .{ &self.fc1_w, &self.fc1_b, &self.fc2_w, &self.fc2_b };
        return out;
    }

    /// Upload `n` crops and run the network; `train` uses batch statistics and updates the
    /// running averages, as nn.BatchNorm2d does in train() mode.
    fn forward(self: *Net, crops: []const u8, train: bool) !void {
        const d = self.dev;
        const n: u32 = @intCast(crops.len / pixels);
        try d.upload(u8, self.crops, crops);
        try d.launch(self.k.crops_to_input, n * pixels, .{ n, @as(u32, size), self.crops, self.input });
        var x = self.input;
        for (&self.layers) |*l| {
            const hw = l.side_out * l.side_out;
            const total = n * l.cout * hw;
            try d.launch(self.k.conv_fwd, total, .{ n, l.cin, l.side_in, l.side_in, l.cout, l.side_out, l.side_out, l.stride, x, l.weight.value, l.z });
            if (train) {
                const m: f32 = @floatFromInt(n * hw);
                try d.zero(l.mean);
                try d.launch(self.k.channel_sum, l.cout * lanes, .{ n, l.cout, hw, l.z, @as(cuda.Ptr, 0), l.mean });
                try d.launch(self.k.scale, l.cout, .{ l.cout, 1 / m, l.mean });
                try d.zero(l.sqdev);
                try d.launch(self.k.channel_sum, l.cout * lanes, .{ n, l.cout, hw, l.z, l.mean, l.sqdev });
                try d.launch(self.k.bn_stats, l.cout, .{ l.cout, m, l.sqdev, l.mean, l.variance, l.running_mean, l.running_var });
                try d.launch(self.k.bn_relu, total, .{ total, l.cout, hw, l.z, l.mean, l.variance, l.gamma.value, l.beta.value, l.y });
            } else {
                try d.launch(self.k.bn_relu, total, .{ total, l.cout, hw, l.z, l.running_mean, l.running_var, l.gamma.value, l.beta.value, l.y });
            }
            x = l.y;
        }
        if (train) self.batches += 1;
        try d.launch(self.k.avgpool_fwd, n * 256, .{ n * 256, @as(u32, 16), x, self.pooled });
        try d.launch(self.k.linear_fwd, n * 128, .{ n, @as(u32, 256), @as(u32, 128), self.pooled, self.fc1_w.value, self.fc1_b.value, self.hidden });
        try d.launch(self.k.relu_fwd, n * 128, .{ n * 128, self.hidden, self.act });
        try d.launch(self.k.linear_fwd, n * 4, .{ n, @as(u32, 128), @as(u32, 4), self.act, self.fc2_w.value, self.fc2_b.value, self.pred });
    }

    /// Gradients of every parameter for the last training forward pass, given d(loss)/d(pred).
    fn backward(self: *Net, n: u32, g_pred: []const f32) !void {
        const d = self.dev;
        try d.upload(f32, self.g_pred, g_pred);
        for (self.paramList()) |p| try d.zero(p.grad);
        try d.launch(self.k.linear_bwd_weight, 4 * 128 + 4, .{ n, @as(u32, 128), @as(u32, 4), self.act, self.g_pred, self.fc2_w.grad, self.fc2_b.grad });
        try d.launch(self.k.linear_bwd_data, n * 128, .{ n, @as(u32, 128), @as(u32, 4), self.g_pred, self.fc2_w.value, self.g_act });
        try d.launch(self.k.relu_bwd, n * 128, .{ n * 128, self.act, self.g_act, self.g_hidden });
        try d.launch(self.k.linear_bwd_weight, 128 * 256 + 128, .{ n, @as(u32, 256), @as(u32, 128), self.pooled, self.g_hidden, self.fc1_w.grad, self.fc1_b.grad });
        try d.launch(self.k.linear_bwd_data, n * 256, .{ n, @as(u32, 256), @as(u32, 128), self.g_hidden, self.fc1_w.value, self.g_pool });
        try d.launch(self.k.avgpool_bwd, n * 256 * 16, .{ n * 256 * 16, @as(u32, 16), self.g_pool, self.g_a });
        var i: usize = self.layers.len;
        while (i > 0) {
            i -= 1;
            const l = &self.layers[i];
            const hw = l.side_out * l.side_out;
            const total = n * l.cout * hw;
            const m: f32 = @floatFromInt(n * hw);
            // g_a: gradient into this layer's output; g_b: into its convolution's output.
            try d.launch(self.k.bn_bwd_reduce, l.cout * lanes, .{ n, l.cout, hw, l.z, l.y, self.g_a, l.mean, l.variance, l.gamma.grad, l.beta.grad });
            try d.launch(self.k.bn_bwd_dx, total, .{ total, l.cout, hw, m, l.z, l.y, self.g_a, l.mean, l.variance, l.gamma.value, l.gamma.grad, l.beta.grad, self.g_b });
            const x = if (i == 0) self.input else self.layers[i - 1].y;
            try d.launch(self.k.conv_bwd_weight, l.cout * l.cin * 9 * n, .{ n, l.cin, l.side_in, l.side_in, l.cout, l.side_out, l.side_out, l.stride, x, self.g_b, l.weight.grad });
            if (i > 0) try d.launch(self.k.conv_bwd_data, n * l.cin * l.side_in * l.side_in, .{ n, l.cin, l.side_in, l.side_in, l.cout, l.side_out, l.side_out, l.stride, self.g_b, l.weight.value, self.g_a });
        }
    }

    fn step(self: *Net, lr: f64, beta1: f64, t: u64) !void {
        const bc1: f32 = @floatCast(1 - std.math.pow(f64, beta1, @floatFromInt(t)));
        const bc2: f32 = @floatCast(@sqrt(1 - std.math.pow(f64, 0.999, @floatFromInt(t))));
        for (self.paramList()) |p| {
            const count: u32 = @intCast(p.value.len);
            try self.dev.launch(self.k.adamw, count, .{ count, p.value, p.grad, p.m, p.v, @as(f32, @floatCast(lr)), @as(f32, @floatCast(beta1)), @as(f32, 0.999), @as(f32, 1e-4), bc1, bc2 });
        }
    }

    fn predictions(self: *Net, n: usize, out: []f32) !void {
        try self.dev.download(f32, out[0 .. n * 4], self.pred);
    }

    /// The state_dict, in nn.Module order, downloaded to host memory.
    fn stateDict(self: *Net, gpa: std.mem.Allocator) ![]const ck.OutEntry {
        var entries: std.ArrayList(ck.OutEntry) = .empty;
        const d = self.dev;
        const Helper = struct {
            fn tensor(g: std.mem.Allocator, dev: *const cuda.Device, buf: Buf, shape: []const usize) !ck.Out {
                const host = try g.alloc(f32, buf.len);
                try dev.download(f32, host, buf);
                return .{ .tensor = .{ .dtype = .float32, .shape = shape, .data = std.mem.sliceAsBytes(host) } };
            }
        };
        const counter = try gpa.alloc(i64, 1);
        counter[0] = self.batches;
        for (&self.layers, 0..) |*l, i| {
            const conv = i * 3;
            const bn = conv + 1;
            const c = try gpa.dupe(usize, &.{l.cout});
            try entries.append(gpa, .{ .key = try std.fmt.allocPrint(gpa, "features.{d}.weight", .{conv}), .value = try Helper.tensor(gpa, d, l.weight.value, l.weight.shape) });
            try entries.append(gpa, .{ .key = try std.fmt.allocPrint(gpa, "features.{d}.weight", .{bn}), .value = try Helper.tensor(gpa, d, l.gamma.value, c) });
            try entries.append(gpa, .{ .key = try std.fmt.allocPrint(gpa, "features.{d}.bias", .{bn}), .value = try Helper.tensor(gpa, d, l.beta.value, c) });
            try entries.append(gpa, .{ .key = try std.fmt.allocPrint(gpa, "features.{d}.running_mean", .{bn}), .value = try Helper.tensor(gpa, d, l.running_mean, c) });
            try entries.append(gpa, .{ .key = try std.fmt.allocPrint(gpa, "features.{d}.running_var", .{bn}), .value = try Helper.tensor(gpa, d, l.running_var, c) });
            try entries.append(gpa, .{ .key = try std.fmt.allocPrint(gpa, "features.{d}.num_batches_tracked", .{bn}), .value = .{ .tensor = .{ .dtype = .int64, .shape = &.{}, .data = std.mem.sliceAsBytes(counter) } } });
        }
        for ([_]*Param{ &self.fc1_w, &self.fc1_b, &self.fc2_w, &self.fc2_b }, [_][]const u8{ "head.2.weight", "head.2.bias", "head.4.weight", "head.4.bias" }) |p, name| {
            try entries.append(gpa, .{ .key = name, .value = try Helper.tensor(gpa, d, p.value, p.shape) });
        }
        return entries.toOwnedSlice(gpa);
    }

    /// Restore parameters and running statistics from a `stateDict` snapshot.
    fn restore(self: *Net, state: []const ck.OutEntry) !void {
        var i: usize = 0;
        const d = self.dev;
        for (&self.layers) |*l| {
            for ([_]Buf{ l.weight.value, l.gamma.value, l.beta.value, l.running_mean, l.running_var }) |buf| {
                try d.upload(f32, buf, std.mem.bytesAsSlice(f32, @as([]align(4) const u8, @alignCast(state[i].value.tensor.data))));
                i += 1;
            }
            i += 1; // num_batches_tracked
        }
        for ([_]*Param{ &self.fc1_w, &self.fc1_b, &self.fc2_w, &self.fc2_b }) |p| {
            try d.upload(f32, p.value, std.mem.bytesAsSlice(f32, @as([]align(4) const u8, @alignCast(state[i].value.tensor.data))));
            i += 1;
        }
    }
};

// ── Loss, schedule, evaluation ────────────────────────────────────────────────

fn angleOf(p: []const f32) f64 {
    return std.math.atan2(@as(f64, p[3] - p[1]), @as(f64, p[2] - p[0]));
}

/// smooth_l1(beta=0.02) over the four coordinates plus `angle_weight` * (1 - cos) of the
/// eye-line angle, both means over the batch; fills d(loss)/d(pred).
fn lossAndGrad(pred: []const f32, target: []const f32, angle_weight: f64, grad: []f32) f64 {
    const n = pred.len / 4;
    const beta = 0.02;
    var points: f64 = 0;
    var angles: f64 = 0;
    const per = 1 / @as(f64, @floatFromInt(pred.len));
    for (pred, target, grad) |p, t, *g| {
        const diff: f64 = @as(f64, p) - t;
        if (@abs(diff) < beta) {
            points += 0.5 * diff * diff / beta;
            g.* = @floatCast(diff / beta * per);
        } else {
            points += @abs(diff) - 0.5 * beta;
            g.* = @floatCast(std.math.sign(diff) * per);
        }
    }
    for (0..n) |b| {
        const p = pred[b * 4 ..][0..4];
        const t = target[b * 4 ..][0..4];
        const delta = angleOf(p) - angleOf(t);
        angles += 1 - @cos(delta);
        // d/dθ (1 - cos(θ - θt)) = sin(θ - θt); θ = atan2(dy, dx) with dy = p3 - p1, dx = p2 - p0.
        const dtheta = angle_weight * @sin(delta) / @as(f64, @floatFromInt(n));
        const dx: f64 = @as(f64, p[2]) - p[0];
        const dy: f64 = @as(f64, p[3]) - p[1];
        const r2 = dx * dx + dy * dy;
        if (r2 == 0) continue;
        const gx = dtheta * -dy / r2;
        const gy = dtheta * dx / r2;
        grad[b * 4 + 0] -= @floatCast(gx);
        grad[b * 4 + 2] += @floatCast(gx);
        grad[b * 4 + 1] -= @floatCast(gy);
        grad[b * 4 + 3] += @floatCast(gy);
    }
    return points * per + angle_weight * angles / @as(f64, @floatFromInt(n));
}

/// torch.optim.lr_scheduler.OneCycleLR's learning rate and Adam beta1 at `step` (cosine
/// annealing, pct_start 0.3, div_factor 25, final_div_factor 1e4, momentum 0.85-0.95).
fn oneCycle(step: usize, total: usize, max_lr: f64) struct { f64, f64 } {
    const initial = max_lr / 25;
    const final = initial / 1e4;
    const s: f64 = @floatFromInt(step);
    const warm_end = 0.3 * @as(f64, @floatFromInt(total)) - 1;
    const end: f64 = @floatFromInt(total - 1);
    const anneal = struct {
        fn f(a: f64, b: f64, pct: f64) f64 {
            return b + (a - b) / 2 * (@cos(std.math.pi * pct) + 1);
        }
    }.f;
    if (s <= warm_end) {
        const pct = s / warm_end;
        return .{ anneal(initial, max_lr, pct), anneal(0.95, 0.85, pct) };
    }
    const pct = (s - warm_end) / (end - warm_end);
    return .{ anneal(max_lr, final, pct), anneal(0.85, 0.95, pct) };
}

const Stats = struct { angle_median: f64, within_5deg: f64, distance_median: f64 };

fn evaluate(gpa: std.mem.Allocator, model: *Net, crops: []const u8, targets: []const [4]f32) !Stats {
    const count = targets.len;
    const angles = try gpa.alloc(f64, count);
    defer gpa.free(angles);
    const distances = try gpa.alloc(f64, count * 2);
    defer gpa.free(distances);
    var pred: [256 * 4]f32 = undefined;
    var start: usize = 0;
    while (start < count) : (start += model.batch) {
        const n = @min(model.batch, count - start);
        try model.forward(crops[start * pixels ..][0 .. n * pixels], false);
        try model.predictions(n, &pred);
        for (0..n) |b| {
            const p = pred[b * 4 ..][0..4];
            const t = &targets[start + b];
            const diff = @mod(@abs((angleOf(p) - angleOf(t)) * (180.0 / std.math.pi)), 360);
            angles[start + b] = @min(diff, 360 - diff);
            // Crop-relative distance; the crop is 1.25x the longer box side, so scale back.
            for (0..2) |slot| distances[(start + b) * 2 + slot] = std.math.hypot(@as(f64, p[slot * 2]) - t[slot * 2], @as(f64, p[slot * 2 + 1]) - t[slot * 2 + 1]) * 1.25;
        }
    }
    std.mem.sort(f64, angles, {}, std.sort.asc(f64));
    std.mem.sort(f64, distances, {}, std.sort.asc(f64));
    var within: usize = 0;
    for (angles) |a| within += @intFromBool(a <= 5);
    return .{ .angle_median = angles[count / 2], .within_5deg = @as(f64, @floatFromInt(within)) / @as(f64, @floatFromInt(count)), .distance_median = distances[distances.len / 2] };
}

fn statsDict(gpa: std.mem.Allocator, s: Stats) !ck.Out {
    return .{ .dict = try gpa.dupe(ck.OutEntry, &.{
        .{ .key = "angle_median", .value = .{ .float = s.angle_median } },
        .{ .key = "within_5deg", .value = .{ .float = s.within_5deg } },
        .{ .key = "distance_median", .value = .{ .float = s.distance_median } },
    }) };
}

fn percent(gpa: std.mem.Allocator, v: f64) ![]const u8 {
    return std.fmt.allocPrint(gpa, "{s}%", .{try u.fixedAlloc(gpa, v * 100, 1)});
}

// ── Self-check ────────────────────────────────────────────────────────────────

/// The GPU network against `refiner.zig`'s CPU reference (eval mode), and its gradients against
/// central finite differences (train mode).
fn selfCheck(gpa: std.mem.Allocator, io: std.Io, dev: *const cuda.Device) !void {
    const n = 4;
    var model = try Net.init(gpa, dev, n, 7);
    var prng: std.Random.DefaultPrng = .init(11);
    const r = prng.random();
    const crops = try gpa.alloc(u8, n * pixels);
    for (crops) |*c| c.* = r.int(u8);
    // Give BatchNorm non-trivial running statistics and affine terms first.
    for (&model.layers) |*l| {
        try dev.upload(f32, l.running_mean, try uniformInit(gpa, r, l.cout, 4));
        const vars = try gpa.alloc(f32, l.cout);
        for (vars) |*v| v.* = 0.5 + r.float(f32);
        try dev.upload(f32, l.running_var, vars);
        const gammas = try gpa.alloc(f32, l.cout);
        for (gammas) |*v| v.* = 0.5 + r.float(f32);
        try dev.upload(f32, l.gamma.value, gammas);
        try dev.upload(f32, l.beta.value, try uniformInit(gpa, r, l.cout, 16));
    }

    // Eval mode against the CPU reference, through the same checkpoint the trainer writes.
    try model.forward(crops, false);
    var gpu: [n * 4]f32 = undefined;
    try model.predictions(n, &gpu);
    const state = try model.stateDict(gpa);
    try std.Io.Dir.cwd().createDirPath(io, ".zig-cache");
    const path = ".zig-cache/refiner-selfcheck.pt";
    try ck.save(gpa, io, path, &.{ .{ .key = "model", .value = .{ .dict = state } }, .{ .key = "size", .value = .{ .int = size } } });
    const reference = try net.Refiner.fromCheckpoint(gpa, try ck.load(gpa, io, path));
    var worst: f64 = 0;
    for (0..n) |b| {
        const input = try gpa.alloc(f32, pixels);
        for (0..3) |c| for (0..size * size) |p| {
            input[c * size * size + p] = (@as(f32, @floatFromInt(crops[(b * size * size + p) * 3 + c])) - 127.5) / 128.0;
        };
        const cpu = try reference.predict(gpa, input);
        for (cpu, gpu[b * 4 ..][0..4]) |a, g| worst = @max(worst, @abs(@as(f64, a) - g));
    }
    std.debug.print("eval forward vs refiner.zig: worst difference {e}\n", .{worst});
    if (worst > 1e-3) return error.ForwardMismatch;

    // Train mode: loss = sum(pred * w) for a fixed random w, so d(loss)/d(pred) = w.
    var w: [n * 4]f32 = undefined;
    for (&w) |*v| v.* = r.float(f32) * 2 - 1;
    const Probe = struct {
        fn loss(m: *Net, c: []const u8, weights: []const f32) !f64 {
            const running = m.batches;
            try m.forward(c, true);
            m.batches = running;
            var p: [n * 4]f32 = undefined;
            try m.predictions(n, &p);
            var sum: f64 = 0;
            for (p, weights) |a, b| sum += @as(f64, a) * b;
            return sum;
        }
    };
    _ = try Probe.loss(&model, crops, &w);
    try model.backward(n, &w);
    var checked: usize = 0;
    var bad: usize = 0;
    const targets = [_]struct { *Param, []const u8 }{
        .{ &model.layers[0].weight, "conv0 weight" }, .{ &model.layers[5].weight, "conv5 weight" }, .{ &model.layers[9].weight, "conv9 weight" },
        .{ &model.layers[2].gamma, "bn2 gamma" },     .{ &model.layers[7].beta, "bn7 beta" },       .{ &model.fc1_w, "fc1 weight" },
        .{ &model.fc1_b, "fc1 bias" },                .{ &model.fc2_w, "fc2 weight" },              .{ &model.fc2_b, "fc2 bias" },
    };
    for (targets) |entry| {
        const p, const name = entry;
        const values = try gpa.alloc(f32, p.value.len);
        try dev.download(f32, values, p.value);
        const grads = try gpa.alloc(f32, p.grad.len);
        try dev.download(f32, grads, p.grad);
        for (0..4) |_| {
            const i = r.uintLessThan(usize, values.len);
            const original = values[i];
            const h: f32 = 2e-3 * @max(1, @abs(original)); // larger steps cross ReLU kinks
            values[i] = original + h;
            try dev.upload(f32, p.value, values);
            const up = try Probe.loss(&model, crops, &w);
            values[i] = original - h;
            try dev.upload(f32, p.value, values);
            const down = try Probe.loss(&model, crops, &w);
            values[i] = original;
            try dev.upload(f32, p.value, values);
            const numeric = (up - down) / (2 * h);
            const analytic: f64 = grads[i];
            const tolerance = 2e-2 * @max(@abs(numeric), @abs(analytic)) + 2e-3;
            checked += 1;
            if (@abs(numeric - analytic) > tolerance) {
                bad += 1;
                std.debug.print("  {s}[{d}]: analytic {e}, numeric {e}\n", .{ name, i, analytic, numeric });
            }
        }
    }
    std.debug.print("gradients vs finite differences: {d} of {d} within tolerance\n", .{ checked - bad, checked });
    if (bad > 0) return error.GradientMismatch;

    var g: [8]f32 = undefined;
    const pred = [_]f32{ 0.3, 0.4, 0.7, 0.45, 0.2, 0.5, 0.8, 0.5 };
    const target = [_]f32{ 0.31, 0.41, 0.69, 0.5, 0.2, 0.5, 0.8, 0.6 };
    _ = lossAndGrad(&pred, &target, 0.5, &g);
    var g_worst: f64 = 0;
    for (0..8) |i| {
        var up = pred;
        var down = pred;
        up[i] += 1e-3;
        down[i] -= 1e-3;
        var scratch: [8]f32 = undefined;
        const numeric = (lossAndGrad(&up, &target, 0.5, &scratch) - lossAndGrad(&down, &target, 0.5, &scratch)) / 2e-3;
        g_worst = @max(g_worst, @abs(numeric - g[i]));
    }
    std.debug.print("loss gradient vs finite differences: worst {e}\n", .{g_worst});
    if (g_worst > 5e-3) return error.LossGradientMismatch;
    std.debug.print("SELF-CHECK PASSED on {s}\n", .{dev.deviceName()});
}

// ── Main ──────────────────────────────────────────────────────────────────────

pub fn main(init: std.process.Init) !void {
    const gpa = init.arena.allocator();
    const io = init.io;
    const args = try init.minimal.args.toSlice(gpa);
    var positional: std.ArrayList([]const u8) = .empty;
    var epochs: usize = 300;
    var batch: usize = 64;
    var angle_weight: f64 = 0.5;
    var val_fraction: f64 = 0.15;
    var out_path: []const u8 = "eye_refiner.pt";
    var seed: u64 = 0;
    var check = false;
    var i: usize = 1;
    while (i < args.len) : (i += 1) {
        const arg = args[i];
        if (std.mem.eql(u8, arg, "--check")) {
            check = true;
        } else if (std.mem.startsWith(u8, arg, "--") and i + 1 < args.len) {
            i += 1;
            const v = args[i];
            if (std.mem.eql(u8, arg, "--epochs")) epochs = try std.fmt.parseInt(usize, v, 10) else if (std.mem.eql(u8, arg, "--batch")) batch = try std.fmt.parseInt(usize, v, 10) else if (std.mem.eql(u8, arg, "--angle-weight")) angle_weight = try std.fmt.parseFloat(f64, v) else if (std.mem.eql(u8, arg, "--val-fraction")) val_fraction = try std.fmt.parseFloat(f64, v) else if (std.mem.eql(u8, arg, "--out")) out_path = v else if (std.mem.eql(u8, arg, "--seed")) seed = try std.fmt.parseInt(u64, v, 10) else return error.UnknownOption;
        } else try positional.append(gpa, arg);
    }
    var dev = try cuda.Device.open(gpa, ptx);
    if (check) return selfCheck(gpa, io, &dev);
    if (positional.items.len != 1 or batch == 0 or batch > 256) {
        std.debug.print("usage: train_eye_refiner <data_dir> [--epochs 300] [--out eye_refiner.pt] [--batch 64] [--angle-weight 0.5] [--val-fraction 0.15] [--seed 0] | --check\n", .{});
        return error.Usage;
    }

    const data = try std.Io.Dir.cwd().realPathFileAlloc(io, positional.items[0], gpa);
    const images = try std.fs.path.join(gpa, &.{ data, "images" });
    // Selection happens on a slice of the training split. The test set is read once, at the
    // end: picking the best of thirty evaluations against it would report the luckiest epoch
    // rather than the model, and that figure is the one quoted against YuNet.
    const fit = try loadFaces(gpa, io, data, "train", val_fraction, false);
    const val = try loadFaces(gpa, io, data, "train", val_fraction, true);
    const test_faces = try loadFaces(gpa, io, data, "test", 0, false);
    var out: u.Stdout = undefined;
    out.init(io);
    const w = out.w();
    try w.print("{d} fitting faces, {d} validation faces (held out of training), {d} test faces\n", .{ fit.len, val.len, test_faces.len });
    try w.print("training on {s}\n", .{dev.deviceName()});
    try w.flush();
    const batches = fit.len / batch; // drop_last
    if (batches == 0) return error.TooFewFaces;

    var model = try Net.init(gpa, &dev, batch, seed);
    var parameters: usize = 0;
    for (model.paramList()) |p| parameters += p.value.len;
    try w.print("{s} M parameters\n", .{try u.fixedAlloc(gpa, @as(f64, @floatFromInt(parameters)) / 1e6, 2)});
    try w.flush();

    // Validation and test crops never change, so they are cut once.
    const Fixed = struct { crops: []u8, targets: [][4]f32 };
    var fixed: [2]Fixed = undefined;
    for ([_][]const Face{ val, test_faces }, &fixed) |faces, *f| {
        f.crops = try gpa.alloc(u8, faces.len * pixels);
        f.targets = try gpa.alloc([4]f32, faces.len);
        const order = try gpa.alloc(usize, faces.len);
        for (order, 0..) |*o, k| o.* = k;
        try cutAll(io, images, faces, order, 0, false, f.crops, f.targets);
    }

    const crops = try gpa.alloc(u8, batches * batch * pixels);
    const targets = try gpa.alloc([4]f32, batches * batch);
    const order = try gpa.alloc(usize, fit.len);
    for (order, 0..) |*o, k| o.* = k;
    var shuffler: std.Random.DefaultPrng = .init(seed +% 1);
    const total_steps = epochs * batches;
    var t: usize = 0;
    var pred: [256 * 4]f32 = undefined;
    var grad: [256 * 4]f32 = undefined;
    var best: ?struct { stats: Stats, epoch: usize, state: []const ck.OutEntry } = null;
    const started = std.Io.Timestamp.now(io, .awake);
    for (1..epochs + 1) |epoch| {
        shuffler.random().shuffle(usize, order);
        try cutAll(io, images, fit, order[0 .. batches * batch], seed *% 1000003 +% epoch, true, crops, targets);
        var total: f64 = 0;
        for (0..batches) |b| {
            const lr, const beta1 = oneCycle(t, total_steps, 3e-3);
            try model.forward(crops[b * batch * pixels ..][0 .. batch * pixels], true);
            try model.predictions(batch, &pred);
            const flat = std.mem.sliceAsBytes(targets[b * batch ..][0..batch]);
            total += lossAndGrad(pred[0 .. batch * 4], std.mem.bytesAsSlice(f32, @as([]align(4) const u8, @alignCast(flat))), angle_weight, grad[0 .. batch * 4]);
            try model.backward(@intCast(batch), grad[0 .. batch * 4]);
            t += 1;
            try model.step(lr, beta1, t);
        }
        if (epoch % 10 == 0 or epoch == epochs) {
            const stats = try evaluate(gpa, &model, fixed[0].crops, fixed[0].targets);
            var flag: []const u8 = "";
            if (best == null or stats.angle_median < best.?.stats.angle_median) {
                const state = try model.stateDict(gpa);
                best = .{ .stats = stats, .epoch = epoch, .state = state };
                try ck.save(gpa, io, out_path, &.{
                    .{ .key = "model", .value = .{ .dict = state } },
                    .{ .key = "size", .value = .{ .int = size } },
                    .{ .key = "val_stats", .value = try statsDict(gpa, stats) },
                    .{ .key = "epoch", .value = .{ .int = @intCast(epoch) } },
                    .{ .key = "trained_by", .value = .{ .string = trainer } },
                });
                flag = "  <- saved";
            }
            try w.print("epoch {d:>4}  loss {s}  val angle {s} deg  within5 {s}  dist {s} of box{s}  ({d:.0}s)\n", .{
                epoch,                                                                                                         try u.fixedAlloc(gpa, total / @as(f64, @floatFromInt(batches)), 4),
                try u.fixedAlloc(gpa, stats.angle_median, 2),                                                                  try percent(gpa, stats.within_5deg),
                try u.fixedAlloc(gpa, stats.distance_median, 4),                                                               flag,
                @as(f64, @floatFromInt(started.durationTo(std.Io.Timestamp.now(io, .awake)).nanoseconds)) / std.time.ns_per_s,
            });
            try w.flush();
        }
    }

    const chosen = best orelse {
        try w.writeAll("no evaluation ran, so there is nothing to report\n");
        return w.flush();
    };
    try w.print("\nselected on validation: {s} deg median at epoch {d}, {s} within 5 deg\n", .{ try u.fixedAlloc(gpa, chosen.stats.angle_median, 2), chosen.epoch, try percent(gpa, chosen.stats.within_5deg) });
    // The one and only look at the test set, using the checkpoint validation chose.
    try model.restore(chosen.state);
    const held = try evaluate(gpa, &model, fixed[1].crops, fixed[1].targets);
    try ck.save(gpa, io, out_path, &.{
        .{ .key = "model", .value = .{ .dict = chosen.state } },
        .{ .key = "size", .value = .{ .int = size } },
        .{ .key = "val_stats", .value = try statsDict(gpa, chosen.stats) },
        .{ .key = "epoch", .value = .{ .int = @intCast(chosen.epoch) } },
        .{ .key = "trained_by", .value = .{ .string = trainer } },
        .{ .key = "test_stats", .value = try statsDict(gpa, held) },
    });
    try w.print("test set, selected checkpoint: {s} deg median, {s} within 5 deg, {s} of box width\n", .{ try u.fixedAlloc(gpa, held.angle_median, 2), try percent(gpa, held.within_5deg), try u.fixedAlloc(gpa, held.distance_median, 4) });
    try w.writeAll("YuNet on the same test faces: 4.05 deg, 57.6% within 5 deg, 0.043 of box width\n");
    try w.flush();
}

test "OneCycle starts at max_lr/25, peaks at 30%, and ends near zero with momentum mirrored" {
    const first = oneCycle(0, 1000, 3e-3);
    try std.testing.expectApproxEqRel(@as(f64, 1.2e-4), first[0], 1e-12);
    try std.testing.expectApproxEqRel(@as(f64, 0.95), first[1], 1e-12);
    const peak = oneCycle(299, 1000, 3e-3);
    try std.testing.expectApproxEqRel(@as(f64, 3e-3), peak[0], 1e-12);
    try std.testing.expectApproxEqRel(@as(f64, 0.85), peak[1], 1e-12);
    const last = oneCycle(999, 1000, 3e-3);
    try std.testing.expectApproxEqRel(@as(f64, 1.2e-8), last[0], 1e-9);
    try std.testing.expectApproxEqRel(@as(f64, 0.95), last[1], 1e-12);
}

test "the loss gradient matches finite differences, angle term included" {
    const pred = [_]f32{ 0.3, 0.4, 0.7, 0.45, 0.2, 0.5, 0.8, 0.5 };
    const target = [_]f32{ 0.31, 0.41, 0.69, 0.5, 0.2, 0.5, 0.8, 0.6 };
    var g: [8]f32 = undefined;
    _ = lossAndGrad(&pred, &target, 0.5, &g);
    for (0..8) |i| {
        var up = pred;
        var down = pred;
        up[i] += 1e-3;
        down[i] -= 1e-3;
        var scratch: [8]f32 = undefined;
        const numeric = (lossAndGrad(&up, &target, 0.5, &scratch) - lossAndGrad(&down, &target, 0.5, &scratch)) / 2e-3;
        try std.testing.expectApproxEqAbs(numeric, g[i], 5e-3);
    }
}

test "the validation split reproduces Python's shuffle of sorted image names" {
    var r: u.PyRandom = .init(1234);
    var names = [_][]const u8{ "a", "b", "c", "d", "e" };
    r.shuffle([]const u8, &names);
    // CPython: n = list('abcde'); random.Random(1234).shuffle(n) gives ['b', 'c', 'e', 'a', 'd'].
    for ([_][]const u8{ "b", "c", "e", "a", "d" }, names) |want, got| try std.testing.expectEqualStrings(want, got);
}
