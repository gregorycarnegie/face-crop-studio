//! SCRFD on the GPU for training: `scrfd.zig`'s step program (read from a checkpoint, so the
//! architecture and every weight name are the exporter's) run forward and backward with
//! training-mode BatchNorm, on `gpu/scrfd_kernels.zig`.
const std = @import("std");
const cuda = @import("cuda.zig");
const ck = @import("checkpoint.zig");
const net = @import("scrfd.zig");

pub const ptx = @embedFile("gpu/scrfd_kernels.ptx");
const block_size = 256; // gpu/scrfd_kernels.zig
const tile_dim = 16; // gpu/scrfd_kernels.zig
const Buf = cuda.Buffer(f32);

pub const Param = struct {
    name: []const u8, // checkpoint key
    value: Buf,
    grad: Buf,
    momentum: Buf,
    shape: []const usize,
};

const Shape = extern struct { n: u32, cin: u32, h: u32, w: u32, cout: u32, oh: u32, ow: u32, k: u32, stride: u32, pad: u32, groups: u32 };

const Layer = struct {
    conv: net.Conv,
    input: usize,
    weight: *Param,
    bias: ?*Param,
    /// `bn_scale`: gamma / sqrt(var + eps) in forward; `bn_bwd_reduce` overwrites it with invstd.
    bn: ?struct { gamma: *Param, beta: *Param, running_mean: Buf, running_var: Buf, mean: Buf, variance: Buf, bn_scale: Buf, name: []const u8 },
    z: Buf,
};

const Kernels = struct {
    conv1x1_fwd: cuda.Function,
    conv1x1_bwd_data: cuda.Function,
    conv3x3_dw_fwd: cuda.Function,
    conv3x3_dw_bwd_data: cuda.Function,
    conv_fwd: cuda.Function,
    conv_bwd_data: cuda.Function,
    conv_bwd_weight: cuda.Function,
    bias_bwd: cuda.Function,
    bn_fwd_stats: cuda.Function,
    bn_eval_params: cuda.Function,
    bn_relu: cuda.Function,
    bn_bwd_reduce: cuda.Function,
    bn_bwd_dx: cuda.Function,
    upsample_fwd: cuda.Function,
    upsample_bwd: cuda.Function,
    add_fwd: cuda.Function,
    add_into: cuda.Function,
    sgd: cuda.Function,
};

pub const Net = struct {
    dev: *const cuda.Device,
    k: Kernels,
    model: net.Scrfd,
    batch: usize,
    size: usize,
    params: std.ArrayList(*Param),
    /// Contiguous backing storage for all parameters and all per-value gradients:
    /// enables 2 memset calls in `zeroGrads` and 1 kernel launch in `sgd`.
    all_param_value: Buf,
    all_param_grad: Buf,
    all_param_momentum: Buf,
    all_val_grad: Buf,
    layers: []?Layer, // per step; null for add/upsample
    /// Per value: its output buffer and the gradient flowing into it.
    out: []Buf,
    grad: []Buf,
    scratch: Buf, // gradient into a BatchNorm'd convolution's raw output
    /// Training forward passes so far (BatchNorm's num_batches_tracked).
    batches: i64 = 0,

    fn sliceBuf(base: Buf, offset: usize, len: usize) Buf {
        return .{ .ptr = base.ptr + offset * @sizeOf(f32), .len = len };
    }

    pub fn init(gpa: std.mem.Allocator, dev: *const cuda.Device, model: net.Scrfd, batch: usize, size: usize) !Net {
        if (size % 32 != 0) return error.SizeNotMultipleOf32;
        var self: Net = .{
            .dev = dev,
            .k = undefined,
            .model = model,
            .batch = batch,
            .size = size,
            .params = .empty,
            .all_param_value = undefined,
            .all_param_grad = undefined,
            .all_param_momentum = undefined,
            .all_val_grad = undefined,
            .layers = try gpa.alloc(?Layer, model.steps.len),
            .out = try gpa.alloc(Buf, model.channels.len),
            .grad = try gpa.alloc(Buf, model.channels.len),
            .scratch = undefined,
        };
        inline for (comptime std.meta.fieldNames(Kernels)) |name| @field(self.k, name) = try dev.function(name);

        // Compute contiguous sizes for per-value gradients and all trainable parameters.
        var total_val_elems: usize = 0;
        var largest: usize = 0;
        for (model.channels, model.scale) |c, s| {
            const side = size / s;
            const elements = batch * c * side * side;
            largest = @max(largest, elements);
            total_val_elems += elements;
        }
        self.all_val_grad = try dev.alloc(f32, total_val_elems);
        var val_offset: usize = 0;
        for (model.channels, model.scale, 0..) |c, s, v| {
            const side = size / s;
            const elements = batch * c * side * side;
            self.out[v] = try dev.alloc(f32, elements);
            self.grad[v] = sliceBuf(self.all_val_grad, val_offset, elements);
            val_offset += elements;
        }
        self.scratch = try dev.alloc(f32, largest);

        var total_param_elems: usize = 0;
        for (model.steps) |step| {
            const s = switch (step) {
                .conv => |s| s,
                else => continue,
            };
            const c = s.conv;
            if ((c.bn != null) != c.relu) return error.UnsupportedLayer;
            total_param_elems += c.weight.len;
            if (c.bias) |b| total_param_elems += b.len;
            if (c.bn) |bn| total_param_elems += bn.gamma.len + bn.beta.len;
        }
        self.all_param_value = try dev.alloc(f32, total_param_elems);
        self.all_param_grad = try dev.alloc(f32, total_param_elems);
        self.all_param_momentum = try dev.alloc(f32, total_param_elems);
        try dev.zero(self.all_param_momentum);
        var param_offset: usize = 0;

        for (model.steps, self.layers, 1..) |step, *slot, id| {
            slot.* = null;
            const s = switch (step) {
                .conv => |s| s,
                else => continue,
            };
            const c = s.conv;
            const side = size / model.scale[id];
            const weight = try self.param(gpa, &param_offset, try std.fmt.allocPrint(gpa, "{s}.weight", .{c.name}), &.{ c.out, c.in_per_group, c.kernel, c.kernel }, c.weight);
            const bias = if (c.bias) |b| try self.param(gpa, &param_offset, try std.fmt.allocPrint(gpa, "{s}.bias", .{c.name}), &.{c.out}, b) else null;
            var layer: Layer = .{ .conv = c, .input = s.input, .weight = weight, .bias = bias, .bn = null, .z = if (c.bn != null) try dev.alloc(f32, batch * c.out * side * side) else self.out[id] };
            if (c.bn) |bn| {
                const bn_name = bnName(gpa, c.name) catch unreachable;
                layer.bn = .{
                    .gamma = try self.param(gpa, &param_offset, try std.fmt.allocPrint(gpa, "{s}.weight", .{bn_name}), &.{c.out}, bn.gamma),
                    .beta = try self.param(gpa, &param_offset, try std.fmt.allocPrint(gpa, "{s}.bias", .{bn_name}), &.{c.out}, bn.beta),
                    .running_mean = try dev.alloc(f32, c.out),
                    .running_var = try dev.alloc(f32, c.out),
                    .mean = try dev.alloc(f32, c.out),
                    .variance = try dev.alloc(f32, c.out),
                    .bn_scale = try dev.alloc(f32, c.out),
                    .name = bn_name,
                };
                try dev.upload(f32, layer.bn.?.running_mean, bn.mean);
                try dev.upload(f32, layer.bn.?.running_var, bn.variance);
            }
            slot.* = layer;
        }
        std.debug.assert(param_offset == total_param_elems);
        return self;
    }

    fn param(self: *Net, gpa: std.mem.Allocator, offset: *usize, name: []const u8, dims: []const usize, host: []const f32) !*Param {
        const p = try gpa.create(Param);
        const len = host.len;
        p.* = .{
            .name = name,
            .value = sliceBuf(self.all_param_value, offset.*, len),
            .grad = sliceBuf(self.all_param_grad, offset.*, len),
            .momentum = sliceBuf(self.all_param_momentum, offset.*, len),
            .shape = try gpa.dupe(usize, dims),
        };
        try self.dev.upload(f32, p.value, host);
        try self.params.append(gpa, p);
        offset.* += len;
        return p;
    }

    fn shape(self: *const Net, l: Layer, n: u32, id: usize) Shape {
        const in_side: u32 = @intCast(self.size / self.model.scale[l.input]);
        const out_side: u32 = @intCast(self.size / self.model.scale[id]);
        return .{ .n = n, .cin = @intCast(self.model.channels[l.input]), .h = in_side, .w = in_side, .cout = @intCast(l.conv.out), .oh = out_side, .ow = out_side, .k = @intCast(l.conv.kernel), .stride = @intCast(l.conv.stride), .pad = @intCast(l.conv.pad()), .groups = @intCast(l.conv.groups) };
    }

    fn count(self: *const Net, v: usize, n: usize) u32 {
        const side = self.size / self.model.scale[v];
        return @intCast(n * self.model.channels[v] * side * side);
    }

    /// Run `n` normalised CHW images (already uploaded to `out[0]`) through the network.
    pub fn forward(self: *Net, n: u32, train: bool) !void {
        const d = self.dev;
        for (self.model.steps, self.layers, 1..) |step, layer, id| switch (step) {
            .conv => {
                const l = layer.?;
                const s = self.shape(l, n, id);
                const hw = s.oh * s.ow;
                const total = s.n * s.cout * hw;
                const bias_ptr = if (l.bias) |b| b.value.ptr else @as(cuda.Ptr, 0);
                if (s.k == 1 and s.stride == 1 and s.pad == 0 and s.groups == 1) {
                    const gx: c_uint = (hw + tile_dim - 1) / tile_dim;
                    const gy: c_uint = (s.cout + tile_dim - 1) / tile_dim;
                    try d.launchGrid(self.k.conv1x1_fwd, .{ gx, gy, s.n }, .{ tile_dim, tile_dim, 1 }, .{ s.n, s.cin, s.cout, hw, self.out[l.input], l.weight.value, bias_ptr, l.z });
                } else if (s.k == 3 and s.pad == 1 and s.groups == s.cin and s.groups == s.cout) {
                    try d.launch(self.k.conv3x3_dw_fwd, total, .{ s.n, s.cin, s.h, s.w, s.oh, s.ow, s.stride, self.out[l.input], l.weight.value, bias_ptr, l.z });
                } else {
                    try d.launch(self.k.conv_fwd, total, .{ s, self.out[l.input], l.weight.value, bias_ptr, l.z });
                }
                if (l.bn) |bn| {
                    if (train) {
                        try d.launchGrid(self.k.bn_fwd_stats, .{ s.cout, 1, 1 }, .{ block_size, 1, 1 }, .{ n, s.cout, hw, l.z, bn.gamma.value, bn.mean, bn.variance, bn.running_mean, bn.running_var, bn.bn_scale });
                        try d.launch(self.k.bn_relu, total, .{ total, s.cout, hw, l.z, bn.mean, bn.bn_scale, bn.beta.value, self.out[id] });
                    } else {
                        try d.launch(self.k.bn_eval_params, s.cout, .{ s.cout, bn.running_var, bn.gamma.value, bn.bn_scale });
                        try d.launch(self.k.bn_relu, total, .{ total, s.cout, hw, l.z, bn.running_mean, bn.bn_scale, bn.beta.value, self.out[id] });
                    }
                }
            },
            .add => |pair| {
                const total = self.count(id, n);
                try d.launch(self.k.add_fwd, total, .{ total, self.out[pair[0]], self.out[pair[1]], self.out[id] });
            },
            .upsample => |from| {
                const total = self.count(id, n);
                const side: u32 = @intCast(self.size / self.model.scale[id]);
                try d.launch(self.k.upsample_fwd, total, .{ total, side, side, self.out[from], self.out[id] });
            },
        };
        if (train) self.batches += 1;
    }

    /// Zero every gradient: parameters and per-value buffers. Then the loss writes the head
    /// outputs' gradients into `grad[...]` and `backward` takes it from there.
    pub fn zeroGrads(self: *Net) !void {
        try self.dev.zero(self.all_param_grad);
        try self.dev.zero(self.all_val_grad);
    }

    pub fn backward(self: *Net, n: u32) !void {
        const d = self.dev;
        var i = self.model.steps.len;
        while (i > 0) {
            i -= 1;
            const id = i + 1;
            switch (self.model.steps[i]) {
                .conv => {
                    const l = self.layers[i].?;
                    const s = self.shape(l, n, id);
                    const hw = s.oh * s.ow;
                    const total = s.n * s.cout * hw;
                    var g = self.grad[id];
                    if (l.bn) |bn| {
                        try d.launchGrid(self.k.bn_bwd_reduce, .{ s.cout, 1, 1 }, .{ block_size, 1, 1 }, .{ n, s.cout, hw, l.z, self.out[id], self.grad[id], bn.mean, bn.variance, bn.gamma.grad, bn.beta.grad, bn.bn_scale });
                        try d.launch(self.k.bn_bwd_dx, total, .{ total, s.cout, hw, @as(f32, @floatFromInt(n * hw)), l.z, self.out[id], self.grad[id], bn.mean, bn.bn_scale, bn.gamma.value, bn.gamma.grad, bn.beta.grad, self.scratch });
                        g = self.scratch;
                    }
                    if (l.bias) |b| try d.launchGrid(self.k.bias_bwd, .{ s.cout, 1, 1 }, .{ block_size, 1, 1 }, .{ n, s.cout, hw, g, b.grad });
                    const rows: u32 = @max(1, 2048 / s.ow);
                    const bands = (s.oh + rows - 1) / rows;
                    const weights = s.cout * (s.cin / s.groups) * s.k * s.k;
                    try d.launchGrid(self.k.conv_bwd_weight, .{ weights, bands, 1 }, .{ block_size, 1, 1 }, .{ s, rows, bands, self.out[l.input], g, l.weight.grad });
                    if (l.input != 0) {
                        if (s.k == 1 and s.stride == 1 and s.pad == 0 and s.groups == 1) {
                            const gx: c_uint = (hw + tile_dim - 1) / tile_dim;
                            const gy: c_uint = (s.cin + tile_dim - 1) / tile_dim;
                            try d.launchGrid(self.k.conv1x1_bwd_data, .{ gx, gy, s.n }, .{ tile_dim, tile_dim, 1 }, .{ s.n, s.cin, s.cout, hw, g, l.weight.value, self.grad[l.input] });
                        } else if (s.k == 3 and s.pad == 1 and s.groups == s.cin and s.groups == s.cout) {
                            try d.launch(self.k.conv3x3_dw_bwd_data, s.n * s.cin * s.h * s.w, .{ s.n, s.cin, s.h, s.w, s.oh, s.ow, s.stride, g, l.weight.value, self.grad[l.input] });
                        } else {
                            try d.launch(self.k.conv_bwd_data, s.n * s.cin * s.h * s.w, .{ s, g, l.weight.value, self.grad[l.input] });
                        }
                    }
                },
                .add => |pair| {
                    const total = self.count(id, n);
                    for (pair) |v| try d.launch(self.k.add_into, total, .{ total, self.grad[id], self.grad[v] });
                },
                .upsample => |from| {
                    const total = self.count(from, n);
                    const side: u32 = @intCast(self.size / self.model.scale[from]);
                    try d.launch(self.k.upsample_bwd, total, .{ total, side, side, self.grad[id], self.grad[from] });
                },
            }
        }
    }

    /// One SGD step over every parameter (`first` seeds the momentum buffers).
    pub fn sgd(self: *Net, lr: f64, momentum: f64, weight_decay: f64, first: bool) !void {
        const c: u32 = @intCast(self.all_param_value.len);
        try self.dev.launch(self.k.sgd, c, .{ c, self.all_param_value, self.all_param_grad, self.all_param_momentum, @as(f32, @floatCast(lr)), @as(f32, @floatCast(momentum)), @as(f32, @floatCast(weight_decay)), @as(u32, @intFromBool(first)) });
    }

    /// The state_dict in checkpoint order: each convolution, then its BatchNorm.
    pub fn stateDict(self: *Net, gpa: std.mem.Allocator) ![]const ck.OutEntry {
        var entries: std.ArrayList(ck.OutEntry) = .empty;
        const counter = try gpa.alloc(i64, 1);
        counter[0] = self.batches;
        for (self.layers) |slot| {
            const l = slot orelse continue;
            try entries.append(gpa, try self.entry(gpa, l.weight.name, l.weight.value, l.weight.shape));
            if (l.bias) |b| try entries.append(gpa, try self.entry(gpa, b.name, b.value, b.shape));
            if (l.bn) |bn| {
                const c = bn.gamma.shape;
                try entries.append(gpa, try self.entry(gpa, bn.gamma.name, bn.gamma.value, c));
                try entries.append(gpa, try self.entry(gpa, bn.beta.name, bn.beta.value, c));
                try entries.append(gpa, try self.entry(gpa, try std.fmt.allocPrint(gpa, "{s}.running_mean", .{bn.name}), bn.running_mean, c));
                try entries.append(gpa, try self.entry(gpa, try std.fmt.allocPrint(gpa, "{s}.running_var", .{bn.name}), bn.running_var, c));
                try entries.append(gpa, .{ .key = try std.fmt.allocPrint(gpa, "{s}.num_batches_tracked", .{bn.name}), .value = .{ .tensor = .{ .dtype = .int64, .shape = &.{}, .data = std.mem.sliceAsBytes(counter) } } });
            }
        }
        return entries.toOwnedSlice(gpa);
    }

    fn entry(self: *Net, gpa: std.mem.Allocator, key: []const u8, buf: Buf, shp: []const usize) !ck.OutEntry {
        const host = try gpa.alloc(f32, buf.len);
        try self.dev.download(f32, host, buf);
        return .{ .key = key, .value = .{ .tensor = .{ .dtype = .float32, .shape = shp, .data = std.mem.sliceAsBytes(host) } } };
    }
};

/// One positive anchor's targets; the layout of `gpu/scrfd_kernels.zig`'s `Positive`.
pub const Positive = extern struct {
    image: u32,
    level: u32,
    cell: u32,
    anchor: u32,
    box: [4]f32,
    kps: [10]f32,
    kps_weight: f32,
    _pad: u32 = 0,
};
comptime {
    std.debug.assert(@sizeOf(Positive) == 80);
}

pub const Losses = struct { cls: f64, box: f64, kps: f64 };

const slots = 1024; // gpu/scrfd_kernels.zig
const targets = @import("scrfd_targets.zig");
const level_offsets = [3]u32{ 0, targets.level_counts[0], targets.level_counts[0] + targets.level_counts[1] };
const anchors_per_image = targets.anchor_count;

/// The loss's device-side inputs, grown as batches need.
pub const LossBuffers = struct {
    label_weight: cuda.Buffer(u8),
    positives: cuda.Buffer(Positive),
    stats: Buf,
    kernels: struct { qfl_negative: cuda.Function, positives_cls: cuda.Function, positives_box_kps: cuda.Function },

    pub fn init(dev: *const cuda.Device, batch: usize) !LossBuffers {
        return .{
            .label_weight = try dev.alloc(u8, batch * anchors_per_image),
            .positives = try dev.alloc(Positive, batch * 1024),
            .stats = try dev.alloc(f32, slots + 4),
            .kernels = .{ .qfl_negative = try dev.function("qfl_negative"), .positives_cls = try dev.function("positives_cls"), .positives_box_kps = try dev.function("positives_box_kps") },
        };
    }
};

/// SCRFDHead.loss for the last training forward pass: writes the head outputs' gradients
/// (call `zeroGrads` first, `backward` after) and returns the three loss terms.
/// `label_weight` is per image and anchor; `samples` is the sum over images of max(positives, 1).
pub fn loss(self: *Net, lb: *LossBuffers, n: u32, label_weight: []const u8, positives: []const Positive, samples: f32) !Losses {
    if (self.size != 640) return error.LossNeeds640;
    const d = self.dev;
    if (positives.len > lb.positives.len) {
        d.free(lb.positives);
        lb.positives = try d.alloc(Positive, positives.len * 2);
    }
    try d.upload(u8, lb.label_weight, label_weight);
    try d.upload(Positive, lb.positives, positives);
    try d.zero(lb.stats);
    const inv = 1 / samples;
    const count: u32 = @intCast(positives.len);
    const k = lb.kernels;
    for (self.model.heads, 0..) |head, level| {
        const side: u32 = @intCast(self.size / self.model.scale[head[0]]);
        try d.launch(k.qfl_negative, n * 2 * side * side, .{ n, side, side, level_offsets[level], @as(u32, anchors_per_image), self.out[head[0]], self.grad[head[0]], lb.label_weight, inv, lb.stats });
    }
    for (self.model.heads, 0..) |head, level| {
        const side: u32 = @intCast(self.size / self.model.scale[head[0]]);
        const stride: f32 = @floatFromInt(self.model.scale[head[0]]);
        try d.launch(k.positives_cls, count, .{ count, lb.positives, @as(u32, @intCast(level)), side, side, stride, self.out[head[0]], self.out[head[1]], self.grad[head[0]], inv, lb.stats });
    }
    for (self.model.heads, 0..) |head, level| {
        const side: u32 = @intCast(self.size / self.model.scale[head[0]]);
        const stride: f32 = @floatFromInt(self.model.scale[head[0]]);
        try d.launch(k.positives_box_kps, count, .{ count, lb.positives, @as(u32, @intCast(level)), side, side, stride, self.out[head[0]], self.out[head[1]], self.out[head[2]], self.grad[head[1]], self.grad[head[2]], lb.stats });
    }
    var stats: [slots + 4]f32 = undefined;
    try d.download(f32, &stats, lb.stats);
    var cls: f64 = stats[slots];
    for (stats[0..slots]) |s| cls += s;
    return .{ .cls = cls, .box = stats[slots + 1], .kps = stats[slots + 2] };
}

/// The BatchNorm beside a convolution, by mmdet's naming: `X.0` -> `X.1`, `X.3` -> `X.4`
/// (MobileNetV1 blocks), `X.conv` -> `X.bn` (ConvModule).
pub fn bnName(gpa: std.mem.Allocator, conv: []const u8) ![]const u8 {
    if (std.mem.endsWith(u8, conv, ".conv")) return std.fmt.allocPrint(gpa, "{s}.bn", .{conv[0 .. conv.len - ".conv".len]});
    const dot = std.mem.findScalarLast(u8, conv, '.') orelse return error.UnknownBatchNorm;
    const index = try std.fmt.parseInt(usize, conv[dot + 1 ..], 10);
    return std.fmt.allocPrint(gpa, "{s}.{d}", .{ conv[0..dot], index + 1 });
}

test "BatchNorm names follow mmdet's layout" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const gpa = arena.allocator();
    try std.testing.expectEqualStrings("backbone.stem.0.1", try bnName(gpa, "backbone.stem.0.0"));
    try std.testing.expectEqualStrings("backbone.layer2.1.4", try bnName(gpa, "backbone.layer2.1.3"));
    try std.testing.expectEqualStrings("bbox_head.cls_stride_convs.(8, 8).0.depthwise_conv.bn", try bnName(gpa, "bbox_head.cls_stride_convs.(8, 8).0.depthwise_conv.conv"));
}
