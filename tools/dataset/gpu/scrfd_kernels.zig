//! GPU kernels for training SCRFD, compiled to the committed PTX beside this file:
//!
//!   zig build-obj -target nvptx64-cuda -mcpu=sm_80 -O ReleaseFast tools/dataset/gpu/scrfd_kernels.zig \
//!       -fno-emit-bin -femit-asm=tools/dataset/gpu/scrfd_kernels.ptx
//!
//! Float32 NCHW. Convolutions are general: any odd kernel, stride, padding and group count, with
//! an optional bias -- SCRFD has full, depthwise and pointwise convolutions, some with bias.
//! Backward kernels *accumulate* into their input gradients (a feature can feed several layers);
//! the trainer zeroes gradients each step. Reductions use `lanes` partial sums per channel and
//! one atomic add each. No libdevice: exp goes through PTX's ex2.approx.

const lanes = 256;
const eps: f32 = 1e-5; // nn.BatchNorm2d's default

inline fn index() u32 {
    return @workGroupId(0) * @workGroupSize(0) + @workItemId(0);
}

inline fn add(p: *f32, v: f32) void {
    _ = @atomicRmw(f32, p, .Add, v, .monotonic);
}

fn exp(v: f32) f32 {
    return asm ("ex2.approx.ftz.f32 %[r], %[x];"
        : [r] "=f" (-> f32),
        : [x] "f" (v * 1.4426950408889634),
    );
}

fn log(v: f32) f32 {
    return asm ("lg2.approx.ftz.f32 %[r], %[x];"
        : [r] "=f" (-> f32),
        : [x] "f" (v),
    ) * 0.6931471805599453;
}

const Conv = extern struct { n: u32, cin: u32, h: u32, w: u32, cout: u32, oh: u32, ow: u32, k: u32, stride: u32, pad: u32, groups: u32 };

export fn conv_fwd(c: Conv, x: [*]const f32, wt: [*]const f32, bias: ?[*]const f32, y: [*]f32) callconv(.kernel) void {
    const i = index();
    if (i >= c.n * c.cout * c.oh * c.ow) return;
    const ox = i % c.ow;
    const oy = (i / c.ow) % c.oh;
    const co = (i / (c.ow * c.oh)) % c.cout;
    const b = i / (c.ow * c.oh * c.cout);
    const in_g = c.cin / c.groups;
    const g = co / (c.cout / c.groups);
    var sum: f32 = if (bias) |bs| bs[co] else 0;
    for (0..in_g) |ci| {
        const plane = x + (b * c.cin + g * in_g + ci) * c.h * c.w;
        const kern = wt + (co * in_g + ci) * c.k * c.k;
        for (0..c.k) |ky| {
            const iy = @as(i32, @intCast(oy * c.stride + ky)) - @as(i32, @intCast(c.pad));
            if (iy < 0 or iy >= c.h) continue;
            for (0..c.k) |kx| {
                const ix = @as(i32, @intCast(ox * c.stride + kx)) - @as(i32, @intCast(c.pad));
                if (ix < 0 or ix >= c.w) continue;
                sum += plane[@as(u32, @intCast(iy)) * c.w + @as(u32, @intCast(ix))] * kern[ky * c.k + kx];
            }
        }
    }
    y[i] = sum;
}

/// dx += the gradient through every output each input element touched.
export fn conv_bwd_data(c: Conv, dy: [*]const f32, wt: [*]const f32, dx: [*]f32) callconv(.kernel) void {
    const i = index();
    if (i >= c.n * c.cin * c.h * c.w) return;
    const ix = i % c.w;
    const iy = (i / c.w) % c.h;
    const ci = (i / (c.w * c.h)) % c.cin;
    const b = i / (c.w * c.h * c.cin);
    const in_g = c.cin / c.groups;
    const out_g = c.cout / c.groups;
    const g = ci / in_g;
    const cl = ci % in_g;
    var sum: f32 = 0;
    for (0..out_g) |ol| {
        const co = g * out_g + ol;
        const plane = dy + (b * c.cout + co) * c.oh * c.ow;
        const kern = wt + (co * in_g + cl) * c.k * c.k;
        for (0..c.k) |ky| {
            const ty = iy + c.pad;
            if (ty < ky or (ty - ky) % c.stride != 0) continue;
            const oy = (ty - ky) / c.stride;
            if (oy >= c.oh) continue;
            for (0..c.k) |kx| {
                const tx = ix + c.pad;
                if (tx < kx or (tx - kx) % c.stride != 0) continue;
                const ox = (tx - kx) / c.stride;
                if (ox >= c.ow) continue;
                sum += plane[oy * c.ow + ox] * kern[ky * c.k + kx];
            }
        }
    }
    dx[i] += sum;
}

/// One thread per (weight, sample, band of `rows` output rows), added atomically into `dw`.
export fn conv_bwd_weight(c: Conv, rows: u32, bands: u32, x: [*]const f32, dy: [*]const f32, dw: [*]f32) callconv(.kernel) void {
    const i = index();
    const in_g = c.cin / c.groups;
    const weights = c.cout * in_g * c.k * c.k;
    if (i >= weights * c.n * bands) return;
    const wi = i % weights;
    const b = (i / weights) % c.n;
    const band = i / (weights * c.n);
    const kx = wi % c.k;
    const ky = (wi / c.k) % c.k;
    const cl = (wi / (c.k * c.k)) % in_g;
    const co = wi / (c.k * c.k * in_g);
    const ci = (co / (c.cout / c.groups)) * in_g + cl;
    const plane = x + (b * c.cin + ci) * c.h * c.w;
    const grad = dy + (b * c.cout + co) * c.oh * c.ow;
    var sum: f32 = 0;
    const first = band * rows;
    const last = @min(c.oh, first + rows);
    var oy = first;
    while (oy < last) : (oy += 1) {
        const iy = @as(i32, @intCast(oy * c.stride + ky)) - @as(i32, @intCast(c.pad));
        if (iy < 0 or iy >= c.h) continue;
        for (0..c.ow) |ox| {
            const ix = @as(i32, @intCast(ox * c.stride + kx)) - @as(i32, @intCast(c.pad));
            if (ix < 0 or ix >= c.w) continue;
            sum += plane[@as(u32, @intCast(iy)) * c.w + @as(u32, @intCast(ix))] * grad[oy * c.ow + ox];
        }
    }
    add(&dw[wi], sum);
}

/// `out[c] += sum of x` over batch and space (also a bias gradient), or of (x - mean[c])^2.
export fn channel_sum(n: u32, c: u32, hw: u32, x: [*]const f32, mean: ?[*]const f32, out: [*]f32) callconv(.kernel) void {
    const i = index();
    if (i >= c * lanes) return;
    const ch = i / lanes;
    var e = i % lanes;
    var sum: f32 = 0;
    const centre = if (mean) |m| m[ch] else 0;
    while (e < n * hw) : (e += lanes) {
        const v = x[((e / hw) * c + ch) * hw + e % hw];
        sum += if (mean != null) (v - centre) * (v - centre) else v;
    }
    add(&out[ch], sum);
}

export fn scale(count: u32, factor: f32, x: [*]f32) callconv(.kernel) void {
    const i = index();
    if (i < count) x[i] *= factor;
}

/// Batch variance from the squared deviations, and the running averages (momentum 0.1,
/// unbiased running variance).
export fn bn_stats(c: u32, m: f32, sqdev: [*]const f32, mean: [*]const f32, variance: [*]f32, running_mean: [*]f32, running_var: [*]f32) callconv(.kernel) void {
    const ch = index();
    if (ch >= c) return;
    const v = sqdev[ch] / m;
    variance[ch] = v;
    running_mean[ch] = 0.9 * running_mean[ch] + 0.1 * mean[ch];
    running_var[ch] = 0.9 * running_var[ch] + 0.1 * v * m / (m - 1);
}

export fn bn_relu(total: u32, c: u32, hw: u32, x: [*]const f32, mean: [*]const f32, variance: [*]const f32, gamma: [*]const f32, beta: [*]const f32, y: [*]f32) callconv(.kernel) void {
    const i = index();
    if (i >= total) return;
    const ch = (i / hw) % c;
    y[i] = @max(0, (x[i] - mean[ch]) / @sqrt(variance[ch] + eps) * gamma[ch] + beta[ch]);
}

export fn bn_bwd_reduce(n: u32, c: u32, hw: u32, x: [*]const f32, y: [*]const f32, dy: [*]const f32, mean: [*]const f32, variance: [*]const f32, dgamma: [*]f32, dbeta: [*]f32) callconv(.kernel) void {
    const i = index();
    if (i >= c * lanes) return;
    const ch = i / lanes;
    const invstd = 1 / @sqrt(variance[ch] + eps);
    var e = i % lanes;
    var sb: f32 = 0;
    var sg: f32 = 0;
    while (e < n * hw) : (e += lanes) {
        const at = ((e / hw) * c + ch) * hw + e % hw;
        const dz = if (y[at] > 0) dy[at] else 0;
        sb += dz;
        sg += dz * (x[at] - mean[ch]) * invstd;
    }
    add(&dbeta[ch], sb);
    add(&dgamma[ch], sg);
}

export fn bn_bwd_dx(total: u32, c: u32, hw: u32, m: f32, x: [*]const f32, y: [*]const f32, dy: [*]const f32, mean: [*]const f32, variance: [*]const f32, gamma: [*]const f32, dgamma: [*]const f32, dbeta: [*]const f32, dx: [*]f32) callconv(.kernel) void {
    const i = index();
    if (i >= total) return;
    const ch = (i / hw) % c;
    const invstd = 1 / @sqrt(variance[ch] + eps);
    const dz = if (y[i] > 0) dy[i] else 0;
    const xhat = (x[i] - mean[ch]) * invstd;
    dx[i] = gamma[ch] * invstd / m * (m * dz - dbeta[ch] - xhat * dgamma[ch]);
}

/// Nearest-neighbour 2x: (oh, ow) is the output size.
export fn upsample_fwd(total: u32, oh: u32, ow: u32, x: [*]const f32, y: [*]f32) callconv(.kernel) void {
    const i = index();
    if (i >= total) return;
    const ox = i % ow;
    const oy = (i / ow) % oh;
    const plane = i / (ow * oh);
    y[i] = x[(plane * (oh / 2) + oy / 2) * (ow / 2) + ox / 2];
}

/// dx += the four output gradients each input pixel was copied to; (h, w) is the input size.
export fn upsample_bwd(total: u32, h: u32, w: u32, dy: [*]const f32, dx: [*]f32) callconv(.kernel) void {
    const i = index();
    if (i >= total) return;
    const ix = i % w;
    const iy = (i / w) % h;
    const plane = i / (w * h);
    const g = dy + plane * 4 * h * w;
    const ow = 2 * w;
    dx[i] += g[(2 * iy) * ow + 2 * ix] + g[(2 * iy) * ow + 2 * ix + 1] + g[(2 * iy + 1) * ow + 2 * ix] + g[(2 * iy + 1) * ow + 2 * ix + 1];
}

export fn add_fwd(count: u32, a: [*]const f32, b: [*]const f32, y: [*]f32) callconv(.kernel) void {
    const i = index();
    if (i < count) y[i] = a[i] + b[i];
}

export fn add_into(count: u32, a: [*]const f32, y: [*]f32) callconv(.kernel) void {
    const i = index();
    if (i < count) y[i] += a[i];
}

/// torch.optim.SGD with momentum and (coupled) weight decay, dampening 0: the first step seeds
/// the momentum buffer with the gradient itself.
export fn sgd(count: u32, p: [*]f32, g: [*]const f32, buf: [*]f32, lr: f32, momentum: f32, weight_decay: f32, first: u32) callconv(.kernel) void {
    const i = index();
    if (i >= count) return;
    const d = g[i] + weight_decay * p[i];
    buf[i] = if (first != 0) d else momentum * buf[i] + d;
    p[i] -= lr * buf[i];
}

// ── Losses (SCRFDHead.loss, use_kps, no DFL) ──────────────────────────────────

/// One positive anchor's targets, mirrored by `scrfd_net.Positive`.
pub const Positive = extern struct {
    image: u32,
    level: u32,
    cell: u32, // y * W + x within its level
    anchor: u32, // 0 or 1 at that location
    box: [4]f32, // target box, pixels
    kps: [10]f32, // target keypoints, pixels
    kps_weight: f32,
    _pad: u32 = 0,
};

/// Loss accumulators: `slots` partial sums of the negative classification loss, then the
/// positive classification, box and keypoint sums and the sum of positive scores.
pub const slots = 1024;
pub const stat_pos_cls = slots;
pub const stat_box = slots + 1;
pub const stat_kps = slots + 2;
pub const stat_weight = slots + 3;

fn sigmoid(x: f32) f32 {
    return 1 / (1 + exp(-x));
}

/// binary_cross_entropy_with_logits(x, t), the stable form.
fn bce(x: f32, t: f32) f32 {
    return @max(x, 0) - x * t + log(1 + exp(-@abs(x)));
}

/// Quality Focal Loss for every anchor of one level as a negative (target score 0), weighted
/// by its label weight (0 for ignored anchors) over `samples`; overwrites the class gradient.
/// `cls` is the level's (n, 2, H, W) logit map; `label_weight` is per image and global anchor.
export fn qfl_negative(n: u32, h: u32, w: u32, level_offset: u32, anchors: u32, cls: [*]const f32, grad: [*]f32, label_weight: [*]const u8, inv_samples: f32, stats: [*]f32) callconv(.kernel) void {
    const i = index();
    if (i >= n * 2 * h * w) return;
    const x = i % w;
    const y = (i / w) % h;
    const a = (i / (w * h)) % 2;
    const b = i / (w * h * 2);
    const lw: f32 = @floatFromInt(label_weight[b * anchors + level_offset + (y * w + x) * 2 + a]);
    const logit = cls[i];
    const s = sigmoid(logit);
    const sp = log(1 + exp(-@abs(logit))) + @max(logit, 0); // softplus = BCE(x, 0)
    grad[i] = lw * inv_samples * (s * s * s + 2 * s * s * (1 - s) * sp);
    add(&stats[i % slots], lw * inv_samples * sp * s * s);
}

const Decoded = struct { cls_at: u32, map: u32, centre: [2]f32, pred: [4]f32, target: [4]f32 };

fn decode(p: Positive, h: u32, w: u32, stride: f32, reg: [*]const f32) Decoded {
    const plane = h * w;
    const cx: f32 = @floatFromInt(p.cell % w);
    const cy: f32 = @floatFromInt(p.cell / w);
    const reg_at = (p.image * 8 + p.anchor * 4) * plane + p.cell;
    const d = [4]f32{ reg[reg_at], reg[reg_at + plane], reg[reg_at + 2 * plane], reg[reg_at + 3 * plane] };
    return .{
        .cls_at = (p.image * 2 + p.anchor) * plane + p.cell,
        .map = reg_at,
        .centre = .{ cx, cy },
        .pred = .{ cx - d[0], cy - d[1], cx + d[2], cy + d[3] },
        .target = .{ p.box[0] / stride, p.box[1] / stride, p.box[2] / stride, p.box[3] / stride },
    };
}

/// bbox_overlaps(is_aligned, eps 1e-6).
fn iou(a: [4]f32, b: [4]f32) f32 {
    const ww = @max(@min(a[2], b[2]) - @max(a[0], b[0]), 0);
    const hh = @max(@min(a[3], b[3]) - @max(a[1], b[1]), 0);
    const inter = ww * hh;
    const un = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter;
    return inter / @max(un, 1e-6);
}

/// Pass 1 over one level's positives: Quality Focal Loss with the IoU of the (detached)
/// decoded box as target, overwriting the class gradient, and each positive's score (detached
/// sigmoid) into the normaliser the box and keypoint losses divide by.
export fn positives_cls(count: u32, positives: [*]const Positive, level: u32, h: u32, w: u32, stride: f32, cls: [*]const f32, reg: [*]const f32, cls_grad: [*]f32, inv_samples: f32, stats: [*]f32) callconv(.kernel) void {
    const i = index();
    if (i >= count or positives[i].level != level) return;
    const dcd = decode(positives[i], h, w, stride, reg);
    const t = iou(dcd.pred, dcd.target);
    const logit = cls[dcd.cls_at];
    const s = sigmoid(logit);
    const loss = bce(logit, t);
    const gap = t - s;
    cls_grad[dcd.cls_at] = inv_samples * ((s - t) * gap * gap - 2 * loss * gap * s * (1 - s));
    // The negative pass already counted this anchor as a negative; QFL replaces that term.
    const negative = (log(1 + exp(-@abs(logit))) + @max(logit, 0)) * s * s;
    add(&stats[stat_pos_cls], inv_samples * (loss * gap * gap - negative));
    add(&stats[stat_weight], s);
}

/// DIoU loss (eps 1e-6) and its gradient with respect to the predicted box corners.
fn diou(p: [4]f32, g: [4]f32, grad: *[4]f32) f32 {
    const e: f32 = 1e-6;
    const ow = @min(p[2], g[2]) - @max(p[0], g[0]);
    const oh = @min(p[3], g[3]) - @max(p[1], g[1]);
    const iw = @max(ow, 0);
    const ih = @max(oh, 0);
    const overlap = iw * ih;
    const pw = p[2] - p[0];
    const ph = p[3] - p[1];
    const un = pw * ph + (g[2] - g[0]) * (g[3] - g[1]) - overlap + e;
    const cw = @max(@max(p[2], g[2]) - @min(p[0], g[0]), 0);
    const ch = @max(@max(p[3], g[3]) - @min(p[1], g[1]), 0);
    const c2 = cw * cw + ch * ch + e;
    const dx = (g[0] + g[2]) - (p[0] + p[2]);
    const dy = (g[1] + g[3]) - (p[1] + p[3]);
    const rho2 = dx * dx / 4 + dy * dy / 4;
    // Overlap and enclosure edges move with whichever side binds the max/min.
    const d_ow = [4]f32{ if (ow > 0 and p[0] > g[0]) -1 else 0, 0, if (ow > 0 and p[2] < g[2]) 1 else 0, 0 };
    const d_oh = [4]f32{ 0, if (oh > 0 and p[1] > g[1]) -1 else 0, 0, if (oh > 0 and p[3] < g[3]) 1 else 0 };
    const d_ap = [4]f32{ -ph, -pw, ph, pw };
    const d_cw = [4]f32{ if (p[0] < g[0]) -1 else 0, 0, if (p[2] > g[2]) 1 else 0, 0 };
    const d_ch = [4]f32{ 0, if (p[1] < g[1]) -1 else 0, 0, if (p[3] > g[3]) 1 else 0 };
    const d_rho = [4]f32{ -dx / 2, -dy / 2, -dx / 2, -dy / 2 };
    for (0..4) |k| {
        const d_overlap = d_ow[k] * ih + iw * d_oh[k];
        const d_union = d_ap[k] - d_overlap;
        const d_iou = (d_overlap * un - overlap * d_union) / (un * un);
        const d_c2 = 2 * cw * d_cw[k] + 2 * ch * d_ch[k];
        grad[k] = -d_iou + (d_rho[k] * c2 - rho2 * d_c2) / (c2 * c2);
    }
    return 1 - (overlap / un - rho2 / c2);
}

fn smoothL1(diff: f32, beta: f32, grad: *f32) f32 {
    const a = @abs(diff);
    if (a < beta) {
        grad.* = diff / beta;
        return 0.5 * a * a / beta;
    }
    grad.* = if (diff > 0) 1 else -1;
    return a - 0.5 * beta;
}

/// Pass 2: DIoU (weight 2) and keypoint smooth-L1 (beta 1/9, weight 0.1) losses, each weighted
/// by the positive's score and divided by the summed scores from pass 1; writes the box and
/// keypoint gradients.
export fn positives_box_kps(count: u32, positives: [*]const Positive, level: u32, h: u32, w: u32, stride: f32, cls: [*]const f32, reg: [*]const f32, kps: [*]const f32, reg_grad: [*]f32, kps_grad: [*]f32, stats: [*]f32) callconv(.kernel) void {
    const i = index();
    if (i >= count or positives[i].level != level) return;
    const p = positives[i];
    const avg = stats[stat_weight];
    if (!(avg > 0)) return;
    const dcd = decode(p, h, w, stride, reg);
    const s = sigmoid(cls[dcd.cls_at]);
    var g: [4]f32 = undefined;
    const box_loss = diou(dcd.pred, dcd.target, &g);
    const k_box = 2 * s / avg;
    const plane = h * w;
    // pred corners: x1 = cx - d0, y1 = cy - d1, x2 = cx + d2, y2 = cy + d3.
    reg_grad[dcd.map] = -g[0] * k_box;
    reg_grad[dcd.map + plane] = -g[1] * k_box;
    reg_grad[dcd.map + 2 * plane] = g[2] * k_box;
    reg_grad[dcd.map + 3 * plane] = g[3] * k_box;
    add(&stats[stat_box], box_loss * k_box);
    const kps_at = (p.image * 20 + p.anchor * 10) * plane + p.cell;
    const k_kps = 0.1 * p.kps_weight * s / avg;
    var kps_loss: f32 = 0;
    for (0..10) |j| {
        const target = p.kps[j] / stride - dcd.centre[j % 2];
        var gj: f32 = undefined;
        kps_loss += smoothL1(kps[kps_at + j * plane] - target, 1.0 / 9.0, &gj);
        kps_grad[kps_at + j * plane] = gj * k_kps;
    }
    add(&stats[stat_kps], kps_loss * k_kps);
}
