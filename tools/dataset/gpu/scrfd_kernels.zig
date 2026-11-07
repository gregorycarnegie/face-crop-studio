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
