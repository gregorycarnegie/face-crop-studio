//! GPU kernels for training the eye refiner, compiled to the committed PTX beside this file:
//!
//!   zig build-obj -target nvptx64-cuda -mcpu=sm_80 -O ReleaseFast tools/dataset/gpu/refiner_kernels.zig \
//!       -fno-emit-bin -femit-asm=tools/dataset/gpu/refiner_kernels.ptx
//!
//! Tensors are float32 NCHW. Convolutions are 3x3, padding 1, no bias -- the refiner's only kind.
//! One thread per output element unless a kernel says otherwise; reductions over the batch use a
//! few hundred partial sums per channel and one atomic add each, so contention stays low.
//! Plain Zig, no libdevice: the only maths beyond + - * / is `@sqrt`, which lowers to sqrt.rn.

/// Partial sums per channel in the reduction kernels.
const lanes = 256;
const eps: f32 = 1e-5; // nn.BatchNorm2d's default

inline fn index() u32 {
    return @workGroupId(0) * @workGroupSize(0) + @workItemId(0);
}

inline fn add(p: *f32, v: f32) void {
    _ = @atomicRmw(f32, p, .Add, v, .monotonic);
}

export fn conv_fwd(n: u32, cin: u32, h: u32, w: u32, cout: u32, oh: u32, ow: u32, stride: u32, x: [*]const f32, wt: [*]const f32, y: [*]f32) callconv(.kernel) void {
    const i = index();
    if (i >= n * cout * oh * ow) return;
    const ox = i % ow;
    const oy = (i / ow) % oh;
    const co = (i / (ow * oh)) % cout;
    const b = i / (ow * oh * cout);
    var sum: f32 = 0;
    for (0..cin) |ci| {
        const plane = x + (b * cin + ci) * h * w;
        const k = wt + (co * cin + ci) * 9;
        for (0..3) |ky| {
            const iy = @as(i32, @intCast(oy * stride + ky)) - 1;
            if (iy < 0 or iy >= h) continue;
            for (0..3) |kx| {
                const ix = @as(i32, @intCast(ox * stride + kx)) - 1;
                if (ix < 0 or ix >= w) continue;
                sum += plane[@as(u32, @intCast(iy)) * w + @as(u32, @intCast(ix))] * k[ky * 3 + kx];
            }
        }
    }
    y[i] = sum;
}

/// One thread per input element: the gradient flowing back through every output it touched.
export fn conv_bwd_data(n: u32, cin: u32, h: u32, w: u32, cout: u32, oh: u32, ow: u32, stride: u32, dy: [*]const f32, wt: [*]const f32, dx: [*]f32) callconv(.kernel) void {
    const i = index();
    if (i >= n * cin * h * w) return;
    const ix = i % w;
    const iy = (i / w) % h;
    const ci = (i / (w * h)) % cin;
    const b = i / (w * h * cin);
    var sum: f32 = 0;
    for (0..cout) |co| {
        const plane = dy + (b * cout + co) * oh * ow;
        const k = wt + (co * cin + ci) * 9;
        for (0..3) |ky| {
            const ty = iy + 1;
            if (ty < ky or (ty - ky) % stride != 0) continue;
            const oy = (ty - ky) / stride;
            if (oy >= oh) continue;
            for (0..3) |kx| {
                const tx = ix + 1;
                if (tx < kx or (tx - kx) % stride != 0) continue;
                const ox = (tx - kx) / stride;
                if (ox >= ow) continue;
                sum += plane[oy * ow + ox] * k[ky * 3 + kx];
            }
        }
    }
    dx[i] = sum;
}

/// One thread per (weight, sample), accumulated atomically into a zeroed `dw`.
export fn conv_bwd_weight(n: u32, cin: u32, h: u32, w: u32, cout: u32, oh: u32, ow: u32, stride: u32, x: [*]const f32, dy: [*]const f32, dw: [*]f32) callconv(.kernel) void {
    const i = index();
    const weights = cout * cin * 9;
    if (i >= weights * n) return;
    const wi = i % weights;
    const b = i / weights;
    const kx = wi % 3;
    const ky = (wi / 3) % 3;
    const ci = (wi / 9) % cin;
    const co = wi / (9 * cin);
    const plane = x + (b * cin + ci) * h * w;
    const grad = dy + (b * cout + co) * oh * ow;
    var sum: f32 = 0;
    for (0..oh) |oy| {
        const iy = @as(i32, @intCast(oy * stride + ky)) - 1;
        if (iy < 0 or iy >= h) continue;
        for (0..ow) |ox| {
            const ix = @as(i32, @intCast(ox * stride + kx)) - 1;
            if (ix < 0 or ix >= w) continue;
            sum += plane[@as(u32, @intCast(iy)) * w + @as(u32, @intCast(ix))] * grad[oy * ow + ox];
        }
    }
    add(&dw[wi], sum);
}

/// `out[c] += sum over batch and space of x`, or of (x - mean[c])^2 when `mean` is given.
/// Launch with channels * lanes threads into a zeroed `out`.
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

/// Batch statistics from the sums, and nn.BatchNorm2d's running averages (momentum 0.1,
/// unbiased variance for the running estimate). One thread per channel.
export fn bn_stats(c: u32, m: f32, sqdev: [*]const f32, mean: [*]f32, variance: [*]f32, running_mean: [*]f32, running_var: [*]f32) callconv(.kernel) void {
    const ch = index();
    if (ch >= c) return;
    const v = sqdev[ch] / m;
    variance[ch] = v;
    running_mean[ch] = 0.9 * running_mean[ch] + 0.1 * mean[ch];
    running_var[ch] = 0.9 * running_var[ch] + 0.1 * v * m / (m - 1);
}

export fn scale(count: u32, factor: f32, x: [*]f32) callconv(.kernel) void {
    const i = index();
    if (i < count) x[i] *= factor;
}

/// BatchNorm with the given statistics, then ReLU.
export fn bn_relu(total: u32, c: u32, hw: u32, x: [*]const f32, mean: [*]const f32, variance: [*]const f32, gamma: [*]const f32, beta: [*]const f32, y: [*]f32) callconv(.kernel) void {
    const i = index();
    if (i >= total) return;
    const ch = (i / hw) % c;
    const v = (x[i] - mean[ch]) / @sqrt(variance[ch] + eps) * gamma[ch] + beta[ch];
    y[i] = @max(0, v);
}

/// dbeta[c] += sum dz, dgamma[c] += sum dz * xhat, where dz is dy through the ReLU.
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

/// The gradient into the convolution's output, from the batch-statistics BatchNorm backward.
export fn bn_bwd_dx(total: u32, c: u32, hw: u32, m: f32, x: [*]const f32, y: [*]const f32, dy: [*]const f32, mean: [*]const f32, variance: [*]const f32, gamma: [*]const f32, dgamma: [*]const f32, dbeta: [*]const f32, dx: [*]f32) callconv(.kernel) void {
    const i = index();
    if (i >= total) return;
    const ch = (i / hw) % c;
    const invstd = 1 / @sqrt(variance[ch] + eps);
    const dz = if (y[i] > 0) dy[i] else 0;
    const xhat = (x[i] - mean[ch]) * invstd;
    dx[i] = gamma[ch] * invstd / m * (m * dz - dbeta[ch] - xhat * dgamma[ch]);
}

export fn avgpool_fwd(nc: u32, hw: u32, x: [*]const f32, y: [*]f32) callconv(.kernel) void {
    const i = index();
    if (i >= nc) return;
    var sum: f32 = 0;
    for (0..hw) |p| sum += x[i * hw + p];
    y[i] = sum / @as(f32, @floatFromInt(hw));
}

export fn avgpool_bwd(total: u32, hw: u32, dy: [*]const f32, dx: [*]f32) callconv(.kernel) void {
    const i = index();
    if (i < total) dx[i] = dy[i / hw] / @as(f32, @floatFromInt(hw));
}

/// y = x W^T + b, optionally followed by ReLU into `act`.
export fn linear_fwd(n: u32, in: u32, out: u32, x: [*]const f32, wt: [*]const f32, bias: [*]const f32, y: [*]f32) callconv(.kernel) void {
    const i = index();
    if (i >= n * out) return;
    const o = i % out;
    const b = i / out;
    var sum = bias[o];
    for (0..in) |k| sum += x[b * in + k] * wt[o * in + k];
    y[i] = sum;
}

export fn relu_fwd(total: u32, x: [*]const f32, y: [*]f32) callconv(.kernel) void {
    const i = index();
    if (i < total) y[i] = @max(0, x[i]);
}

/// dx = dy through a ReLU whose output was `y`.
export fn relu_bwd(total: u32, y: [*]const f32, dy: [*]const f32, dx: [*]f32) callconv(.kernel) void {
    const i = index();
    if (i < total) dx[i] = if (y[i] > 0) dy[i] else 0;
}

export fn linear_bwd_data(n: u32, in: u32, out: u32, dy: [*]const f32, wt: [*]const f32, dx: [*]f32) callconv(.kernel) void {
    const i = index();
    if (i >= n * in) return;
    const k = i % in;
    const b = i / in;
    var sum: f32 = 0;
    for (0..out) |o| sum += dy[b * out + o] * wt[o * in + k];
    dx[i] = sum;
}

/// dW = dy^T x and db = column sums of dy; one thread per weight, plus `out` for the bias.
export fn linear_bwd_weight(n: u32, in: u32, out: u32, x: [*]const f32, dy: [*]const f32, dw: [*]f32, db: [*]f32) callconv(.kernel) void {
    const i = index();
    if (i < out * in) {
        const k = i % in;
        const o = i / in;
        var sum: f32 = 0;
        for (0..n) |b| sum += dy[b * out + o] * x[b * in + k];
        dw[i] = sum;
    } else if (i < out * in + out) {
        const o = i - out * in;
        var sum: f32 = 0;
        for (0..n) |b| sum += dy[b * out + o];
        db[o] = sum;
    }
}

/// torch.optim.AdamW (amsgrad off): decoupled decay, then the bias-corrected Adam step.
/// `bc1` = 1 - beta1^t and `bc2_sqrt` = sqrt(1 - beta2^t), computed on the host.
export fn adamw(count: u32, p: [*]f32, g: [*]const f32, m: [*]f32, v: [*]f32, lr: f32, beta1: f32, beta2: f32, weight_decay: f32, bc1: f32, bc2_sqrt: f32) callconv(.kernel) void {
    const i = index();
    if (i >= count) return;
    p[i] *= 1 - lr * weight_decay;
    m[i] += (g[i] - m[i]) * (1 - beta1);
    v[i] = v[i] * beta2 + (1 - beta2) * g[i] * g[i];
    const denom = @sqrt(v[i]) / bc2_sqrt + 1e-8;
    p[i] -= lr / bc1 * m[i] / denom;
}

/// u8 HWC crops to normalised float NCHW: (pixel - 127.5) / 128, as the refiner reads them.
export fn crops_to_input(n: u32, side: u32, crops: [*]const u8, x: [*]f32) callconv(.kernel) void {
    const i = index();
    const plane = side * side;
    if (i >= n * 3 * plane) return;
    const p = i % plane;
    const c = (i / plane) % 3;
    const b = i / (3 * plane);
    x[i] = (@as(f32, @floatFromInt(crops[(b * plane + p) * 3 + c])) - 127.5) / 128.0;
}
