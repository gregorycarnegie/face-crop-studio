//! GPU kernels for training the eye refiner, compiled to the committed PTX beside this file:
//!
//!   zig build-obj -target nvptx64-cuda -mcpu=sm_80 -O ReleaseFast tools/dataset/gpu/refiner_kernels.zig \
//!       -fno-emit-bin -femit-asm=tools/dataset/gpu/refiner_kernels.ptx
//!
//! Tensors are float32 NCHW. Convolutions are 3x3, padding 1, no bias -- the refiner's only kind.
//! Stride-1 convolutions tile 16x16 outputs with an 18x18 shared-memory halo; reductions use
//! one 256-thread block per channel/weight with shared-memory tree reduction (zero global atomics).
//! Plain Zig, no libdevice: the only maths beyond + - * / is `@sqrt`, which lowers to sqrt.rn.

const block_size = 256;
const tile_dim = 16;
const eps: f32 = 1e-5; // nn.BatchNorm2d's default

inline fn index() u32 {
    return @workGroupId(0) * @workGroupSize(0) + @workItemId(0);
}

inline fn barrier() void {
    asm volatile ("bar.sync 0;" ::: .{ .memory = true });
}

inline fn warpReduce(val: f32) f32 {
    var v = val;
    inline for (.{ 16, 8, 4, 2, 1 }) |offset| {
        v += asm ("shfl.sync.bfly.b32 %[r], %[x], %[o], 31, 0xffffffff;"
            : [r] "=f" (-> f32),
            : [x] "f" (v),
              [o] "n" (offset),
        );
    }
    return v;
}

/// Reduce `val` across a 256-thread block and return the sum uniformly to all threads.
inline fn blockReduce(smem: *addrspace(.shared) [block_size / 32]f32, tid: u32, val: f32) f32 {
    const wsum = warpReduce(val);
    if ((tid & 31) == 0) smem[tid >> 5] = wsum;
    barrier();
    var total: f32 = 0;
    inline for (0..block_size / 32) |w| total += smem[w];
    barrier();
    return total;
}

/// 3x3 stride-1 padding-1 convolution with 16x16 output tiles and an 18x18 shared input halo.
/// Launch with grid (n*cout, ceil(w/16), ceil(h/16)), block (16, 16, 1); n*cout goes in x, the
/// only grid dimension not capped at 65535 (batch 256 x 256 channels is 65536).
export fn conv_fwd_s1(n: u32, cin: u32, h: u32, w: u32, cout: u32, x: [*]const f32, wt: [*]const f32, y: [*]f32) callconv(.kernel) void {
    const S = struct {
        var tile: [tile_dim + 2][tile_dim + 2]f32 addrspace(.shared) = undefined;
    };
    const tx = @workItemId(0);
    const ty = @workItemId(1);
    const bx = @workGroupId(1) * tile_dim;
    const by = @workGroupId(2) * tile_dim;
    const ox = bx + tx;
    const oy = by + ty;
    const nc = @workGroupId(0);
    if (nc >= n * cout) return;
    const co = nc % cout;
    const b = nc / cout;
    const tid = ty * tile_dim + tx;
    var sum: f32 = 0;
    for (0..cin) |ci| {
        const plane = x + (b * cin + ci) * h * w;
        var idx = tid;
        while (idx < (tile_dim + 2) * (tile_dim + 2)) : (idx += tile_dim * tile_dim) {
            const ly = idx / (tile_dim + 2);
            const lx = idx % (tile_dim + 2);
            const iy = @as(i32, @intCast(by + ly)) - 1;
            const ix = @as(i32, @intCast(bx + lx)) - 1;
            S.tile[ly][lx] = if (iy >= 0 and iy < @as(i32, @intCast(h)) and ix >= 0 and ix < @as(i32, @intCast(w)))
                plane[@as(u32, @intCast(iy)) * w + @as(u32, @intCast(ix))]
            else
                0;
        }
        barrier();
        const k = wt + (co * cin + ci) * 9;
        inline for (0..3) |ky| {
            inline for (0..3) |kx| {
                sum += S.tile[ty + ky][tx + kx] * k[ky * 3 + kx];
            }
        }
        barrier();
    }
    if (ox < w and oy < h) {
        y[(nc * h + oy) * w + ox] = sum;
    }
}

/// 3x3 stride-2 padding-1 convolution, unrolled 3x3 inner loop.
export fn conv_fwd_s2(n: u32, cin: u32, h: u32, w: u32, cout: u32, oh: u32, ow: u32, x: [*]const f32, wt: [*]const f32, y: [*]f32) callconv(.kernel) void {
    const i = index();
    if (i >= n * cout * oh * ow) return;
    const ox = i % ow;
    const oy = (i / ow) % oh;
    const co = (i / (ow * oh)) % cout;
    const b = i / (ow * oh * cout);
    const ih: i32 = @intCast(h);
    const iw: i32 = @intCast(w);
    const base_y: i32 = @as(i32, @intCast(oy * 2)) - 1;
    const base_x: i32 = @as(i32, @intCast(ox * 2)) - 1;
    var sum: f32 = 0;
    for (0..cin) |ci| {
        const plane = x + (b * cin + ci) * h * w;
        const k = wt + (co * cin + ci) * 9;
        inline for (0..3) |ky| {
            const iy = base_y + @as(i32, ky);
            if (iy >= 0 and iy < ih) {
                const row = plane + @as(u32, @intCast(iy)) * w;
                inline for (0..3) |kx| {
                    const ix = base_x + @as(i32, kx);
                    if (ix >= 0 and ix < iw) {
                        sum += row[@as(u32, @intCast(ix))] * k[ky * 3 + kx];
                    }
                }
            }
        }
    }
    y[i] = sum;
}

/// 3x3 stride-1 padding-1 backward data with 16x16 output tiles and an 18x18 shared dy halo.
/// Launch with grid (n*cin, ceil(w/16), ceil(h/16)), block (16, 16, 1); see `conv_fwd_s1`.
export fn conv_bwd_data_s1(n: u32, cin: u32, h: u32, w: u32, cout: u32, dy: [*]const f32, wt: [*]const f32, dx: [*]f32) callconv(.kernel) void {
    const S = struct {
        var tile: [tile_dim + 2][tile_dim + 2]f32 addrspace(.shared) = undefined;
    };
    const tx = @workItemId(0);
    const ty = @workItemId(1);
    const bx = @workGroupId(1) * tile_dim;
    const by = @workGroupId(2) * tile_dim;
    const ix = bx + tx;
    const iy = by + ty;
    const nc = @workGroupId(0);
    if (nc >= n * cin) return;
    const ci = nc % cin;
    const b = nc / cin;
    const tid = ty * tile_dim + tx;
    var sum: f32 = 0;
    for (0..cout) |co| {
        const plane = dy + (b * cout + co) * h * w;
        var idx = tid;
        while (idx < (tile_dim + 2) * (tile_dim + 2)) : (idx += tile_dim * tile_dim) {
            const ly = idx / (tile_dim + 2);
            const lx = idx % (tile_dim + 2);
            const oy = @as(i32, @intCast(by + ly)) - 1;
            const ox = @as(i32, @intCast(bx + lx)) - 1;
            S.tile[ly][lx] = if (oy >= 0 and oy < @as(i32, @intCast(h)) and ox >= 0 and ox < @as(i32, @intCast(w)))
                plane[@as(u32, @intCast(oy)) * w + @as(u32, @intCast(ox))]
            else
                0;
        }
        barrier();
        const k = wt + (co * cin + ci) * 9;
        inline for (0..3) |ky| {
            inline for (0..3) |kx| {
                sum += S.tile[ty + 2 - ky][tx + 2 - kx] * k[ky * 3 + kx];
            }
        }
        barrier();
    }
    if (ix < w and iy < h) {
        dx[(nc * h + iy) * w + ix] = sum;
    }
}

/// 3x3 stride-2 padding-1 backward data, branch-free parity shifts (no runtime modulo/division).
export fn conv_bwd_data_s2(n: u32, cin: u32, h: u32, w: u32, cout: u32, oh: u32, ow: u32, dy: [*]const f32, wt: [*]const f32, dx: [*]f32) callconv(.kernel) void {
    const i = index();
    if (i >= n * cin * h * w) return;
    const ix = i % w;
    const iy = (i / w) % h;
    const ci = (i / (w * h)) % cin;
    const b = i / (w * h * cin);
    const ty = iy + 1;
    const tx = ix + 1;
    var sum: f32 = 0;
    for (0..cout) |co| {
        const plane = dy + (b * cout + co) * oh * ow;
        const k = wt + (co * cin + ci) * 9;
        inline for (0..3) |ky| {
            if (ty >= ky and ((ty - ky) & 1) == 0) {
                const oy = (ty - ky) >> 1;
                if (oy < oh) {
                    const row = plane + oy * ow;
                    inline for (0..3) |kx| {
                        if (tx >= kx and ((tx - kx) & 1) == 0) {
                            const ox = (tx - kx) >> 1;
                            if (ox < ow) {
                                sum += row[ox] * k[ky * 3 + kx];
                            }
                        }
                    }
                }
            }
        }
    }
    dx[i] = sum;
}

/// One 256-thread block per weight: coalesced spatial/batch reduction with zero global atomics.
/// Launch with grid (cout*cin*9, 1, 1), block (256, 1, 1).
export fn conv_bwd_weight(n: u32, cin: u32, h: u32, w: u32, cout: u32, oh: u32, ow: u32, stride: u32, x: [*]const f32, dy: [*]const f32, dw: [*]f32) callconv(.kernel) void {
    const S = struct {
        var smem: [block_size / 32]f32 addrspace(.shared) = undefined;
    };
    const wi = @workGroupId(0);
    if (wi >= cout * cin * 9) return;
    const tid = @workItemId(0);
    const kx = wi % 3;
    const ky = (wi / 3) % 3;
    const ci = (wi / 9) % cin;
    const co = wi / (9 * cin);
    const hw_in = h * w;
    const hw_out = oh * ow;
    const total = n * hw_out;
    const ih: i32 = @intCast(h);
    const iw: i32 = @intCast(w);
    var sum: f32 = 0;
    var e = tid;
    while (e < total) : (e += block_size) {
        const b = e / hw_out;
        const rem = e % hw_out;
        const oy = rem / ow;
        const ox = rem % ow;
        const iy = @as(i32, @intCast(oy * stride + ky)) - 1;
        const ix = @as(i32, @intCast(ox * stride + kx)) - 1;
        if (iy >= 0 and iy < ih and ix >= 0 and ix < iw) {
            const x_idx = (b * cin + ci) * hw_in + @as(u32, @intCast(iy)) * w + @as(u32, @intCast(ix));
            const dy_idx = (b * cout + co) * hw_out + rem;
            sum += x[x_idx] * dy[dy_idx];
        }
    }
    const reduced = blockReduce(&S.smem, tid, sum);
    if (tid == 0) dw[wi] = reduced;
}

/// Single-kernel BatchNorm training statistics: one 256-thread block per channel computes mean,
/// variance, running averages (momentum 0.1, unbiased running variance), and `scale = gamma/std`.
/// Launch with grid (c, 1, 1), block (256, 1, 1).
export fn bn_fwd_stats(n: u32, c: u32, hw: u32, x: [*]const f32, gamma: [*]const f32, mean: [*]f32, variance: [*]f32, running_mean: [*]f32, running_var: [*]f32, bn_scale: [*]f32) callconv(.kernel) void {
    const S = struct {
        var smem: [block_size / 32]f32 addrspace(.shared) = undefined;
    };
    const ch = @workGroupId(0);
    if (ch >= c) return;
    const tid = @workItemId(0);
    const total = n * hw;
    const m: f32 = @floatFromInt(total);

    var sum: f32 = 0;
    var e = tid;
    while (e < total) : (e += block_size) {
        sum += x[((e / hw) * c + ch) * hw + e % hw];
    }
    const total_sum = blockReduce(&S.smem, tid, sum);
    const mu = total_sum / m;
    if (tid == 0) mean[ch] = mu;

    var sq: f32 = 0;
    e = tid;
    while (e < total) : (e += block_size) {
        const d = x[((e / hw) * c + ch) * hw + e % hw] - mu;
        sq += d * d;
    }
    const total_sq = blockReduce(&S.smem, tid, sq);
    if (tid == 0) {
        const v = total_sq / m;
        variance[ch] = v;
        running_mean[ch] = 0.9 * running_mean[ch] + 0.1 * mu;
        running_var[ch] = 0.9 * running_var[ch] + 0.1 * v * m / (m - 1);
        bn_scale[ch] = gamma[ch] / @sqrt(v + eps);
    }
}

/// Precompute `gamma / sqrt(running_var + eps)` per channel for eval-mode BatchNorm.
export fn bn_eval_params(c: u32, running_var: [*]const f32, gamma: [*]const f32, bn_scale: [*]f32) callconv(.kernel) void {
    const ch = index();
    if (ch < c) bn_scale[ch] = gamma[ch] / @sqrt(running_var[ch] + eps);
}

/// BatchNorm with precomputed `bn_scale = gamma / sqrt(var + eps)`, then ReLU (no per-element sqrt).
export fn bn_relu(total: u32, c: u32, hw: u32, x: [*]const f32, mean: [*]const f32, bn_scale: [*]const f32, beta: [*]const f32, y: [*]f32) callconv(.kernel) void {
    const i = index();
    if (i >= total) return;
    const ch = (i / hw) % c;
    y[i] = @max(0, (x[i] - mean[ch]) * bn_scale[ch] + beta[ch]);
}

/// One 256-thread block per channel: reduces dbeta and dgamma without global atomics,
/// and caches `invstd = 1 / sqrt(var + eps)` for `bn_bwd_dx`.
/// Launch with grid (c, 1, 1), block (256, 1, 1).
export fn bn_bwd_reduce(n: u32, c: u32, hw: u32, x: [*]const f32, y: [*]const f32, dy: [*]const f32, mean: [*]const f32, variance: [*]const f32, dgamma: [*]f32, dbeta: [*]f32, invstd_out: [*]f32) callconv(.kernel) void {
    const S = struct {
        var smem_b: [block_size / 32]f32 addrspace(.shared) = undefined;
        var smem_g: [block_size / 32]f32 addrspace(.shared) = undefined;
    };
    const ch = @workGroupId(0);
    if (ch >= c) return;
    const tid = @workItemId(0);
    const invstd = 1 / @sqrt(variance[ch] + eps);
    const mu = mean[ch];
    const total = n * hw;
    var sb: f32 = 0;
    var sg: f32 = 0;
    var e = tid;
    while (e < total) : (e += block_size) {
        const at = ((e / hw) * c + ch) * hw + e % hw;
        const dz = if (y[at] > 0) dy[at] else 0;
        sb += dz;
        sg += dz * (x[at] - mu) * invstd;
    }
    const rb = blockReduce(&S.smem_b, tid, sb);
    const rg = blockReduce(&S.smem_g, tid, sg);
    if (tid == 0) {
        dbeta[ch] = rb;
        dgamma[ch] = rg;
        invstd_out[ch] = invstd;
    }
}

/// The gradient into the convolution's output, using cached `invstd` from `bn_bwd_reduce`.
export fn bn_bwd_dx(total: u32, c: u32, hw: u32, m: f32, x: [*]const f32, y: [*]const f32, dy: [*]const f32, mean: [*]const f32, invstd: [*]const f32, gamma: [*]const f32, dgamma: [*]const f32, dbeta: [*]const f32, dx: [*]f32) callconv(.kernel) void {
    const i = index();
    if (i >= total) return;
    const ch = (i / hw) % c;
    const istd = invstd[ch];
    const dz = if (y[i] > 0) dy[i] else 0;
    const xhat = (x[i] - mean[ch]) * istd;
    dx[i] = gamma[ch] * istd / m * (m * dz - dbeta[ch] - xhat * dgamma[ch]);
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

/// y = x W^T + b
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
