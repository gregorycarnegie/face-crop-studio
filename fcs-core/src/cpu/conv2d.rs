//! CPU convolution for the detector's graph.
//!
//! Semantics match `gpu/conv2d.wgsl` exactly, because the two backends are
//! compared against each other: cross-correlation (not flipped), zero padding,
//! grouped channels with `group_in = in_channels / groups`, weights laid out as
//! `[oc][local_ic][ky][kx]`, one bias per output channel, optional fused
//! activation.
//!
//! Three paths, because the graph only ever asks for three shapes and they want
//! very different code:
//!
//! * **pointwise** (1x1, dense) — a matrix multiply over channels, and where
//!   most of the model's arithmetic lives.
//! * **depthwise** (KxK, `groups == channels`) — one independent kernel per
//!   channel. This is the case `tract` lowers to a scalar loop, which is
//!   59-61% of a CPU detection there.
//! * **general** — everything else. The detector uses it once, for the 3x3 stride-2
//!   stem, so it is written for clarity rather than speed.

use anyhow::Result;
use std::ops::Range;
use rayon::prelude::*;

use super::tensor::Tensor;

/// Activation fused into the convolution's epilogue.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Activation {
    /// Leave the convolution output unchanged.
    None,
    /// Clamp negative output values to zero.
    Relu,
}

impl Activation {
    #[inline]
    fn apply(self, v: f32) -> f32 {
        match self {
            Self::None => v,
            // `max` rather than a branch: it is one instruction and vectorises.
            Self::Relu => v.max(0.0),
        }
    }
}

/// Everything a convolution needs beyond its tensors.
#[derive(Debug, Clone, Copy)]
pub struct ConvConfig {
    /// Kernel step in input pixels along both spatial axes; must be nonzero.
    pub stride: usize,
    /// Number of zero-padding pixels on each side of each spatial axis.
    pub padding: usize,
    /// 1 for a dense convolution, `channels` for depthwise.
    pub groups: usize,
    /// Activation applied after adding the bias.
    pub activation: Activation,
}

impl ConvConfig {
    /// A 1x1 dense convolution: stride 1, no padding.
    pub fn pointwise(activation: Activation) -> Self {
        Self {
            stride: 1,
            padding: 0,
            groups: 1,
            activation,
        }
    }

    /// A KxK depthwise convolution with `padding` chosen to preserve size.
    pub fn depthwise(groups: usize, padding: usize, activation: Activation) -> Self {
        Self {
            stride: 1,
            padding,
            groups,
            activation,
        }
    }
}

/// Convolution weights, kept as plain data so they can be shared across threads.
#[derive(Debug, Clone)]
pub struct ConvWeights {
    /// Number of output channels.
    pub out_channels: usize,
    /// Number of input channels consumed by each convolution group.
    pub in_channels_per_group: usize,
    /// Kernel height in pixels.
    pub kernel_h: usize,
    /// Kernel width in pixels.
    pub kernel_w: usize,
    /// Flattened weights in `[output_channel, input_channel_in_group, y, x]` order.
    pub data: Vec<f32>,
    /// One additive bias per output channel.
    pub bias: Vec<f32>,
}

impl ConvWeights {
    /// Wrap convolution weights and biases, checking their lengths against the dimensions.
    ///
    /// Returns an error unless `data` has `out_channels * in_channels_per_group *
    /// kernel_h * kernel_w` elements and `bias` has `out_channels` elements.
    pub fn new(
        out_channels: usize,
        in_channels_per_group: usize,
        kernel_h: usize,
        kernel_w: usize,
        data: Vec<f32>,
        bias: Vec<f32>,
    ) -> Result<Self> {
        let expected = out_channels * in_channels_per_group * kernel_h * kernel_w;
        anyhow::ensure!(
            data.len() == expected,
            "conv weights {out_channels}x{in_channels_per_group}x{kernel_h}x{kernel_w} need \
             {expected} elements, got {}",
            data.len()
        );
        anyhow::ensure!(
            bias.len() == out_channels,
            "conv bias needs {out_channels} elements, got {}",
            bias.len()
        );
        Ok(Self {
            out_channels,
            in_channels_per_group,
            kernel_h,
            kernel_w,
            data,
            bias,
        })
    }

    fn weights_per_output(&self) -> usize {
        self.in_channels_per_group * self.kernel_h * self.kernel_w
    }
}

/// How many output rows one parallel task should cover.
///
/// Chunking purely by output channel starves the early layers: the stem has 16
/// output channels, so on a 32-thread machine half the cores idle through the
/// most expensive layer in the network. Splitting rows as well gives every
/// thread work. The result always divides `out_h`, so a chunk never straddles
/// two channel planes — which would silently mix channels.
///
/// `threads` is a parameter rather than read here so the arithmetic can be tested; callers pass
/// `rayon::current_num_threads()`.
///
/// Takes `batch` and `channels` rather than their product: every caller computed the plane
/// count itself, and no output could tell a wrong one from a right one -- any value yields
/// a divisor of `out_h`, so it only moves work between tasks. Inside, the multiplication is
/// covered by this function's own test.
fn rows_per_task(out_h: usize, batch: usize, channels: usize, threads: usize) -> usize {
    let planes = batch * channels;
    // Several tasks per thread so rayon can balance a ragged tail.
    let target_tasks = threads.max(1) * 4;
    let wanted_per_plane = (target_tasks / planes.max(1)).max(1);
    let rows = out_h.div_ceil(wanted_per_plane);
    // The next divisor of out_h at or above that, so a chunk never straddles two channel planes.
    (rows.max(1)..=out_h)
        .find(|r| out_h.is_multiple_of(*r))
        .unwrap_or(1)
}

/// Output columns whose whole kernel window lies inside the image, so the general path can
/// read them without a bounds check per tap.
///
/// Column `ox` reads input columns from `ox * stride - pad` for `kw` columns; it is interior
/// when that span starts at or after 0 and ends at or before `w`. When the kernel is wider
/// than the padded image no column qualifies and the range is empty. This used
/// `saturating_sub`, which turned "no column" into "column 0": the general path then skipped
/// the bounds checks for it and read past the end of the input plane.
fn interior_columns(w: usize, kw: usize, stride: usize, pad: usize, out_w: usize) -> Range<usize> {
    let hi = (w + pad)
        .checked_sub(kw)
        .map_or(0, |span| span / stride + 1)
        .min(out_w);
    let lo = pad.div_ceil(stride).min(hi);
    lo..hi
}

/// Output spatial size for a given input, kernel, stride and padding.
fn output_dim(input: usize, kernel: usize, stride: usize, padding: usize) -> usize {
    (input + 2 * padding).saturating_sub(kernel) / stride + 1
}

/// Run a convolution, dispatching to whichever path fits the shape.
pub fn conv2d(input: &Tensor, weights: &ConvWeights, config: &ConvConfig) -> Result<Tensor> {
    anyhow::ensure!(config.groups > 0, "groups must be non-zero");
    anyhow::ensure!(config.stride > 0, "stride must be non-zero");
    anyhow::ensure!(
        input.channels().is_multiple_of(config.groups),
        "input channels ({}) not divisible by groups ({})",
        input.channels(),
        config.groups
    );
    anyhow::ensure!(
        weights.out_channels.is_multiple_of(config.groups),
        "output channels ({}) not divisible by groups ({})",
        weights.out_channels,
        config.groups
    );
    anyhow::ensure!(
        input.channels() / config.groups == weights.in_channels_per_group,
        "weights expect {} input channels per group, tensor has {}",
        weights.in_channels_per_group,
        input.channels() / config.groups
    );

    let depthwise = config.groups == input.channels() && config.groups == weights.out_channels;
    let pointwise = weights.kernel_h == 1
        && weights.kernel_w == 1
        && config.groups == 1
        && config.stride == 1
        && config.padding == 0;

    if pointwise {
        conv_pointwise(input, weights, config)
    } else if depthwise {
        conv_depthwise(input, weights, config)
    } else {
        conv_general(input, weights, config)
    }
}

/// 1x1 dense convolution.
///
/// Every output pixel is a dot product over input channels at the same spatial
/// position, so this is a `[out_c, in_c] x [in_c, HW]` matrix multiply. The
/// spatial axis runs innermost, which makes the inner statement a scaled add of
/// two contiguous slices — the shape that vectorises.
///
/// This reads the whole input once per output channel — 105 MB for a single
/// 64->64 layer at 80x80 — and cutting that traffic has now been attempted
/// twice, both times measuring *slower*. Do not attempt a third time without a
/// new idea:
///
/// * Blocking four output channels per pass: 18.5 ms against 17.4 ms. It also
///   fought the row-splitting in `rows_per_task` for the same tasks, and with
///   at most 64 channels against 32 threads the row split is worth more.
/// * Regrouping `chunks_mut` by row-band so one task writes every channel plane
///   and a band of input rows is read once for all of them: 18.2 ms against
///   15.4 ms. Safe, no `unsafe` needed — but building the `Vec<Vec<&mut [f32]>>`
///   per call, the indirection through it in the inner loop, and scattering the
///   writes across `c_out` bands each cost more than the reads saved.
///
/// The plain form below wins because its inner statement is a scaled add of two
/// contiguous slices, which vectorises and which the prefetcher handles; the
/// arithmetic is memory-bound in theory and latency-hidden in practice.
fn conv_pointwise(input: &Tensor, weights: &ConvWeights, config: &ConvConfig) -> Result<Tensor> {
    let (n, c_in, h, w) = (
        input.batch(),
        input.channels(),
        input.height(),
        input.width(),
    );
    let c_out = weights.out_channels;
    let plane = h * w;
    let mut output = Tensor::zeros(n, c_out, h, w);

    let src = input.data();
    let act = config.activation;

    // Tasks are (plane, row-block) rather than whole planes: a 16-channel layer
    // would otherwise leave most threads idle, and those are the layers running
    // at the largest spatial sizes.
    let rpt = rows_per_task(h, n, c_out, rayon::current_num_threads());
    let rows_per_plane = h / rpt;

    output
        .data_mut()
        .par_chunks_mut(rpt * w)
        .enumerate()
        .for_each(|(task, out_rows)| {
            let plane_idx = task / rows_per_plane;
            let oy0 = (task % rows_per_plane) * rpt;
            let batch = plane_idx / c_out;
            let oc = plane_idx % c_out;
            let in_base = batch * c_in * plane;
            let span = out_rows.len();
            let offset = oy0 * w;

            out_rows.fill(weights.bias[oc]);

            for ic in 0..c_in {
                let scale = weights.data[oc * c_in + ic];
                if scale == 0.0 {
                    continue;
                }
                let base = in_base + ic * plane + offset;
                let in_rows = &src[base..base + span];
                for (o, i) in out_rows.iter_mut().zip(in_rows) {
                    *o += scale * i;
                }
            }

            if act != Activation::None {
                for v in out_rows.iter_mut() {
                    *v = act.apply(*v);
                }
            }
        });

    Ok(output)
}

/// Depthwise convolution: one kernel per channel, stride 1.
///
/// Accumulates a whole output row at a time rather than gathering nine taps per
/// pixel. For each kernel tap the contribution to the row is
/// `out[x] += k * in[x + shift]` over a contiguous span, which is a plain
/// scaled add of two slices — the shape LLVM vectorises. The alternative, a
/// nine-tap gather per output pixel, reloads overlapping inputs and does not.
///
/// The per-tap span also removes bounds checking entirely: instead of testing
/// each pixel, the valid range of `x` is computed once from the shift.
fn conv_depthwise(input: &Tensor, weights: &ConvWeights, config: &ConvConfig) -> Result<Tensor> {
    if config.stride != 1 {
        return conv_general(input, weights, config);
    }
    let (n, c, h, w) = (
        input.batch(),
        input.channels(),
        input.height(),
        input.width(),
    );
    let (kh, kw) = (weights.kernel_h, weights.kernel_w);
    let pad = config.padding;
    let out_h = output_dim(h, kh, 1, pad);
    let out_w = output_dim(w, kw, 1, pad);
    let mut output = Tensor::zeros(n, c, out_h, out_w);

    let src = input.data();
    let act = config.activation;
    let kernel_area = kh * kw;

    let rpt = rows_per_task(out_h, n, c, rayon::current_num_threads());
    let rows_per_plane = out_h / rpt;

    output
        .data_mut()
        .par_chunks_mut(rpt * out_w)
        .enumerate()
        .for_each(|(task, out_rows)| {
            let plane_idx = task / rows_per_plane;
            let oy0 = (task % rows_per_plane) * rpt;
            let batch = plane_idx / c;
            let ch = plane_idx % c;
            let in_plane = &src[(batch * c + ch) * h * w..(batch * c + ch + 1) * h * w];
            let kernel = &weights.data[ch * kernel_area..(ch + 1) * kernel_area];
            let bias = weights.bias[ch];

            for (r, out_row) in out_rows.chunks_mut(out_w).enumerate() {
                let oy = oy0 + r;
                out_row.fill(bias);

                for ky in 0..kh {
                    let iy = oy as isize + ky as isize - pad as isize;
                    if iy < 0 || iy >= h as isize {
                        continue;
                    }
                    let in_row = &in_plane[iy as usize * w..(iy as usize + 1) * w];

                    for kx in 0..kw {
                        let k = kernel[ky * kw + kx];
                        if k == 0.0 {
                            continue;
                        }
                        // out[x] reads in[x + shift]; clamp x to where both sides exist.
                        let shift = kx as isize - pad as isize;
                        let lo = (-shift).max(0) as usize;
                        let hi = ((w as isize - shift).min(out_w as isize)).max(0) as usize;
                        if lo >= hi {
                            continue;
                        }
                        let dst = &mut out_row[lo..hi];
                        let srcs =
                            &in_row[(lo as isize + shift) as usize..(hi as isize + shift) as usize];
                        for (o, i) in dst.iter_mut().zip(srcs) {
                            *o += k * i;
                        }
                    }
                }

                if act != Activation::None {
                    for v in out_row.iter_mut() {
                        *v = act.apply(*v);
                    }
                }
            }
        });

    Ok(output)
}

/// Everything the specialised paths do not cover.
///
/// The detector uses this once, for the 3x3 stride-2 stem over RGB — which measured as
/// the single most expensive layer in the network, since it runs at the full
/// 640x640 input. So it is not the naive loop it looks like: output rows are
/// classified as interior (every kernel tap lands inside the image) or edge,
/// and the interior case drops all bounds checking, which is the whole cost at
/// a one-pixel border.
fn conv_general(input: &Tensor, weights: &ConvWeights, config: &ConvConfig) -> Result<Tensor> {
    let (n, c_in, h, w) = (
        input.batch(),
        input.channels(),
        input.height(),
        input.width(),
    );
    let c_out = weights.out_channels;
    let (kh, kw) = (weights.kernel_h, weights.kernel_w);
    let (stride, pad) = (config.stride, config.padding);
    let out_h = output_dim(h, kh, stride, pad);
    let out_w = output_dim(w, kw, stride, pad);

    let group_out = c_out / config.groups;
    let group_in = c_in / config.groups;
    let weights_per_out = weights.weights_per_output();
    let plane_in = h * w;

    let mut output = Tensor::zeros(n, c_out, out_h, out_w);
    let src = input.data();
    let act = config.activation;

    let interior_cols = interior_columns(w, kw, stride, pad, out_w);

    let rpt = rows_per_task(out_h, n, c_out, rayon::current_num_threads());
    let rows_per_plane = out_h / rpt;

    output
        .data_mut()
        .par_chunks_mut(rpt * out_w)
        .enumerate()
        .for_each(|(task, out_rows)| {
            let plane_idx = task / rows_per_plane;
            let oy0 = (task % rows_per_plane) * rpt;
            let batch = plane_idx / c_out;
            let oc = plane_idx % c_out;
            let group = oc / group_out;
            let in_base = batch * c_in * plane_in + group * group_in * plane_in;
            let w_base = oc * weights_per_out;
            let bias = weights.bias[oc];

            for (r, out_row) in out_rows.chunks_mut(out_w).enumerate() {
                let oy = oy0 + r;
                let iy0 = (oy * stride) as isize - pad as isize;
                let rows_inside = iy0 >= 0 && iy0 + kh as isize <= h as isize;

                for (ox, out) in out_row.iter_mut().enumerate() {
                    let ix0 = (ox * stride) as isize - pad as isize;
                    let interior = rows_inside && interior_cols.contains(&ox);
                    let mut acc = bias;

                    if interior {
                        for ic in 0..group_in {
                            let plane =
                                &src[in_base + ic * plane_in..in_base + (ic + 1) * plane_in];
                            let kernel =
                                &weights.data[w_base + ic * kh * kw..w_base + (ic + 1) * kh * kw];
                            for ky in 0..kh {
                                let row = (iy0 + ky as isize) as usize * w + ix0 as usize;
                                let irow = &plane[row..row + kw];
                                let krow = &kernel[ky * kw..(ky + 1) * kw];
                                for (k, v) in krow.iter().zip(irow) {
                                    acc += k * v;
                                }
                            }
                        }
                    } else {
                        for ic in 0..group_in {
                            let plane =
                                &src[in_base + ic * plane_in..in_base + (ic + 1) * plane_in];
                            let kernel =
                                &weights.data[w_base + ic * kh * kw..w_base + (ic + 1) * kh * kw];
                            for ky in 0..kh {
                                let iy = iy0 + ky as isize;
                                if iy < 0 || iy >= h as isize {
                                    continue;
                                }
                                let row = iy as usize * w;
                                for kx in 0..kw {
                                    let ix = ix0 + kx as isize;
                                    if ix < 0 || ix >= w as isize {
                                        continue;
                                    }
                                    acc += kernel[ky * kw + kx] * plane[row + ix as usize];
                                }
                            }
                        }
                    }
                    *out = act.apply(acc);
                }
            }
        });

    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A convolution written straight from the definition, with no special
    /// cases at all. The optimised paths above are compared against this rather
    /// than against hand-computed values: the interior/border split and the
    /// pointwise channel-major loop are exactly the kind of rewrite that stays
    /// correct in the middle of an image and goes wrong at an edge, and a
    /// handful of golden pixels would not catch that.
    fn reference(input: &Tensor, weights: &ConvWeights, config: &ConvConfig) -> Tensor {
        // `dims()` reads the fields directly: going through `batch()` and friends would let a
        // broken accessor corrupt the reference and the code under test identically.
        let [n, c_in, h, w] = input.dims();
        let c_out = weights.out_channels;
        let (kh, kw) = (weights.kernel_h, weights.kernel_w);
        let (stride, pad) = (config.stride, config.padding);
        let out_h = output_dim(h, kh, stride, pad);
        let out_w = output_dim(w, kw, stride, pad);
        let group_out = c_out / config.groups;
        let group_in = c_in / config.groups;

        let mut out = Tensor::zeros(n, c_out, out_h, out_w);
        for b in 0..n {
            for oc in 0..c_out {
                let group = oc / group_out;
                for oy in 0..out_h {
                    for ox in 0..out_w {
                        let mut acc = weights.bias[oc];
                        for lic in 0..group_in {
                            let ic = group * group_in + lic;
                            for ky in 0..kh {
                                for kx in 0..kw {
                                    let iy = (oy * stride) as isize + ky as isize - pad as isize;
                                    let ix = (ox * stride) as isize + kx as isize - pad as isize;
                                    if iy < 0 || iy >= h as isize || ix < 0 || ix >= w as isize {
                                        continue;
                                    }
                                    let iv = input.data()
                                        [((b * c_in + ic) * h + iy as usize) * w + ix as usize];
                                    let wv =
                                        weights.data[((oc * group_in + lic) * kh + ky) * kw + kx];
                                    acc += wv * iv;
                                }
                            }
                        }
                        let v = config.activation.apply(acc);
                        out.data_mut()[((b * c_out + oc) * out_h + oy) * out_w + ox] = v;
                    }
                }
            }
        }
        out
    }

    /// Deterministic pseudo-random values, spread either side of zero so ReLU
    /// actually clamps some of them.
    fn ramp(len: usize, seed: usize) -> Vec<f32> {
        (0..len)
            .map(|i| (((i * 37 + seed * 17) % 23) as f32 - 11.0) / 7.0)
            .collect()
    }

    fn assert_close(a: &Tensor, b: &Tensor, what: &str) {
        assert_eq!(a.dims(), b.dims(), "{what}: shape");
        for (i, (x, y)) in a.data().iter().zip(b.data()).enumerate() {
            assert!(
                (x - y).abs() <= 1e-4 * x.abs().max(1.0),
                "{what}: element {i} is {x} but reference says {y}"
            );
        }
    }

    #[test]
    fn rows_per_task_divides_the_output_and_scales_with_threads() {
        // 4 threads -> 16 tasks; 16 / 5 planes = 3 per plane; ceil(36 / 3) = 12, which divides 36.
        assert_eq!(rows_per_task(36, 1, 5, 4), 12);
        assert_eq!(
            rows_per_task(37, 1, 5, 4),
            37,
            "a prime height has no smaller divisor to round to"
        );
        assert_eq!(
            rows_per_task(36, 4, 5, 4),
            36,
            "more planes than tasks: one task per plane"
        );
        assert_eq!(rows_per_task(0, 1, 5, 4), 1);
        assert_eq!(rows_per_task(1, 1, 5, 4), 1);
        // Planes are batch times channels. With a batch of 1 above, `batch * channels` and
        // `channels` are one number, so this pins the product with two factors that are not:
        // 6 planes -> 2 per plane -> 18 rows, where 2 + 3 = 5 planes would give 12 and
        // 2 / 3 = 0 (clamped to 1) would give 3.
        assert_eq!(rows_per_task(36, 2, 3, 4), 18);
    }

    /// `interior_columns` against its definition, over every small shape including the ones
    /// where the kernel is wider than the padded image.
    ///
    /// The shape-agreement test can only see this bound from one side. A range that is too
    /// narrow only sends columns through the bounds-checked path, which gives the same answer
    /// more slowly, so shrinking it (`w - pad` for `w + pad`, or dropping the `+ 1`) left every
    /// output identical. Too wide is the direction that corrupts, and it happened: a kernel
    /// wider than the padded image made column 0 "interior" and the general path panicked.
    #[test]
    fn interior_columns_are_exactly_the_ones_whose_window_fits() {
        for w in 1..=7 {
            for kw in 1..=5 {
                for stride in 1..=3 {
                    for pad in 0..=2 {
                        let out_w = output_dim(w, kw, stride, pad);
                        // Both conditions are monotone in `ox`, so the set is one interval.
                        let expected: Vec<usize> = (0..out_w)
                            .filter(|&ox| {
                                let start = (ox * stride) as isize - pad as isize;
                                start >= 0 && start + kw as isize <= w as isize
                            })
                            .collect();
                        let got: Vec<usize> =
                            interior_columns(w, kw, stride, pad, out_w).collect();
                        assert_eq!(got, expected, "w {w} kernel {kw} stride {stride} pad {pad}");
                    }
                }
            }
        }
    }

    /// `reference` below calls `output_dim` too, so the shape-agreement test moves with any
    /// mutation of it and cannot see one: replacing the `/ stride` with `* stride` left every
    /// conv test passing and only showed up as a timeout, when the inflated dimensions made
    /// `scrfd_parity` grind. Assert the arithmetic directly, with terms that do not collapse —
    /// stride 3 and padding 2 are distinct from each other and from the kernel.
    #[test]
    fn output_dim_matches_the_convolution_formula() {
        // (13 + 2*2 - 5)/3 + 1 = 12/3 + 1 = 5.
        assert_eq!(output_dim(13, 5, 3, 2), 5);
        // Stride is a division, not a multiplication: halving it must roughly double the output.
        assert_eq!(output_dim(13, 5, 1, 2), 13);
        // Padding widens, the kernel narrows, and neither is the other's inverse here.
        assert_eq!(output_dim(13, 5, 1, 0), 9);
        assert_eq!(output_dim(13, 1, 1, 0), 13);
        // A kernel larger than the padded input saturates to a single output, never underflows.
        assert_eq!(output_dim(3, 9, 2, 1), 1);
    }

    /// Whatever path `conv2d` picks must agree with the reference. Each fast path is valid only
    /// for some shapes: a 3x1 kernel, an unpadded or strided 1x1, a strided depthwise layer or a
    /// depth multiplier of 2 routed to one of them computes the wrong thing, and the detector's
    /// own layers never try those combinations.
    #[test]
    fn every_layer_shape_matches_the_reference_whichever_path_runs() {
        let c_in = 4usize;
        // (groups, out_channels): dense to 1 or 6 channels, depthwise with multiplier 1 or 2.
        let layers = [(1, 1), (1, 6), (c_in, c_in), (c_in, c_in * 2)];
        // 5x7 is non-square so axis swaps show; 3x3 is small enough that interior/border
        // bounds go wrong visibly; 5x2 is narrower than a 3-wide kernel, so with no padding
        // no column is interior at all -- the general path panicked there.
        for (h, w) in [(5usize, 7usize), (3, 3), (5, 2)] {
            // Batch of 2 with the second item negated: batch 0 alone makes every
            // `batch * ...` offset 0.
            let mut data = ramp(2 * c_in * h * w, 30);
            for v in data[c_in * h * w..].iter_mut() {
                *v = -*v;
            }
            let input = Tensor::new(2, c_in, h, w, data).expect("input");
            for (groups, c_out) in layers {
                for (kh, kw) in [(1, 1), (1, 3), (3, 1), (3, 3)] {
                    for stride in [1, 2] {
                        for padding in [0, 1, 2] {
                            let per_group = c_in / groups;
                            let weights = ConvWeights::new(
                                c_out,
                                per_group,
                                kh,
                                kw,
                                ramp(c_out * per_group * kh * kw, 31),
                                ramp(c_out, 32),
                            )
                            .expect("weights");
                            let cfg = ConvConfig {
                                stride,
                                padding,
                                groups,
                                activation: Activation::None,
                            };
                            let what = format!(
                                "{h}x{w} groups {groups} out {c_out} kernel {kh}x{kw} \
                                 stride {stride} pad {padding}"
                            );
                            assert_close(
                                &conv2d(&input, &weights, &cfg)
                                    .unwrap_or_else(|e| panic!("{what}: {e:#}")),
                                &reference(&input, &weights, &cfg),
                                &what,
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn pointwise_matches_the_reference() {
        for (c_in, c_out, h, w) in [
            (3usize, 4usize, 5usize, 7usize),
            (16, 8, 9, 9),
            (1, 1, 1, 1),
        ] {
            let input = Tensor::new(1, c_in, h, w, ramp(c_in * h * w, 1)).expect("input");
            let weights =
                ConvWeights::new(c_out, c_in, 1, 1, ramp(c_out * c_in, 2), ramp(c_out, 3))
                    .expect("weights");
            for act in [Activation::None, Activation::Relu] {
                let cfg = ConvConfig::pointwise(act);
                assert_close(
                    &conv2d(&input, &weights, &cfg).expect("conv"),
                    &reference(&input, &weights, &cfg),
                    &format!("pointwise {c_in}->{c_out} {h}x{w} {act:?}"),
                );
            }
        }
    }

    /// Exercises the interior/border split directly: a 1x1 image is all border,
    /// and a large one is mostly interior.
    #[test]
    fn depthwise_matches_the_reference_including_the_borders() {
        for (c, h, w) in [(4usize, 1usize, 1usize), (3, 2, 2), (8, 9, 11), (2, 17, 5)] {
            let input = Tensor::new(1, c, h, w, ramp(c * h * w, 4)).expect("input");
            let weights =
                ConvWeights::new(c, 1, 3, 3, ramp(c * 9, 5), ramp(c, 6)).expect("weights");
            for act in [Activation::None, Activation::Relu] {
                let cfg = ConvConfig::depthwise(c, 1, act);
                assert_close(
                    &conv2d(&input, &weights, &cfg).expect("conv"),
                    &reference(&input, &weights, &cfg),
                    &format!("depthwise c={c} {h}x{w} {act:?}"),
                );
            }
        }
    }

    /// The stem: 3x3 stride 2 with padding, dense over 3 input channels.
    #[test]
    fn strided_dense_conv_matches_the_reference() {
        let input = Tensor::new(1, 3, 12, 10, ramp(3 * 12 * 10, 7)).expect("input");
        let weights =
            ConvWeights::new(16, 3, 3, 3, ramp(16 * 3 * 9, 8), ramp(16, 9)).expect("weights");
        let cfg = ConvConfig {
            stride: 2,
            padding: 1,
            groups: 1,
            activation: Activation::Relu,
        };
        assert_close(
            &conv2d(&input, &weights, &cfg).expect("conv"),
            &reference(&input, &weights, &cfg),
            "stem conv",
        );
    }

    /// Grouped but not depthwise, so the group indexing is checked independently
    /// of the depthwise fast path.
    #[test]
    fn grouped_conv_matches_the_reference() {
        let (c_in, c_out, groups) = (8usize, 4usize, 2usize);
        let input = Tensor::new(1, c_in, 6, 6, ramp(c_in * 36, 10)).expect("input");
        let weights = ConvWeights::new(
            c_out,
            c_in / groups,
            3,
            3,
            ramp(c_out * (c_in / groups) * 9, 11),
            ramp(c_out, 12),
        )
        .expect("weights");
        let cfg = ConvConfig {
            stride: 1,
            padding: 1,
            groups,
            activation: Activation::None,
        };
        assert_close(
            &conv2d(&input, &weights, &cfg).expect("conv"),
            &reference(&input, &weights, &cfg),
            "grouped conv",
        );
    }

    #[test]
    fn batches_do_not_bleed_into_each_other() {
        let c = 3;
        let mut data = ramp(2 * c * 4 * 4, 13);
        // Make the second batch item obviously different from the first.
        for v in data[c * 16..].iter_mut() {
            *v = -*v;
        }
        let input = Tensor::new(2, c, 4, 4, data).expect("input");
        let weights = ConvWeights::new(c, 1, 3, 3, ramp(c * 9, 14), ramp(c, 15)).expect("weights");
        let cfg = ConvConfig::depthwise(c, 1, Activation::Relu);
        assert_close(
            &conv2d(&input, &weights, &cfg).expect("conv"),
            &reference(&input, &weights, &cfg),
            "batched depthwise",
        );
    }

    /// A batch of 2 with 10 output channels. Chunking the output buffer without
    /// splitting by batch item first lets a chunk cross the batch boundary and
    /// pair one image's outputs with the other's inputs. An earlier version of
    /// this path did exactly that.
    #[test]
    fn pointwise_blocks_do_not_straddle_the_batch_boundary() {
        let (c_in, c_out) = (5usize, 10usize);
        let mut data = ramp(2 * c_in * 6 * 6, 20);
        for v in data[c_in * 36..].iter_mut() {
            *v = -*v * 2.0;
        }
        let input = Tensor::new(2, c_in, 6, 6, data).expect("input");
        let weights = ConvWeights::new(c_out, c_in, 1, 1, ramp(c_out * c_in, 21), ramp(c_out, 22))
            .expect("weights");
        let cfg = ConvConfig::pointwise(Activation::None);
        assert_close(
            &conv2d(&input, &weights, &cfg).expect("conv"),
            &reference(&input, &weights, &cfg),
            "batched pointwise with a ragged block",
        );
    }

    #[test]
    fn mismatched_weights_are_rejected() {
        let input = Tensor::new(1, 4, 3, 3, vec![0.0; 36]).expect("input");
        let weights = ConvWeights::new(4, 2, 1, 1, vec![0.0; 8], vec![0.0; 4]).expect("weights");
        let err = conv2d(&input, &weights, &ConvConfig::pointwise(Activation::None))
            .expect_err("4 input channels cannot feed weights expecting 2");
        assert!(
            format!("{err}").contains("input channels per group"),
            "{err}"
        );
    }
}
