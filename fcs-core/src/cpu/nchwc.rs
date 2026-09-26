//! Channel-blocked ("NCHWc") kernels: the layout ONNX Runtime runs CPU convolutions in.
//!
//! Plain NCHW stores one channel plane after another, so a convolution computes one output
//! channel at a time and rereads its whole input for each. Here channels are grouped in
//! blocks of [`BLOCK`], and one pixel of one block is a single `f32x8`. Every kernel then
//! works on eight channels per instruction, and a dense convolution keeps a tile of output
//! blocks x pixels in registers while it walks the input once -- the design of
//! `onnxruntime/core/mlas/lib/snchwc.cpp` and its FMA3 kernel (`SconvKernelFma3.S`), which is
//! what makes ONNX Runtime six times faster per core than the NCHW kernels in [`super::conv2d`].
//!
//! `f32x8` is `wide`'s portable vector: one AVX2 register on the x86-64-v3 builds, two NEON
//! registers on Apple silicon, so there is one code path for every platform. The block is 8
//! on all of them; MLAS uses 16 only with AVX-512, which the builds do not target.

use std::{mem::MaybeUninit, ops::Range};

use anyhow::{Result, ensure};
use rayon::prelude::*;
use wide::f32x8;

use super::tensor::Tensor;

/// Channels per block.
pub const BLOCK: usize = 8;

/// Output blocks one dense-convolution task computes together. Each input value loaded is
/// used by all of them, which is what the tile exists for.
const FILTERS: usize = 4;

/// Buffers of activations nothing will read again, reused for later outputs.
///
/// A network run used to allocate every layer's output afresh -- up to 6.5 MB each, 66 times
/// -- and the allocator hands large freed blocks back to the OS, so each run paid page faults
/// on memory it had just released. Recycling instead is ONNX Runtime's memory planning in its
/// simplest form: a buffer that comes back is still committed, and often still in cache.
#[derive(Debug, Default)]
pub struct Arena {
    free: Vec<Vec<f32x8>>,
}

impl Arena {
    /// Give back a tensor that nothing will read again.
    pub fn recycle(&mut self, tensor: Blocked) {
        self.free.push(tensor.data);
    }

    /// Whether it holds no buffers.
    pub fn is_empty(&self) -> bool {
        self.free.is_empty()
    }

    /// Room for `len` elements: the smallest free buffer that fits, or a new one.
    fn take(&mut self, len: usize) -> Vec<MaybeUninit<f32x8>> {
        let fit = self
            .free
            .iter()
            .enumerate()
            .filter(|(_, buffer)| buffer.capacity() >= len)
            .min_by_key(|(_, buffer)| buffer.capacity())
            .map(|(index, _)| index);
        let buffer = match fit {
            Some(index) => self.free.swap_remove(index),
            None => Vec::with_capacity(len),
        };
        let mut buffer = std::mem::ManuallyDrop::new(buffer);
        // SAFETY: `MaybeUninit<f32x8>` has the layout of `f32x8`, and treating initialised
        // contents as uninitialised only forgets what they were.
        let mut data = unsafe {
            Vec::from_raw_parts(
                buffer.as_mut_ptr().cast::<MaybeUninit<f32x8>>(),
                0,
                buffer.capacity(),
            )
        };
        // SAFETY: the capacity is at least `len`, and `MaybeUninit` needs no initialisation.
        unsafe { data.set_len(len) };
        data
    }
}

/// Activations with channels grouped in blocks of [`BLOCK`].
///
/// Element `((n * blocks + c / 8) * height + y) * width + x` holds channels
/// `c / 8 * 8 .. + 8` of pixel `(y, x)`. Channels past `channels` in the last block are
/// zero, so they contribute nothing to the next layer.
#[derive(Debug, Clone)]
pub struct Blocked {
    batch: usize,
    channels: usize,
    height: usize,
    width: usize,
    data: Vec<f32x8>,
}

impl Blocked {
    /// A tensor whose every element `fill` writes, in a buffer from `arena` that is not
    /// zeroed first.
    ///
    /// Zeroing is a single-threaded memset of up to 6.5 MB per layer, and it was what kept
    /// the graph from scaling past four threads: every kernel overwrites its whole output
    /// from parallel tasks anyway. Debug builds fill with NaN first, so an element a kernel
    /// forgets shows up in the tests instead of reading as garbage.
    ///
    /// # Safety
    ///
    /// `fill` must write every element of the slice it is given.
    unsafe fn written(
        arena: &mut Arena,
        batch: usize,
        channels: usize,
        height: usize,
        width: usize,
        fill: impl FnOnce(&mut [MaybeUninit<f32x8>]),
    ) -> Self {
        let len = batch * channels.div_ceil(BLOCK) * height * width;
        let mut data = arena.take(len);
        if cfg!(debug_assertions) {
            data.fill(MaybeUninit::new(f32x8::splat(f32::NAN)));
        }
        fill(&mut data);
        let mut data = std::mem::ManuallyDrop::new(data);
        // SAFETY: the caller wrote every element, and `MaybeUninit<f32x8>` has the layout of
        // `f32x8`, so the allocation is handed over unchanged.
        let data =
            unsafe { Vec::from_raw_parts(data.as_mut_ptr().cast::<f32x8>(), len, data.capacity()) };
        Self {
            batch,
            channels,
            height,
            width,
            data,
        }
    }

    #[cfg(test)]
    fn zeros(batch: usize, channels: usize, height: usize, width: usize) -> Self {
        Self {
            batch,
            channels,
            height,
            width,
            data: vec![f32x8::ZERO; batch * channels.div_ceil(BLOCK) * height * width],
        }
    }

    /// Channel blocks, the last one possibly part padding.
    pub fn blocks(&self) -> usize {
        self.channels.div_ceil(BLOCK)
    }

    /// Real channels, not counting the padding in the last block.
    pub fn channels(&self) -> usize {
        self.channels
    }

    /// Channels `first..first + count`, back in NCHW.
    pub fn to_nchw(&self, first: usize, count: usize) -> Result<Tensor> {
        ensure!(
            first + count <= self.channels,
            "channels {first}..{} of a {}-channel tensor",
            first + count,
            self.channels
        );
        let plane = self.height * self.width;
        let lanes: &[f32] = bytemuck::cast_slice(&self.data);
        let mut data = Vec::with_capacity(self.batch * count * plane);
        for n in 0..self.batch {
            for c in first..first + count {
                let block = (n * self.blocks() + c / BLOCK) * plane;
                data.extend((0..plane).map(|p| lanes[(block + p) * BLOCK + c % BLOCK]));
            }
        }
        Tensor::new(self.batch, count, self.height, self.width, data)
    }

    /// Mean of every channel over its plane, one value per real channel and batch item.
    pub fn global_average(&self) -> Vec<f32> {
        let plane = self.height * self.width;
        let mut means = Vec::with_capacity(self.batch * self.channels);
        for n in 0..self.batch {
            for b in 0..self.blocks() {
                let start = (n * self.blocks() + b) * plane;
                let sum = self.data[start..start + plane]
                    .iter()
                    .fold(f32x8::ZERO, |acc, &v| acc + v);
                let take = BLOCK.min(self.channels - b * BLOCK);
                means.extend(sum.to_array()[..take].iter().map(|s| s / plane as f32));
            }
        }
        means
    }
}

/// What a dense convolution reads: blocked activations, or a plain NCHW tensor seen as
/// blocks of one channel.
///
/// The second is for a network's first layer. Its input has 3 channels, and padding them to
/// a block of 8 would multiply the most expensive layer's arithmetic by 8/3; reading NCHW
/// directly is ONNX Runtime's `MLAS_NCHWC_CONV_NCHW_ALGORITHM`.
#[derive(Debug, Clone, Copy)]
pub struct Source<'a> {
    lanes: &'a [f32],
    batch: usize,
    blocks: usize,
    block: usize,
    height: usize,
    width: usize,
}

impl<'a> From<&'a Tensor> for Source<'a> {
    fn from(tensor: &'a Tensor) -> Self {
        let [batch, channels, height, width] = tensor.dims();
        Self {
            lanes: tensor.data(),
            batch,
            blocks: channels,
            block: 1,
            height,
            width,
        }
    }
}

impl<'a> From<&'a Blocked> for Source<'a> {
    fn from(tensor: &'a Blocked) -> Self {
        Self {
            lanes: bytemuck::cast_slice(&tensor.data),
            batch: tensor.batch,
            blocks: tensor.blocks(),
            block: BLOCK,
            height: tensor.height,
            width: tensor.width,
        }
    }
}

/// A dense convolution's weights, reordered for [`conv`].
///
/// For output block `ob`, input block `ib`, tap `(ky, kx)` and channel `j` within the input
/// block, one vector holds the weights of that input channel for the block's eight output
/// channels, at `(((ob * in_blocks + ib) * k + ky) * k + kx) * block_in + j`. Padding
/// channels on either side carry zero weights.
#[derive(Debug)]
pub struct DenseWeights {
    out_channels: usize,
    in_blocks: usize,
    block_in: usize,
    kernel: usize,
    data: Vec<f32x8>,
    bias: Vec<f32x8>,
}

impl DenseWeights {
    /// Reorder ONNX's `[out][in][k][k]` weights. `block_in` is the input's block: 1 when the
    /// convolution reads a plain NCHW tensor, [`BLOCK`] when it reads a [`Blocked`] one.
    pub fn new(
        out_channels: usize,
        in_channels: usize,
        kernel: usize,
        block_in: usize,
        weights: &[f32],
        bias: &[f32],
    ) -> Result<Self> {
        ensure!(
            weights.len() == out_channels * in_channels * kernel * kernel,
            "weights {out_channels}x{in_channels}x{kernel}x{kernel} need {} values, got {}",
            out_channels * in_channels * kernel * kernel,
            weights.len()
        );
        ensure!(
            bias.len() == out_channels,
            "bias needs {out_channels} values"
        );
        ensure!(block_in == 1 || block_in == BLOCK, "input block {block_in}");
        let in_blocks = in_channels.div_ceil(block_in);
        let taps = kernel * kernel;
        let mut data =
            Vec::with_capacity(out_channels.div_ceil(BLOCK) * in_blocks * taps * block_in);
        for ob in 0..out_channels.div_ceil(BLOCK) {
            for ib in 0..in_blocks {
                for tap in 0..taps {
                    for j in 0..block_in {
                        let ic = ib * block_in + j;
                        data.push(lanes(|lane| {
                            let oc = ob * BLOCK + lane;
                            if oc < out_channels && ic < in_channels {
                                weights[(oc * in_channels + ic) * taps + tap]
                            } else {
                                0.0
                            }
                        }));
                    }
                }
            }
        }
        Ok(Self {
            out_channels,
            in_blocks,
            block_in,
            kernel,
            data,
            bias: blocked_bias(bias),
        })
    }

    fn per_block(&self) -> usize {
        self.in_blocks * self.kernel * self.kernel * self.block_in
    }
}

/// A depthwise convolution's weights, reordered for [`depthwise`]: one vector per block and
/// tap, at `(block * k + ky) * k + kx`.
#[derive(Debug)]
pub struct DepthwiseWeights {
    channels: usize,
    kernel: usize,
    data: Vec<f32x8>,
    bias: Vec<f32x8>,
}

impl DepthwiseWeights {
    /// Reorder ONNX's `[channels][1][k][k]` weights.
    pub fn new(channels: usize, kernel: usize, weights: &[f32], bias: &[f32]) -> Result<Self> {
        let taps = kernel * kernel;
        ensure!(
            weights.len() == channels * taps,
            "depthwise weights need {} values, got {}",
            channels * taps,
            weights.len()
        );
        ensure!(bias.len() == channels, "bias needs {channels} values");
        let mut data = Vec::with_capacity(channels.div_ceil(BLOCK) * taps);
        for b in 0..channels.div_ceil(BLOCK) {
            for tap in 0..taps {
                data.push(lanes(|lane| {
                    let c = b * BLOCK + lane;
                    if c < channels {
                        weights[c * taps + tap]
                    } else {
                        0.0
                    }
                }));
            }
        }
        Ok(Self {
            channels,
            kernel,
            data,
            bias: blocked_bias(bias),
        })
    }
}

fn lanes(f: impl FnMut(usize) -> f32) -> f32x8 {
    f32x8::new(std::array::from_fn(f))
}

fn blocked_bias(bias: &[f32]) -> Vec<f32x8> {
    (0..bias.len().div_ceil(BLOCK))
        .map(|b| lanes(|lane| bias.get(b * BLOCK + lane).copied().unwrap_or(0.0)))
        .collect()
}

fn output_dim(input: usize, kernel: usize, stride: usize, pad: usize) -> usize {
    (input + 2 * pad).saturating_sub(kernel) / stride + 1
}

/// Output columns `lo..hi` whose every tap lands inside the input, so they need no checks.
fn interior(
    width: usize,
    kernel: usize,
    stride: usize,
    pad: usize,
    out_w: usize,
) -> (usize, usize) {
    let hi = (width + pad)
        .checked_sub(kernel)
        .map_or(0, |span| span / stride + 1)
        .min(out_w);
    (pad.div_ceil(stride).min(hi), hi)
}

/// Rows of taps `lo..hi` that land inside the input for output row `oy`.
fn tap_rows(oy: usize, height: usize, kernel: usize, stride: usize, pad: usize) -> (usize, usize) {
    let top = (oy * stride) as isize - pad as isize;
    let lo = (-top).clamp(0, kernel as isize) as usize;
    let hi = (height as isize - top).clamp(0, kernel as isize) as usize;
    (lo, hi.max(lo))
}

/// Enough jobs per rayon task that queueing is not the cost, few enough that every thread
/// gets several.
fn min_len(jobs: usize) -> usize {
    (jobs / (rayon::current_num_threads() * 4)).max(1)
}

/// Output pointer shared by rayon tasks that each write their own elements.
#[derive(Clone, Copy)]
struct OutPtr(*mut MaybeUninit<f32x8>);
// SAFETY: every caller hands each task a disjoint set of output elements.
unsafe impl Send for OutPtr {}
unsafe impl Sync for OutPtr {}

impl OutPtr {
    /// A method rather than `.0`, so closures capture the `Send` wrapper and not the bare
    /// pointer inside it.
    fn at(self, index: usize) -> *mut MaybeUninit<f32x8> {
        // SAFETY: callers index inside the output allocation.
        unsafe { self.0.add(index) }
    }
}

/// Dense convolution, fusing an optional ReLU. Covers KxK at any stride and padding,
/// including 1x1, reading either blocked or NCHW input (see [`Source`]).
pub fn conv(
    arena: &mut Arena,
    input: Source<'_>,
    weights: &DenseWeights,
    stride: usize,
    pad: usize,
    relu: bool,
) -> Result<Blocked> {
    ensure!(stride > 0, "stride must be non-zero");
    ensure!(
        input.block == weights.block_in && input.blocks == weights.in_blocks,
        "weights expect {} blocks of {}, input has {} of {}",
        weights.in_blocks,
        weights.block_in,
        input.blocks,
        input.block
    );
    let k = weights.kernel;
    let out_h = output_dim(input.height, k, stride, pad);
    let out_w = output_dim(input.width, k, stride, pad);
    let (batch, out_channels, full_h, full_w) = (input.batch, weights.out_channels, out_h, out_w);
    let out_blocks = out_channels.div_ceil(BLOCK);

    // A 1x1 stride-1 convolution has no geometry, so each plane is computed as one long row:
    // register tiles then run straight across row ends instead of stopping at every one.
    let mut input = input;
    let (out_h, out_w) = if k == 1 && stride == 1 && pad == 0 {
        (input.height, input.width) = (1, input.height * input.width);
        (1, out_h * out_w)
    } else {
        (out_h, out_w)
    };
    let geometry = Geometry {
        stride,
        pad,
        out_h,
        out_w,
        out_blocks,
        relu,
    };
    let groups = out_blocks.div_ceil(FILTERS);
    let rows = input.batch * groups * out_h;
    // Long rows are split so every thread gets work; a 1x1 layer is one row per plane.
    let wanted = (rayon::current_num_threads() * 4).div_ceil(rows.max(1));
    let chunk = out_w
        .div_ceil(wanted.clamp(1, out_w.div_ceil(48).max(1)))
        .max(1);
    let chunks = out_w.div_ceil(chunk);
    let jobs = rows * chunks;

    // SAFETY: the jobs cover every (batch item, block group, row, column chunk), and each
    // `Row::run` writes every column of its chunk for each of its blocks.
    let output = unsafe {
        Blocked::written(arena, batch, out_channels, full_h, full_w, |data| {
            let out = OutPtr(data.as_mut_ptr());
            (0..jobs)
                .into_par_iter()
                .with_min_len(min_len(jobs))
                .for_each(|job| {
                    let (row_job, c) = (job / chunks, job % chunks);
                    let (n, group, oy) = (
                        row_job / (groups * out_h),
                        row_job / out_h % groups,
                        row_job % out_h,
                    );
                    let first = group * FILTERS;
                    let row = Row {
                        input: &input,
                        weights,
                        geometry: &geometry,
                        n,
                        first,
                        oy,
                        out,
                    };
                    let span = c * chunk..((c + 1) * chunk).min(out_w);
                    // Accumulators F x P, plus P broadcasts and one weight, within sixteen registers.
                    match FILTERS.min(geometry.out_blocks - first) {
                        1 => row.run::<1, 8>(span),
                        2 => row.run::<2, 5>(span),
                        3 => row.run::<3, 4>(span),
                        _ => row.run::<4, 3>(span),
                    }
                });
        })
    };
    Ok(output)
}

struct Geometry {
    stride: usize,
    pad: usize,
    out_h: usize,
    out_w: usize,
    out_blocks: usize,
    relu: bool,
}

/// Part of one output row, for `F` output blocks.
struct Row<'a> {
    input: &'a Source<'a>,
    weights: &'a DenseWeights,
    geometry: &'a Geometry,
    n: usize,
    first: usize,
    oy: usize,
    out: OutPtr,
}

impl Row<'_> {
    /// Columns `span` in tiles of `P` where no tap can leave the input, one checked pixel at
    /// a time at the borders.
    fn run<const F: usize, const P: usize>(&self, span: Range<usize>) {
        let g = self.geometry;
        let (lo, hi) = interior(
            self.input.width,
            self.weights.kernel,
            g.stride,
            g.pad,
            g.out_w,
        );
        let lo = lo.clamp(span.start, span.end);
        let hi = hi.clamp(lo, span.end);
        let mut ox = span.start;
        while ox < lo {
            self.tile::<F, 1, true>(ox);
            ox += 1;
        }
        while ox + P <= hi {
            self.tile::<F, P, false>(ox);
            ox += P;
        }
        while ox < hi {
            self.tile::<F, 1, false>(ox);
            ox += 1;
        }
        while ox < span.end {
            self.tile::<F, 1, true>(ox);
            ox += 1;
        }
    }

    /// `F` output blocks x `P` pixels from column `ox`, in registers. `CHECK` skips taps that
    /// fall outside the input, which only border pixels need.
    ///
    /// The loop order is MLAS's: broadcast the `P` input values once, then stream each
    /// weight vector through `P` FMAs. Holding the `F` weights instead would need `F` more
    /// registers than AVX2 has.
    #[inline(always)]
    fn tile<const F: usize, const P: usize, const CHECK: bool>(&self, ox: usize) {
        let (input, weights, g) = (self.input, self.weights, self.geometry);
        let (k, block, width) = (weights.kernel, input.block, input.width);
        let per_block = weights.per_block();
        let filters = &weights.data[self.first * per_block..(self.first + F) * per_block];
        let mut acc: [[f32x8; P]; F] = std::array::from_fn(|f| [weights.bias[self.first + f]; P]);

        let (ky_lo, ky_hi) = tap_rows(self.oy, input.height, k, g.stride, g.pad);
        let top = self.oy * g.stride + ky_lo - g.pad;
        for ib in 0..input.blocks {
            let plane = (self.n * input.blocks + ib) * input.height;
            for ky in ky_lo..ky_hi {
                let row = (plane + top + ky - ky_lo) * width;
                for kx in 0..k {
                    let mut at = [0usize; P];
                    let mut inside = true;
                    for (p, at) in at.iter_mut().enumerate() {
                        let ix = ((ox + p) * g.stride + kx).wrapping_sub(g.pad);
                        if CHECK && ix >= width {
                            inside = false;
                        } else {
                            *at = (row + ix) * block;
                        }
                    }
                    if !inside {
                        continue;
                    }
                    debug_assert!(at[P - 1] + block <= input.lanes.len());
                    let tap = ((ib * k + ky) * k + kx) * block;
                    for j in 0..block {
                        // SAFETY: `at[p]` is the first lane of an in-bounds pixel -- interior
                        // tiles by construction of `interior` and `tap_rows`, border ones by the
                        // check above -- and `j < block`.
                        let x: [f32x8; P] = std::array::from_fn(|p| {
                            f32x8::splat(unsafe { *input.lanes.get_unchecked(at[p] + j) })
                        });
                        for (f, acc) in acc.iter_mut().enumerate() {
                            // SAFETY: `f < F`, and `tap + j < per_block` because `ib`, `ky`,
                            // `kx` and `j` are each below the sizes `per_block` multiplies.
                            let w = unsafe { *filters.get_unchecked(f * per_block + tap + j) };
                            for p in 0..P {
                                acc[p] = x[p].mul_add(w, acc[p]);
                            }
                        }
                    }
                }
            }
        }

        for (f, pixels) in acc.iter().enumerate() {
            let base =
                ((self.n * g.out_blocks + self.first + f) * g.out_h + self.oy) * g.out_w + ox;
            for (p, &v) in pixels.iter().enumerate() {
                let v = if g.relu { v.max(f32x8::ZERO) } else { v };
                // SAFETY: columns `ox..ox + P` of row `oy` of blocks `first..first + F` belong
                // to this task alone, and `ox + P <= out_w`.
                unsafe { self.out.at(base + p).write(MaybeUninit::new(v)) };
            }
        }
    }
}

/// Depthwise convolution (`groups == channels`), fusing an optional ReLU.
pub fn depthwise(
    arena: &mut Arena,
    input: &Blocked,
    weights: &DepthwiseWeights,
    stride: usize,
    pad: usize,
    relu: bool,
) -> Result<Blocked> {
    ensure!(stride > 0, "stride must be non-zero");
    ensure!(
        input.channels == weights.channels,
        "depthwise weights for {} channels, input has {}",
        weights.channels,
        input.channels
    );
    let k = weights.kernel;
    let (h, w) = (input.height, input.width);
    let out_h = output_dim(h, k, stride, pad);
    let out_w = output_dim(w, k, stride, pad);
    let blocks = input.blocks();
    let (lo, hi) = interior(w, k, stride, pad, out_w);
    let rows = input.batch * blocks * out_h;

    // SAFETY: one task per output row, and each writes every column of it.
    let output = unsafe {
        Blocked::written(arena, input.batch, input.channels, out_h, out_w, |data| {
            data.par_chunks_mut(out_w.max(1))
                .with_min_len(min_len(rows))
                .enumerate()
                .for_each(|(r, out_row)| {
                    let (n, b, oy) = (r / (blocks * out_h), r / out_h % blocks, r % out_h);
                    let (ky_lo, ky_hi) = tap_rows(oy, h, k, stride, pad);
                    let first_row = oy * stride + ky_lo - pad;
                    let plane = &input.data[(n * blocks + b) * h * w..(n * blocks + b + 1) * h * w];
                    let row = DepthwiseRow {
                        rows: &plane[first_row * w..(first_row + ky_hi - ky_lo) * w],
                        taps: &weights.data[(b * k + ky_lo) * k..(b * k + ky_hi) * k],
                        bias: weights.bias[b],
                        width: w,
                        kernel: k,
                        stride,
                        pad,
                        relu,
                    };
                    let mut ox = 0;
                    while ox < lo {
                        row.tile::<1, true>(ox, out_row);
                        ox += 1;
                    }
                    while ox + 4 <= hi {
                        row.tile::<4, false>(ox, out_row);
                        ox += 4;
                    }
                    while ox < out_w {
                        row.tile::<1, true>(ox, out_row);
                        ox += 1;
                    }
                });
        })
    };
    Ok(output)
}

/// The input rows and taps one depthwise output row reads.
struct DepthwiseRow<'a> {
    /// The input rows whose taps land inside the image, top to bottom.
    rows: &'a [f32x8],
    /// The kernel rows matching `rows`, `kernel` taps each.
    taps: &'a [f32x8],
    bias: f32x8,
    width: usize,
    kernel: usize,
    stride: usize,
    pad: usize,
    relu: bool,
}

impl DepthwiseRow<'_> {
    /// `P` output pixels from `ox`. One pixel is a chain of nine dependent FMAs, bound by
    /// their latency; `P` independent chains keep the FMA units busy instead.
    #[inline(always)]
    fn tile<const P: usize, const CHECK: bool>(&self, ox: usize, out: &mut [MaybeUninit<f32x8>]) {
        let mut acc = [self.bias; P];
        for (row, taps) in self
            .rows
            .chunks_exact(self.width)
            .zip(self.taps.chunks_exact(self.kernel))
        {
            for (kx, &tap) in taps.iter().enumerate() {
                for (p, acc) in acc.iter_mut().enumerate() {
                    let ix = ((ox + p) * self.stride + kx).wrapping_sub(self.pad);
                    if CHECK && ix >= self.width {
                        continue;
                    }
                    *acc = row[ix].mul_add(tap, *acc);
                }
            }
        }
        for (o, v) in out[ox..ox + P].iter_mut().zip(acc) {
            o.write(if self.relu { v.max(f32x8::ZERO) } else { v });
        }
    }
}

/// Elementwise sum of two tensors of one shape.
pub fn add(arena: &mut Arena, lhs: &Blocked, rhs: &Blocked) -> Result<Blocked> {
    ensure!(
        (lhs.batch, lhs.channels, lhs.height, lhs.width)
            == (rhs.batch, rhs.channels, rhs.height, rhs.width),
        "add expects matching shapes"
    );
    // SAFETY: the chunks of `data`, `lhs` and `rhs` line up one to one, all of one length.
    let out = unsafe {
        Blocked::written(
            arena,
            lhs.batch,
            lhs.channels,
            lhs.height,
            lhs.width,
            |data| {
                data.par_chunks_mut(4096)
                    .zip(lhs.data.par_chunks(4096).zip(rhs.data.par_chunks(4096)))
                    .for_each(|(o, (a, b))| {
                        for (o, (a, b)) in o.iter_mut().zip(a.iter().zip(b)) {
                            o.write(*a + *b);
                        }
                    });
            },
        )
    };
    Ok(out)
}

/// Nearest-neighbour 2x upsample.
pub fn upsample2x(arena: &mut Arena, input: &Blocked) -> Result<Blocked> {
    let (h, w) = (input.height, input.width);
    let rows = input.batch * input.blocks() * h * 2;
    // SAFETY: one task per output row, and each writes every column of it.
    let output = unsafe {
        Blocked::written(arena, input.batch, input.channels, h * 2, w * 2, |data| {
            data.par_chunks_mut((w * 2).max(1))
                .with_min_len(min_len(rows))
                .enumerate()
                .for_each(|(r, out_row)| {
                    // Output row r of plane r / (2h) comes from input row r / 2 of the same plane.
                    let (plane, oy) = (r / (2 * h), r % (2 * h));
                    let in_row = &input.data[(plane * h + oy / 2) * w..][..w];
                    for (x, o) in out_row.iter_mut().enumerate() {
                        o.write(in_row[x / 2]);
                    }
                });
        })
    };
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A convolution written straight from the definition, with no special cases, for the
    /// kernels to be checked against: the interior/border split and the register tiles are
    /// exactly the rewrites that stay right mid-image and go wrong at an edge. Weights are
    /// ONNX's `[out][in / groups][k][k]`.
    #[allow(clippy::too_many_arguments)]
    fn reference(
        input: &Tensor,
        weights: &[f32],
        bias: &[f32],
        c_out: usize,
        k: usize,
        stride: usize,
        pad: usize,
        groups: usize,
        relu: bool,
    ) -> Tensor {
        let [n, c_in, h, w] = input.dims();
        let out_h = output_dim(h, k, stride, pad);
        let out_w = output_dim(w, k, stride, pad);
        let (group_in, group_out) = (c_in / groups, c_out / groups);
        let mut out = Vec::with_capacity(n * c_out * out_h * out_w);
        for b in 0..n {
            for oc in 0..c_out {
                for oy in 0..out_h {
                    for ox in 0..out_w {
                        let mut acc = bias[oc];
                        for lic in 0..group_in {
                            let ic = oc / group_out * group_in + lic;
                            for ky in 0..k {
                                for kx in 0..k {
                                    let iy = (oy * stride + ky) as isize - pad as isize;
                                    let ix = (ox * stride + kx) as isize - pad as isize;
                                    if iy < 0 || iy >= h as isize || ix < 0 || ix >= w as isize {
                                        continue;
                                    }
                                    acc += weights[((oc * group_in + lic) * k + ky) * k + kx]
                                        * input.plane_slice(b, ic)[iy as usize * w + ix as usize];
                                }
                            }
                        }
                        out.push(if relu { acc.max(0.0) } else { acc });
                    }
                }
            }
        }
        Tensor::new(n, c_out, out_h, out_w, out).unwrap()
    }

    fn ramp(len: usize, seed: usize) -> Vec<f32> {
        (0..len)
            .map(|i| (((i * 37 + seed * 17) % 23) as f32 - 11.0) / 7.0)
            .collect()
    }

    fn blocked(tensor: &Tensor) -> Blocked {
        let [n, c, h, w] = tensor.dims();
        let mut out = Blocked::zeros(n, c, h, w);
        let lanes: &mut [f32] = bytemuck::cast_slice_mut(&mut out.data);
        for b in 0..n {
            for ch in 0..c {
                for (p, &v) in tensor.plane_slice(b, ch).iter().enumerate() {
                    lanes[(((b * c.div_ceil(BLOCK) + ch / BLOCK) * h * w) + p) * BLOCK
                        + ch % BLOCK] = v;
                }
            }
        }
        out
    }

    fn assert_close(got: &Tensor, want: &Tensor, what: &str) {
        assert_eq!(got.dims(), want.dims(), "{what}: shape");
        for (i, (x, y)) in got.data().iter().zip(want.data()).enumerate() {
            assert!(
                (x - y).abs() <= 1e-4 * y.abs().max(1.0),
                "{what}: element {i} is {x}, the definition says {y}"
            );
        }
    }

    /// Channel counts straddle block boundaries (5, 13) and fill them (8, 16); output blocks
    /// of 1, 2, 3 and 4 + 1 reach every register tile shape; widths are narrow enough that
    /// border pixels outnumber interior ones and wide enough for full tiles, and a 9x16 1x1
    /// layer is split across tasks mid-plane; batch 2 moves every offset.
    #[test]
    fn dense_matches_the_definition() {
        for (c_in, c_out) in [(3, 16), (8, 5), (16, 13), (8, 24), (13, 40)] {
            for (h, w) in [(1, 1), (5, 7), (9, 16)] {
                let input = Tensor::new(2, c_in, h, w, ramp(2 * c_in * h * w, c_in)).unwrap();
                for (k, stride, pad) in [(1, 1, 0), (3, 1, 1), (3, 2, 1), (3, 1, 0), (1, 2, 0)] {
                    let wts = ramp(c_out * c_in * k * k, c_out);
                    let bias = ramp(c_out, 3);
                    let reference = reference(&input, &wts, &bias, c_out, k, stride, pad, 1, true);
                    let what = format!("{c_in}->{c_out} {h}x{w} k{k} s{stride} p{pad}");
                    // NCHW input as blocks of one, and blocked input.
                    let from_nchw = DenseWeights::new(c_out, c_in, k, 1, &wts, &bias).unwrap();
                    let got = conv(
                        &mut Arena::default(),
                        (&input).into(),
                        &from_nchw,
                        stride,
                        pad,
                        true,
                    )
                    .unwrap();
                    assert_close(&got.to_nchw(0, c_out).unwrap(), &reference, &what);
                    let packed = blocked(&input);
                    let from_blocked =
                        DenseWeights::new(c_out, c_in, k, BLOCK, &wts, &bias).unwrap();
                    let got = conv(
                        &mut Arena::default(),
                        (&packed).into(),
                        &from_blocked,
                        stride,
                        pad,
                        true,
                    )
                    .unwrap();
                    assert_close(&got.to_nchw(0, c_out).unwrap(), &reference, &what);
                }
            }
        }
    }

    #[test]
    fn depthwise_matches_the_definition() {
        for c in [5, 16, 40] {
            for (h, w) in [(1, 1), (5, 7), (10, 11)] {
                let input = Tensor::new(2, c, h, w, ramp(2 * c * h * w, c)).unwrap();
                for (stride, pad) in [(1, 1), (2, 1), (1, 0)] {
                    let wts = ramp(c * 9, 4);
                    let bias = ramp(c, 5);
                    let reference = reference(&input, &wts, &bias, c, 3, stride, pad, c, false);
                    let weights = DepthwiseWeights::new(c, 3, &wts, &bias).unwrap();
                    let got = depthwise(
                        &mut Arena::default(),
                        &blocked(&input),
                        &weights,
                        stride,
                        pad,
                        false,
                    )
                    .unwrap();
                    assert_close(
                        &got.to_nchw(0, c).unwrap(),
                        &reference,
                        &format!("depthwise c{c} {h}x{w} s{stride} p{pad}"),
                    );
                }
            }
        }
    }

    #[test]
    fn add_upsample_and_channel_ranges_round_trip() {
        let a = Tensor::new(2, 11, 3, 4, ramp(2 * 11 * 12, 6)).unwrap();
        let b = Tensor::new(2, 11, 3, 4, ramp(2 * 11 * 12, 7)).unwrap();
        let sum = add(&mut Arena::default(), &blocked(&a), &blocked(&b))
            .unwrap()
            .to_nchw(0, 11)
            .unwrap();
        let want: Vec<f32> = a.data().iter().zip(b.data()).map(|(x, y)| x + y).collect();
        assert_eq!(sum.data(), want, "add");

        let up = upsample2x(&mut Arena::default(), &blocked(&a))
            .unwrap()
            .to_nchw(0, 11)
            .unwrap();
        assert_eq!(up.dims(), [2, 11, 6, 8]);
        for n in 0..2 {
            for c in 0..11 {
                for (i, &v) in up.plane_slice(n, c).iter().enumerate() {
                    let (y, x) = (i / 8, i % 8);
                    assert_eq!(v, a.plane_slice(n, c)[y / 2 * 4 + x / 2], "upsample");
                }
            }
        }

        // A range starting mid-block and crossing into the next.
        let part = blocked(&a).to_nchw(6, 4).unwrap();
        for n in 0..2 {
            for c in 0..4 {
                assert_eq!(part.plane_slice(n, c), a.plane_slice(n, c + 6));
            }
        }
    }

    /// A recycled buffer is reused for an output that fits, and its old contents never leak
    /// into the new tensor.
    #[test]
    fn a_recycled_buffer_is_reused_and_fully_overwritten() {
        let mut arena = Arena::default();
        let big = blocked(&Tensor::new(1, 16, 9, 9, ramp(16 * 81, 9)).unwrap());
        let address = big.data.as_ptr();
        arena.recycle(big);

        let input = Tensor::new(1, 5, 4, 6, ramp(5 * 24, 10)).unwrap();
        let (wts, bias) = (ramp(13 * 5 * 9, 11), ramp(13, 12));
        let weights = DenseWeights::new(13, 5, 3, 1, &wts, &bias).unwrap();
        let got = conv(&mut arena, (&input).into(), &weights, 1, 1, false).unwrap();
        assert_eq!(
            got.data.as_ptr(),
            address,
            "the free buffer should have been reused"
        );
        assert!(arena.free.is_empty());
        assert_close(
            &got.to_nchw(0, 13).unwrap(),
            &reference(&input, &wts, &bias, 13, 3, 1, 1, 1, false),
            "conv into a recycled buffer",
        );
    }

    #[test]
    fn global_average_ignores_the_padding_lanes() {
        let a = Tensor::new(2, 11, 3, 4, ramp(2 * 11 * 12, 8)).unwrap();
        let means = blocked(&a).global_average();
        assert_eq!(means.len(), 22);
        for n in 0..2 {
            for c in 0..11 {
                let want = a.plane_slice(n, c).iter().sum::<f32>() / 12.0;
                assert!((means[n * 11 + c] - want).abs() < 1e-5);
            }
        }
    }
}
