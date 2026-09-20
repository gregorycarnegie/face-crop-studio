//! The non-convolution operators the graph needs: max pooling, elementwise add,
//! and a nearest 2x upsample.
//!
//! Semantics mirror `gpu/pool.wgsl`, `gpu/add.wgsl` and `gpu/resize2x.wgsl`,
//! since the two backends are compared against one another.

use anyhow::Result;
use rayon::prelude::*;

use super::tensor::Tensor;

/// Output extent for a pooling window, matching `gpu::utils::compute_output_dim`.
fn pooled_dim(size: usize, pad: usize, kernel: usize, stride: usize) -> Result<usize> {
    anyhow::ensure!(stride > 0, "stride must be > 0");
    anyhow::ensure!(kernel > 0, "kernel must be > 0");
    let padded = size + 2 * pad;
    anyhow::ensure!(
        padded >= kernel,
        "kernel {kernel} is larger than the padded input {padded}"
    );
    Ok((padded - kernel) / stride + 1)
}

/// Max pooling.
///
/// Padded positions contribute nothing rather than contributing zero — the
/// window is simply clipped. That matters for negative activations, where a
/// zero would win the max and silently brighten the border.
pub fn max_pool(input: &Tensor, kernel: usize, stride: usize, pad: usize) -> Result<Tensor> {
    let (n, c, h, w) = (
        input.batch(),
        input.channels(),
        input.height(),
        input.width(),
    );
    let out_h = pooled_dim(h, pad, kernel, stride)?;
    let out_w = pooled_dim(w, pad, kernel, stride)?;
    let mut output = Tensor::zeros(n, c, out_h, out_w);
    let src = input.data();

    output
        .data_mut()
        .par_chunks_mut(out_h * out_w)
        .enumerate()
        .for_each(|(idx, out_plane)| {
            let plane = &src[idx * h * w..(idx + 1) * h * w];
            for oy in 0..out_h {
                for ox in 0..out_w {
                    let iy0 = (oy * stride) as isize - pad as isize;
                    let ix0 = (ox * stride) as isize - pad as isize;
                    let mut best = f32::NEG_INFINITY;
                    for ky in 0..kernel {
                        let iy = iy0 + ky as isize;
                        if iy < 0 || iy >= h as isize {
                            continue;
                        }
                        let row = iy as usize * w;
                        for kx in 0..kernel {
                            let ix = ix0 + kx as isize;
                            if ix < 0 || ix >= w as isize {
                                continue;
                            }
                            best = best.max(plane[row + ix as usize]);
                        }
                    }
                    out_plane[oy * out_w + ox] = best;
                }
            }
        });

    Ok(output)
}

/// Elementwise sum of two identically shaped tensors.
pub fn add(lhs: &Tensor, rhs: &Tensor) -> Result<Tensor> {
    anyhow::ensure!(
        lhs.dims() == rhs.dims(),
        "add expects matching shapes, got {:?} and {:?}",
        lhs.dims(),
        rhs.dims()
    );
    let mut out = lhs.clone();
    out.data_mut()
        .par_iter_mut()
        .zip(rhs.data().par_iter())
        .for_each(|(o, r)| *o += r);
    Ok(out)
}

/// Nearest-neighbour 2x upsample: every input pixel becomes a 2x2 block.
pub fn resize2x(input: &Tensor) -> Result<Tensor> {
    let (n, c, h, w) = (
        input.batch(),
        input.channels(),
        input.height(),
        input.width(),
    );
    let (out_h, out_w) = (h * 2, w * 2);
    let mut output = Tensor::zeros(n, c, out_h, out_w);
    let src = input.data();

    output
        .data_mut()
        .par_chunks_mut(out_h * out_w)
        .enumerate()
        .for_each(|(idx, out_plane)| {
            let plane = &src[idx * h * w..(idx + 1) * h * w];
            for oy in 0..out_h {
                let in_row = &plane[(oy >> 1) * w..(oy >> 1) * w + w];
                let out_row = &mut out_plane[oy * out_w..(oy + 1) * out_w];
                for (ox, o) in out_row.iter_mut().enumerate() {
                    *o = in_row[ox >> 1];
                }
            }
        });

    Ok(output)
}

/// Logistic sigmoid, applied to the classification and objectness heads.
#[inline]
pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tensor(c: usize, h: usize, w: usize, data: Vec<f32>) -> Tensor {
        Tensor::new(1, c, h, w, data).expect("valid tensor")
    }

    #[test]
    fn max_pool_2x2_halves_each_dimension() {
        // 4x4 counting up; each 2x2 window's max is its bottom-right corner.
        let data: Vec<f32> = (0..16).map(|i| i as f32).collect();
        let out = max_pool(&tensor(1, 4, 4, data), 2, 2, 0).expect("pool");
        assert_eq!(out.dims(), [1, 1, 2, 2]);
        assert_eq!(out.data(), &[5.0, 7.0, 13.0, 15.0]);
    }

    /// With an odd extent the floor formula drops the ragged edge, matching the
    /// GPU path. Pinning it stops a later "fix" to ceil mode passing silently.
    #[test]
    fn max_pool_floors_an_odd_input() {
        let data: Vec<f32> = (0..9).map(|i| i as f32).collect();
        let out = max_pool(&tensor(1, 3, 3, data), 2, 2, 0).expect("pool");
        assert_eq!(out.dims(), [1, 1, 1, 1]);
        assert_eq!(out.data(), &[4.0]);
    }

    /// Padding must clip the window, not contribute zeros — otherwise an
    /// all-negative plane pools to 0 instead of its true maximum.
    #[test]
    fn padding_does_not_introduce_zeros_into_the_max() {
        let out = max_pool(&tensor(1, 2, 2, vec![-4.0, -3.0, -2.0, -1.0]), 2, 2, 1).expect("pool");
        assert_eq!(out.dims(), [1, 1, 2, 2]);
        // Corners see one real value each; none may become 0.
        assert_eq!(out.data(), &[-4.0, -3.0, -2.0, -1.0]);
    }

    #[test]
    fn pool_rejects_a_kernel_larger_than_the_input() {
        let err = max_pool(&tensor(1, 1, 1, vec![1.0]), 3, 1, 0).expect_err("3 > 1");
        assert!(format!("{err}").contains("larger than"), "{err}");
    }

    #[test]
    fn max_pool_keeps_channels_independent() {
        // Channel 1 is uniformly larger; neither may leak into the other.
        let data = vec![1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0];
        let out = max_pool(&tensor(2, 2, 2, data), 2, 2, 0).expect("pool");
        assert_eq!(out.data(), &[4.0, 40.0]);
    }

    #[test]
    fn add_sums_elementwise() {
        let a = tensor(1, 2, 2, vec![1.0, 2.0, 3.0, 4.0]);
        let b = tensor(1, 2, 2, vec![10.0, 20.0, 30.0, 40.0]);
        assert_eq!(add(&a, &b).expect("add").data(), &[11.0, 22.0, 33.0, 44.0]);
    }

    #[test]
    fn add_rejects_mismatched_shapes() {
        let a = tensor(1, 2, 2, vec![0.0; 4]);
        let b = tensor(1, 1, 4, vec![0.0; 4]);
        let err = add(&a, &b).expect_err("shapes differ");
        assert!(format!("{err}").contains("matching shapes"), "{err}");
    }

    #[test]
    fn resize2x_replicates_each_pixel_into_a_block() {
        let out = resize2x(&tensor(1, 1, 2, vec![1.0, 2.0])).expect("resize");
        assert_eq!(out.dims(), [1, 1, 2, 4]);
        assert_eq!(out.data(), &[1.0, 1.0, 2.0, 2.0, 1.0, 1.0, 2.0, 2.0]);
    }

    #[test]
    fn resize2x_keeps_channels_independent() {
        let out = resize2x(&tensor(2, 1, 1, vec![5.0, 9.0])).expect("resize");
        assert_eq!(out.dims(), [1, 2, 2, 2]);
        assert_eq!(out.data(), &[5.0, 5.0, 5.0, 5.0, 9.0, 9.0, 9.0, 9.0]);
    }

    #[test]
    fn sigmoid_matches_known_values() {
        assert!((sigmoid(0.0) - 0.5).abs() < 1e-7);
        assert!(sigmoid(-100.0) < 1e-6);
        assert!(sigmoid(100.0) > 1.0 - 1e-6);
    }
}
