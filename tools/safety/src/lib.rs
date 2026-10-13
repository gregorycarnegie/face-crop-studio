//! Small Miri scenarios against the production CPU kernels.

#[path = "../../../fcs-core/src/cpu/nchwc.rs"]
pub mod nchwc;
#[path = "../../../fcs-core/src/cpu/tensor.rs"]
pub mod tensor;

#[cfg(test)]
mod tests {
    use super::{nchwc::*, tensor::Tensor};

    #[test]
    fn miri_cpu_buffers_are_initialized_disjoint_and_reusable() {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(2)
            .build()
            .unwrap();
        pool.install(|| {
            let input = Tensor::new(1, 3, 2, 3, (1..=18).map(|v| v as f32).collect()).unwrap();
            for channels in [5, 9, 33] {
                for kernel in [1, 3] {
                    let mut arena = Arena::default();
                    let mut weights = vec![0.0; channels * 3 * kernel * kernel];
                    for c in 0..channels {
                        weights[(c * 3 + c % 3) * kernel * kernel + kernel * kernel / 2] = 1.0;
                    }
                    let bias: Vec<_> = (0..channels).map(|c| c as f32).collect();
                    let weights =
                        DenseWeights::new(channels, 3, kernel, 1, &weights, &bias).unwrap();
                    let expected: Vec<_> = (0..channels)
                        .flat_map(|c| (0..6).map(move |p| (c % 3 * 6 + p + 1 + c) as f32))
                        .collect();
                    let packed =
                        conv(&mut arena, (&input).into(), &weights, 1, kernel / 2, false).unwrap();
                    assert_eq!(packed.to_nchw(0, channels).unwrap().data(), expected);
                    let means: Vec<_> = (0..channels)
                        .map(|c| (c % 3 * 6 + c) as f32 + 3.5)
                        .collect();
                    assert_eq!(packed.global_average(), means);
                    // Includes a range crossing into a partially filled SIMD block.
                    assert_eq!(
                        packed.to_nchw(channels - 2, 2).unwrap().data(),
                        &expected[(channels - 2) * 6..]
                    );

                    let mut identity_weights = vec![0.0; channels * channels];
                    for c in 0..channels {
                        identity_weights[c * channels + c] = 1.0;
                    }
                    let identity_weights = DenseWeights::new(
                        channels,
                        channels,
                        1,
                        BLOCK,
                        &identity_weights,
                        &vec![0.0; channels],
                    )
                    .unwrap();
                    let blocked =
                        conv(&mut arena, (&packed).into(), &identity_weights, 1, 0, false).unwrap();
                    assert_eq!(blocked.to_nchw(0, channels).unwrap().data(), expected);

                    let mut dw = vec![0.0; channels * 9];
                    for c in 0..channels {
                        dw[c * 9 + 4] = 1.0;
                    }
                    let dw = DepthwiseWeights::new(channels, 3, &dw, &vec![0.0; channels]).unwrap();
                    let identity = depthwise(&mut arena, &packed, &dw, 1, 1, false).unwrap();
                    assert_eq!(identity.to_nchw(0, channels).unwrap().data(), expected);
                    let strided = depthwise(&mut arena, &packed, &dw, 2, 1, false).unwrap();
                    let sampled: Vec<_> = (0..channels)
                        .flat_map(|c| [expected[c * 6], expected[c * 6 + 2]])
                        .collect();
                    assert_eq!(strided.to_nchw(0, channels).unwrap().data(), sampled);

                    let sum = add(&mut arena, &packed, &identity).unwrap();
                    let doubled: Vec<_> = expected.iter().map(|v| v * 2.0).collect();
                    assert_eq!(sum.to_nchw(0, channels).unwrap().data(), doubled);
                    let up = upsample2x(&mut arena, &packed).unwrap();
                    let expanded: Vec<_> = (0..channels)
                        .flat_map(|c| {
                            let expected = &expected;
                            (0..24).map(move |p| expected[c * 6 + (p / 6 / 2) * 3 + p % 6 / 2])
                        })
                        .collect();
                    assert_eq!(up.to_nchw(0, channels).unwrap().data(), expanded);
                    arena.recycle(up);
                    let reused =
                        conv(&mut arena, (&input).into(), &weights, 1, kernel / 2, false).unwrap();
                    assert_eq!(reused.to_nchw(0, channels).unwrap().data(), expected);
                }
            }

            // A long row gives the raw-pointer writer separate parallel column spans.
            let input = Tensor::new(1, 1, 1, 97, (0..97).map(|v| v as f32).collect()).unwrap();
            let weights = DenseWeights::new(1, 1, 1, 1, &[1.0], &[0.0]).unwrap();
            let output = conv(
                &mut Arena::default(),
                (&input).into(),
                &weights,
                1,
                0,
                false,
            )
            .unwrap();
            assert_eq!(output.to_nchw(0, 1).unwrap().data(), input.data());
        });
    }
}
