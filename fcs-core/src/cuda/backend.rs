use std::{
    path::Path,
    sync::{Arc, Mutex, PoisonError},
};

use anyhow::{Context, Result, ensure};
use cudarc::driver::{CudaGraph, CudaSlice, CudaStream, LaunchConfig, PushKernelArg, sys};
use image::RgbImage;

use crate::{
    onnx::OnnxInitializerMap,
    scrfd::{
        INPUT_SIZE,
        plan::{Step, siblings},
        topology::{OUTPUTS, STEPS},
    },
};

use fcs_utils::cuda::runtime::Runtime;

/// Weights are shared read-only; each inference borrows an [`Executor`] of its own.
#[derive(Debug)]
pub(crate) struct CudaDetector {
    runtime: Arc<Runtime>,
    weights: Vec<Option<(CudaSlice<f32>, CudaSlice<f32>)>>,
    shapes: Vec<[usize; 4]>,
    /// How many consecutive steps each step's launch writes: sibling heads run as one wider
    /// convolution on the first of them (n), and the rest launch nothing (0).
    covers: Vec<usize>,
    /// Captured executors not running an inference right now. There are as many as the most
    /// inferences ever run at once, each holding its ~68 MiB of buffers.
    // ponytail: never shrinks; bounded by the batch worker count, so trim only if that grows.
    idle: Mutex<Vec<Executor>>,
}

/// What a CUDA run hands back for decoding.
#[derive(Debug)]
pub(crate) enum Heads {
    /// Anchors whose score logit passed the device's cutoff, by row id in the CPU decode's
    /// order, each with its nine raw values (`fcs_candidates` in `tools/cuda/kernels.rs`).
    Candidates(Vec<(u32, [f32; 9])>),
    /// The nine head maps, when no threshold was given or the shortlist overflowed.
    Maps(Vec<(Vec<f32>, [usize; 4])>),
}

/// Shortlist slots per inference. A face lights up a handful of anchors at 0.5, so this only
/// overflows -- falling back to reading every map -- at very low thresholds or in vast crowds.
const CANDIDATES: usize = 4096;

/// How far below the threshold's logit the device still shortlists. The host re-applies the
/// exact threshold; the margin only has to cover the sigmoid's rounding near the boundary.
const CUTOFF_MARGIN: f32 = 1e-3;

/// One inference's launches captured as a CUDA graph, and the buffers they point into. Replaying
/// the graph replaces ~60 kernel launches from the host with one.
struct Executor {
    stream: Arc<CudaStream>,
    graph: CudaGraph,
    /// The 8-bit letterbox, room for a full square, and `[width, height, cutoff logit bits]`.
    rgb: CudaSlice<u8>,
    params: CudaSlice<u32>,
    /// The shortlist: how many anchors passed, their row ids and their raw values.
    count: CudaSlice<u32>,
    rows: CudaSlice<u32>,
    values: CudaSlice<f32>,
    /// Slot 0 is the network input, and the split-K scratch follows the last slot. The graph's
    /// launches hold these pointers, so they live exactly as long as it does.
    buffers: Vec<CudaSlice<f32>>,
    /// Per slot: which buffer holds it, and where.
    views: Vec<(usize, usize, usize)>,
}

// SAFETY: a CUDA graph must not be used from two threads at once, and an executor never is:
// it belongs to one inference at a time and moves between threads only through `idle`.
unsafe impl Send for Executor {}

impl std::fmt::Debug for Executor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Executor").finish_non_exhaustive()
    }
}

/// Input channels per thread in split-K convolutions. Each detection head (64 inputs) then
/// runs four threads per output, which beat 4, 8 and 32 on an RTX 4090 (2026-10-21) and asks
/// least of smaller GPUs; layers with fewer than two chunks of input keep the plain kernel.
const SPLIT_CHUNK: usize = 16;

impl CudaDetector {
    pub(crate) fn load(path: &Path) -> Result<Self> {
        let names: Vec<_> = STEPS
            .iter()
            .flat_map(|step| match step {
                Step::Conv { weight, bias, .. } => vec![*weight, *bias],
                _ => Vec::new(),
            })
            .filter(|name| !name.is_empty())
            .collect();
        let map = OnnxInitializerMap::load(path, &names)?;
        let runtime = Runtime::new()?;
        let stream = runtime.context.default_stream();
        let mut host = Vec::with_capacity(STEPS.len());
        let mut shapes = vec![[1, 3, INPUT_SIZE as usize, INPUT_SIZE as usize]];
        for (index, step) in STEPS.iter().enumerate() {
            let (shape, conv) = match *step {
                Step::Conv {
                    input,
                    weight,
                    bias,
                    out_channels,
                    in_per_group,
                    kernel,
                    stride,
                    pad,
                    groups,
                    ..
                } => {
                    let [_, channels, height, width] =
                        *shapes.get(input).context("invalid CUDA topology input")?;
                    ensure!(
                        channels == in_per_group * groups && out_channels % groups == 0,
                        "step {index}: invalid convolution channels"
                    );
                    let shape = [
                        1,
                        out_channels,
                        (height + 2 * pad - kernel) / stride + 1,
                        (width + 2 * pad - kernel) / stride + 1,
                    ];
                    let data = map.tensor(weight)?.data();
                    ensure!(
                        data.len() == out_channels * in_per_group * kernel * kernel,
                        "step {index}: invalid weights"
                    );
                    let biases = if bias.is_empty() {
                        vec![0.0; out_channels]
                    } else {
                        map.tensor(bias)?.data().to_vec()
                    };
                    ensure!(biases.len() == out_channels, "step {index}: invalid bias");
                    (shape, Some((data.to_vec(), biases)))
                }
                Step::Add { a, b } => {
                    ensure!(shapes[a] == shapes[b], "step {index}: add shapes differ");
                    (shapes[a], None)
                }
                Step::Upsample2x { input } => {
                    let [n, c, h, w] = shapes[input];
                    ([n, c, h * 2, w * 2], None)
                }
            };
            ensure!(
                shape.iter().product::<usize>() <= u32::MAX as usize,
                "CUDA tensor is too large"
            );
            shapes.push(shape);
            host.push(conv);
        }
        // Merge each run of siblings into its first step, as the CPU plan does: one launch
        // instead of three, with three times the threads for the narrow heads.
        let mut covers = vec![1; STEPS.len()];
        let mut at = 0;
        while at < STEPS.len() {
            let count = siblings(&STEPS, at);
            for follower in at + 1..at + count {
                let (data, biases) = host[follower].take().context("sibling without weights")?;
                let lead = host[at].as_mut().context("sibling without weights")?;
                lead.0.extend(data);
                lead.1.extend(biases);
                covers[follower] = 0;
            }
            covers[at] = count;
            at += count;
        }
        // Merged slots are views into the lead's buffer, read back only as outputs.
        let whole = |slot: usize| slot == 0 || covers[slot - 1] == 1;
        for (index, step) in STEPS.iter().enumerate() {
            let reads = match *step {
                Step::Conv { input, .. } | Step::Upsample2x { input } => whole(input),
                Step::Add { a, b } => whole(a) && whole(b),
            };
            ensure!(reads, "step {index} reads part of a merged convolution");
        }
        let weights = host
            .into_iter()
            .map(|conv| {
                conv.map(|(data, biases)| {
                    Ok((stream.clone_htod(&data)?, stream.clone_htod(&biases)?))
                })
                .transpose()
            })
            .collect::<Result<_>>()?;
        stream.synchronize()?;
        Ok(Self {
            runtime,
            weights,
            shapes,
            covers,
            idle: Mutex::default(),
        })
    }

    /// Run the network on an 8-bit letterbox (`scrfd::letterbox`), normalising it on the device.
    /// With a score threshold the device shortlists the anchors that may pass it; without one,
    /// or when the shortlist overflows, every head map comes back.
    pub(crate) fn infer(&self, letterbox: &RgbImage, threshold: Option<f32>) -> Result<Heads> {
        let (width, height) = letterbox.dimensions();
        ensure!(
            width > 0 && height > 0 && width <= INPUT_SIZE && height <= INPUT_SIZE,
            "invalid CUDA letterbox {width}x{height}"
        );
        let idle = self
            .idle
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .pop();
        let mut executor = match idle {
            Some(executor) => executor,
            None => self.capture()?,
        };
        let stream = executor.stream.clone();
        let bytes = letterbox.as_raw();
        stream.memcpy_htod(bytes, &mut executor.rgb.slice_mut(0..bytes.len()))?;
        let cutoff = match threshold {
            // Nothing is shortlisted, so the counter stays at zero.
            None => f32::INFINITY,
            Some(t) if t > 0.0 && t < 1.0 => {
                ((t as f64 / (1.0 - t as f64)).ln() as f32) - CUTOFF_MARGIN
            }
            // Every anchor qualifies: the shortlist overflows and the maps come back.
            Some(_) => f32::NEG_INFINITY,
        };
        stream.memcpy_htod(&[width, height, cutoff.to_bits()], &mut executor.params)?;
        executor.graph.launch()?;
        let shortlisted = if threshold.is_some() {
            Some(stream.clone_dtoh(&executor.count)?[0] as usize).filter(|&n| n <= CANDIDATES)
        } else {
            None
        };
        let heads = match shortlisted {
            Some(0) => Heads::Candidates(Vec::new()),
            Some(n) => {
                let rows = stream.clone_dtoh(&executor.rows.slice(0..n))?;
                let values = stream.clone_dtoh(&executor.values.slice(0..n * 9))?;
                let mut candidates: Vec<(u32, [f32; 9])> = rows
                    .into_iter()
                    .zip(values.as_chunks::<9>().0)
                    .map(|(row, values)| (row, *values))
                    .collect();
                // The device fills slots in whatever order threads arrive.
                candidates.sort_unstable_by_key(|&(row, _)| row);
                Heads::Candidates(candidates)
            }
            None => Heads::Maps(
                OUTPUTS
                    .iter()
                    .map(|&at| {
                        let (buffer, offset, n) = executor.views[at];
                        let values = stream
                            .clone_dtoh(&executor.buffers[buffer].slice(offset..offset + n))?;
                        Ok((values, self.shapes[at]))
                    })
                    .collect::<Result<Vec<_>>>()?,
            ),
        };
        // An executor whose run failed is dropped above rather than trusted again.
        self.idle
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .push(executor);
        Ok(heads)
    }

    /// Allocate an inference's buffers, then record its launches into a graph.
    fn capture(&self) -> Result<Executor> {
        let stream = self.runtime.context.new_stream()?;
        let side = INPUT_SIZE as usize;
        let rgb = stream.alloc_zeros::<u8>(side * side * 3)?;
        let params = stream.alloc_zeros::<u32>(3)?;
        let mut count = stream.alloc_zeros::<u32>(1)?;
        let mut rows = stream.alloc_zeros::<u32>(CANDIDATES)?;
        let mut values = stream.alloc_zeros::<f32>(CANDIDATES * 9)?;
        let mut buffers = vec![stream.alloc_zeros::<f32>(self.shapes[0].iter().product())?];
        let mut views = vec![(0, 0, buffers[0].len())];
        let mut scratch = Vec::with_capacity(STEPS.len());
        for (index, step) in STEPS.iter().enumerate() {
            let mut len = 0;
            for slot in index + 1..index + 1 + self.covers[index] {
                let n = self.shapes[slot].iter().product::<usize>();
                views.push((buffers.len(), len, n));
                len += n;
            }
            scratch.push(match *step {
                Step::Conv {
                    groups: 1,
                    kernel,
                    in_per_group,
                    ..
                } if len > 0 && kernel > 1 && in_per_group >= 2 * SPLIT_CHUNK => {
                    Some(stream.alloc_zeros::<f32>(len * in_per_group.div_ceil(SPLIT_CHUNK))?)
                }
                _ => None,
            });
            if len > 0 {
                buffers.push(stream.alloc_zeros::<f32>(len)?);
            }
        }
        // Allocations must be complete, not merely queued, before a capture refers to them.
        stream.synchronize()?;
        stream.begin_capture(sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)?;
        let enqueued = self
            .normalize(&stream, &rgb, &params, &mut buffers[0])
            .and_then(|()| self.enqueue(&stream, &mut buffers, &mut scratch, &views))
            .and_then(|()| {
                let shortlist = (&mut count, &mut rows, &mut values);
                self.shortlist(&stream, &buffers, &views, &params, shortlist)
            });
        // Always end the capture, even after a failed launch, so the stream is usable again.
        let graph = stream.end_capture(sys::CUgraphInstantiate_flags(0));
        enqueued?;
        let graph = graph?.context("CUDA graph capture recorded nothing")?;
        buffers.extend(scratch.into_iter().flatten());
        Ok(Executor {
            stream,
            graph,
            rgb,
            params,
            count,
            rows,
            values,
            buffers,
            views,
        })
    }

    /// Zero the counter, then shortlist each level's anchors from its merged head buffer.
    fn shortlist(
        &self,
        stream: &Arc<CudaStream>,
        buffers: &[CudaSlice<f32>],
        views: &[(usize, usize, usize)],
        params: &CudaSlice<u32>,
        (count, rows, values): (
            &mut CudaSlice<u32>,
            &mut CudaSlice<u32>,
            &mut CudaSlice<f32>,
        ),
    ) -> Result<()> {
        stream.memset_zeros(count)?;
        let mut base = 0u32;
        for level in 0..3 {
            let [score, boxes, points] = [level, level + 3, level + 6].map(|i| OUTPUTS[i]);
            let [_, _, cells, _] = self.shapes[score];
            let plane = (cells * cells) as u32;
            for (slot, channels) in [(score, 2), (boxes, 8), (points, 20)] {
                ensure!(
                    self.shapes[slot] == [1, channels, cells, cells],
                    "level {level} head shape {:?}",
                    self.shapes[slot]
                );
            }
            let buffer = views[score].0;
            ensure!(
                views[boxes].0 == buffer && views[points].0 == buffer,
                "level {level} heads are not merged into one buffer"
            );
            // SAFETY: the three offsets index the level's merged buffer at the shapes checked
            // above; the shortlist buffers hold CANDIDATES rows of one id and nine values.
            unsafe {
                stream
                    .launch_builder(&self.runtime.candidates)
                    .arg(&buffers[buffer])
                    .arg(params)
                    .arg(&mut *count)
                    .arg(&mut *rows)
                    .arg(&mut *values)
                    .arg(&plane)
                    .arg(&(views[score].1 as u32))
                    .arg(&(views[boxes].1 as u32))
                    .arg(&(views[points].1 as u32))
                    .arg(&base)
                    .arg(&(CANDIDATES as u32))
                    .launch(LaunchConfig {
                        grid_dim: ((plane * 2).div_ceil(128), 1, 1),
                        block_dim: (128, 1, 1),
                        shared_mem_bytes: 0,
                    })?;
            }
            base += plane * 2;
        }
        Ok(())
    }

    fn normalize(
        &self,
        stream: &Arc<CudaStream>,
        rgb: &CudaSlice<u8>,
        dims: &CudaSlice<u32>,
        input: &mut CudaSlice<f32>,
    ) -> Result<()> {
        let len = input.len() as u32;
        // SAFETY: `rgb` holds a full square and `input` is exactly the 3 x size x size tensor.
        unsafe {
            stream
                .launch_builder(&self.runtime.normalize)
                .arg(rgb)
                .arg(dims)
                .arg(input)
                .arg(&INPUT_SIZE)
                .launch(LaunchConfig {
                    grid_dim: (len.div_ceil(128), 1, 1),
                    block_dim: (128, 1, 1),
                    shared_mem_bytes: 0,
                })?;
        }
        Ok(())
    }

    fn enqueue(
        &self,
        stream: &Arc<CudaStream>,
        buffers: &mut [CudaSlice<f32>],
        scratch: &mut [Option<CudaSlice<f32>>],
        views: &[(usize, usize, usize)],
    ) -> Result<()> {
        for (index, step) in STEPS.iter().enumerate() {
            let covers = self.covers[index];
            if covers == 0 {
                continue;
            }
            let shape = self.shapes[index + 1];
            let written = index + 1..index + 1 + covers;
            let len = written
                .clone()
                .map(|slot| self.shapes[slot].iter().product::<usize>())
                .sum::<usize>() as u32;
            let channels: usize = written.map(|slot| self.shapes[slot][1]).sum();
            let config = LaunchConfig {
                grid_dim: (len.div_ceil(128), 1, 1),
                block_dim: (128, 1, 1),
                shared_mem_bytes: 0,
            };
            // Every step writes a buffer after all the ones it reads.
            let (before, after) = buffers.split_at_mut(views[index + 1].0);
            let output = &mut after[0];
            // SAFETY: load validates the fixed topology and capture allocated every buffer at its
            // length; outputs are disjoint from inputs, and every operation uses this stream.
            unsafe {
                match *step {
                    Step::Conv {
                        input,
                        in_per_group,
                        out_channels,
                        kernel,
                        stride,
                        pad,
                        groups,
                        relu,
                        ..
                    } => {
                        let [_, _, h, w] = self.shapes[input];
                        let params = [
                            h as u32,
                            w as u32,
                            shape[2] as u32,
                            shape[3] as u32,
                            in_per_group as u32,
                            channels as u32,
                            kernel as u32,
                            stride as u32,
                            pad as u32,
                            groups as u32,
                            u32::from(relu),
                        ];
                        let (weights, bias) = self.weights[index]
                            .as_ref()
                            .context("missing CUDA weights")?;
                        if groups == 1 && kernel > 1 && in_per_group >= 2 * SPLIT_CHUNK {
                            let chunks = in_per_group.div_ceil(SPLIT_CHUNK);
                            let total = u32::try_from(len as usize * chunks)?;
                            let partial =
                                scratch[index].as_mut().context("missing CUDA scratch")?;
                            stream
                                .launch_builder(&self.runtime.conv_partial)
                                .arg(&before[views[input].0])
                                .arg(weights)
                                .arg(&mut *partial)
                                .arg(&(h as u32))
                                .arg(&(w as u32))
                                .arg(&(shape[2] as u32))
                                .arg(&(shape[3] as u32))
                                .arg(&(in_per_group as u32))
                                .arg(&(channels as u32))
                                .arg(&(kernel as u32))
                                .arg(&(stride as u32))
                                .arg(&(pad as u32))
                                .arg(&(SPLIT_CHUNK as u32))
                                .launch(LaunchConfig {
                                    grid_dim: (total.div_ceil(128), 1, 1),
                                    block_dim: (128, 1, 1),
                                    shared_mem_bytes: 0,
                                })?;
                            stream
                                .launch_builder(&self.runtime.conv_reduce)
                                .arg(&*partial)
                                .arg(bias)
                                .arg(&mut *output)
                                .arg(&len)
                                .arg(&((shape[2] * shape[3]) as u32))
                                .arg(&(chunks as u32))
                                .arg(&u32::from(relu))
                                .launch(config)?;
                        } else {
                            let function = if kernel == 1 && groups == 1 {
                                &self.runtime.pointwise
                            } else if in_per_group == 1 && groups == out_channels {
                                &self.runtime.depthwise
                            } else {
                                &self.runtime.conv
                            };
                            let mut launch = stream.launch_builder(function);
                            launch
                                .arg(&before[views[input].0])
                                .arg(weights)
                                .arg(bias)
                                .arg(&mut *output);
                            for param in &params {
                                launch.arg(param);
                            }
                            launch.launch(config)?;
                        }
                    }
                    Step::Add { a, b } => {
                        stream
                            .launch_builder(&self.runtime.add)
                            .arg(&before[views[a].0])
                            .arg(&before[views[b].0])
                            .arg(&mut *output)
                            .arg(&len)
                            .launch(config)?;
                    }
                    Step::Upsample2x { input } => {
                        let [_, c, h, w] = self.shapes[input];
                        stream
                            .launch_builder(&self.runtime.upsample)
                            .arg(&before[views[input].0])
                            .arg(&mut *output)
                            .arg(&(c as u32))
                            .arg(&(h as u32))
                            .arg(&(w as u32))
                            .launch(config)?;
                    }
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "requires a compatible NVIDIA GPU and the detector model"]
    fn the_shortlist_decodes_exactly_as_the_maps() -> Result<()> {
        use crate::scrfd::{decode_candidates, decode_maps};
        let model =
            fcs_utils::model_path("models/scrfd80k_500m_640.onnx")?.context("model required")?;
        let cuda = CudaDetector::load(&model)?;
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap();
        let mut images = std::fs::read_dir(root.join("fixtures/images"))?
            .map(|entry| entry.map(|entry| entry.path()))
            .collect::<std::io::Result<Vec<_>>>()?;
        images.retain(|path| fcs_utils::is_supported_image_path(path));
        images.sort();
        let mut faces = 0;
        let mut last = None;
        for path in &images {
            let (letterbox, fit) =
                crate::scrfd::letterbox(&fcs_utils::load_image(path)?, INPUT_SIZE);
            let Heads::Maps(maps) = cuda.infer(&letterbox, None)? else {
                anyhow::bail!("maps were asked for");
            };
            for threshold in [0.2, 0.5, 0.8] {
                let expected = decode_maps(&maps, fit, threshold, INPUT_SIZE)?;
                let Heads::Candidates(candidates) = cuda.infer(&letterbox, Some(threshold))? else {
                    panic!(
                        "{}: the shortlist overflowed at {threshold}",
                        path.display()
                    );
                };
                let actual = decode_candidates(&candidates, fit, threshold, INPUT_SIZE);
                assert_eq!(actual, expected, "{} at {threshold}", path.display());
                faces += expected.len();
            }
            last = Some(letterbox);
        }
        ensure!(faces > 0, "the fixtures produced no detections to compare");
        // At 0 every anchor qualifies, so the shortlist overflows into the full maps.
        let letterbox = last.context("image fixtures required")?;
        assert!(matches!(cuda.infer(&letterbox, Some(0.0))?, Heads::Maps(_)));
        Ok(())
    }

    #[test]
    #[ignore = "requires a compatible NVIDIA GPU; run explicitly before shipping CUDA changes"]
    fn cuda_matches_cpu_and_is_safe_for_concurrent_inference() -> Result<()> {
        let path =
            fcs_utils::model_path("models/scrfd80k_500m_640.onnx")?.context("model required")?;
        let mut cuda = CudaDetector::load(&path)?;
        let cpu = crate::scrfd::plan::ScrfdWeights::load(&path)?;
        let side = INPUT_SIZE as usize;
        // Landscape pads below the image and portrait to its right, as letterboxes do.
        let cases = [(640, 481), (333, 640)]
            .map(|(width, height)| -> Result<_> {
                let letterbox = RgbImage::from_fn(width, height, |x, y| {
                    image::Rgb([(x * 7 + y) as u8, (x + y * 5) as u8, (x * y % 251) as u8])
                });
                let input = crate::scrfd::normalize(&letterbox, INPUT_SIZE);
                let expected = crate::scrfd::plan::run(
                    crate::cpu::tensor::Tensor::new(1, 3, side, side, input)?,
                    &cpu,
                )?;
                Ok((letterbox, expected))
            })
            .into_iter()
            .collect::<Result<Vec<_>>>()?;
        let check = |cuda: &CudaDetector| -> Result<()> {
            for (letterbox, expected) in &cases {
                let Heads::Maps(actual) = cuda.infer(letterbox, None)? else {
                    anyhow::bail!("maps were asked for");
                };
                assert_eq!(actual.len(), expected.len());
                for ((values, shape), expected) in actual.iter().zip(expected) {
                    assert_eq!(*shape, expected.dims());
                    assert_eq!(values.len(), expected.data().len());
                    let error = values
                        .iter()
                        .zip(expected.data())
                        .map(|(a, b)| (a - b).abs())
                        .fold(0.0f32, f32::max);
                    assert!(error < 2e-4, "CUDA/CPU error {error}");
                    assert!(values.iter().all(|x| x.is_finite()));
                }
            }
            Ok(())
        };
        for ptx in [
            include_str!("../../../fcs-utils/src/cuda/windows.ptx"),
            include_str!("../../../fcs-utils/src/cuda/linux.ptx"),
        ] {
            cuda.runtime = Arc::new(Runtime::from_ptx(ptx)?);
            // Executors captured the previous module's kernels.
            cuda.idle.get_mut().unwrap().clear();
            check(&cuda)?;
            std::thread::scope(|scope| -> Result<()> {
                let first = scope.spawn(|| check(&cuda));
                let second = scope.spawn(|| check(&cuda));
                first.join().expect("CUDA worker panic")?;
                second.join().expect("CUDA worker panic")
            })?;
        }
        Ok(())
    }
}
