//! SCRFD: the detector trained on this project's own licence-clean data.
//!
//! It finds 86.0% of the faces in the Open Images test split at 0.11 false positives per image,
//! against YuNet's 71.6% at 0.14, and on a corpus neither model had seen it found 135 faces
//! YuNet missed while missing 13 it found (`tools/dataset/SCRFD_80K.md`). It cost about 10%
//! more per image: 6.4 ms against 5.8 at 640x640 on ONNX Runtime.
//!
//! **It is the only detector.** It runs on the WGSL kernels (`gpu`) or the built-in CPU
//! graph (`plan`). Both agree with the test-only ONNX Runtime oracle to about 1e-05 and produce
//! identical detections, so there is nothing a machine can be missing that leaves it without a
//! detector. That is what let YuNet go: its weights come from WIDER FACE, "non-commercial
//! academic research only". [`ScrfdDetector::load`] still returns `None` when the model file
//! itself is absent, and then detection cannot run at all.
//!
//! The graph is exported by `tools/dataset/export_scrfd.py` and emits nine raw tensors, three
//! per stride: class scores already sigmoided, box distances, and keypoint distances, both in
//! units of the stride. No NMS inside the graph, so the decoding below is the whole of it.

/// Running the network on the WGSL engine.
pub mod gpu;
/// Running the network without ONNX Runtime, on the built-in CPU graph.
pub mod plan;
/// The generated step table; see `tools/dataset/scrfd_topology.py`.
pub mod topology;

use std::path::Path;

use anyhow::Result;
use image::DynamicImage;
use log::{debug, info, warn};

use crate::{
    nms::apply_nms_in_place,
    postprocess::{BoundingBox, Detection, Landmark},
};

/// The exported graph's fixed input side.
pub const INPUT_SIZE: u32 = 640;

/// Strides the head predicts at, in the order their tensors are matched.
const STRIDES: [u32; 3] = [8, 16, 32];

/// Anchors per cell, from the config's two scales at one ratio.
const ANCHORS_PER_CELL: usize = 2;

/// Landmarks the head predicts: eyes, nose, then the two mouth corners.
const LANDMARKS: usize = 5;

/// How many of those five this model was actually taught: the two eyes, and nothing else.
/// See the note in `decode_level` on why the rest are reported absent rather than passed through.
const TRAINED_LANDMARKS: usize = 2;

/// Workspace-relative location of the exported model.
pub(crate) const DEFAULT_MODEL: &str = "models/scrfd80k_500m_640.onnx";

/// A loaded SCRFD detector, on whichever engine is available.
#[derive(Debug)]
pub struct ScrfdDetector {
    backend: Backend,
}

/// Where the network actually runs: whichever of the two measured faster at load (see
/// [`prefer_gpu`]). Both were checked against ONNX Runtime (used only in tests and
/// benchmarks) on the same input: 1.1e-05 between GPU and ONNX Runtime, 1.2e-05 between CPU
/// and ONNX Runtime.
#[derive(Debug)]
enum Backend {
    Gpu(Box<(crate::gpu::GpuInferenceOps, gpu::ScrfdGpuWeights)>),
    Cpu(Box<plan::ScrfdWeights>),
}

/// How a source image was laid onto the square input, and what undoes it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Letterbox {
    /// Factor the source was multiplied by; detections are divided by it to come back.
    pub scale: f32,
}

impl ScrfdDetector {
    /// Load from the default location, or `None` when it cannot run.
    ///
    /// Returns `None` if the model is missing or cannot load on either built-in engine.
    pub fn load() -> Option<Self> {
        Self::load_from(fcs_utils::resolve_data_path(DEFAULT_MODEL))
    }

    /// Load from an explicit path. See [`Self::load`] for the `None` cases.
    ///
    /// With a GPU, both engines are loaded and timed on one input, and the slower is dropped.
    /// Neither wins everywhere: on an RTX 4090 the WGSL engine takes 2 ms to the CPU graph's
    /// 5.5, and on an integrated GPU it takes 22 ms to the same CPU's 7.5. The probe costs a
    /// few detections' time at load. Without a GPU, or when it fails, the CPU graph is the
    /// floor.
    pub fn load_from<P: AsRef<Path>>(path: P) -> Option<Self> {
        let path = path.as_ref();
        if !path.exists() {
            debug!("SCRFD not loaded: no model at {}", path.display());
            return None;
        }

        let cpu = match plan::ScrfdWeights::load(path) {
            Ok(weights) => Self {
                backend: Backend::Cpu(Box::new(weights)),
            },
            Err(err) => {
                warn!("SCRFD at {} would not load ({err})", path.display());
                return None;
            }
        };
        let gpu = match fcs_utils::GpuContext::init_with_fallback(&Default::default()) {
            fcs_utils::GpuAvailability::Available(context) => {
                match crate::gpu::GpuInferenceOps::new(context, None)
                    .and_then(|ops| Ok((gpu::ScrfdGpuWeights::load(&ops, path)?, ops)))
                {
                    Ok((weights, ops)) => Self {
                        backend: Backend::Gpu(Box::new((ops, weights))),
                    },
                    Err(err) => {
                        warn!("SCRFD on the GPU failed ({err}); using the CPU graph");
                        return Some(cpu);
                    }
                }
            }
            other => {
                debug!("no GPU for SCRFD ({other:?}); using the CPU graph");
                return Some(cpu);
            }
        };

        let chosen = match (gpu.probe_ms(), cpu.probe_ms()) {
            (Ok(gpu_ms), Ok(cpu_ms)) => {
                let use_gpu = prefer_gpu(gpu_ms, cpu_ms);
                info!(
                    "SCRFD timed at load: wgsl-gpu {gpu_ms:.1} ms, cpu-graph {cpu_ms:.1} ms; using {}",
                    if use_gpu { "wgsl-gpu" } else { "cpu-graph" }
                );
                if use_gpu { gpu } else { cpu }
            }
            (Err(err), _) => {
                warn!("SCRFD on the GPU failed its first run ({err}); using the CPU graph");
                cpu
            }
            // The CPU graph failing where the GPU ran is not a reason to give up the GPU.
            (Ok(_), Err(err)) => {
                warn!("SCRFD's CPU graph failed its first run ({err}); using the GPU");
                gpu
            }
        };
        info!("SCRFD on {} from {}", chosen.engine(), path.display());
        Some(chosen)
    }

    /// Median time of the network alone, in milliseconds, after untimed runs that absorb
    /// first-use costs (buffer allocation, the GPU's lazy pipeline state). One warm-up was not
    /// enough: an RTX 4090 still read 4.1 ms against its steady 2.
    fn probe_ms(&self) -> Result<f64> {
        const WARMUP: usize = 3;
        const RUNS: usize = 3;
        let side = INPUT_SIZE as usize;
        let input = vec![0.0f32; 3 * side * side];
        for _ in 0..WARMUP {
            self.infer(input.clone())?;
        }
        let mut times = (0..RUNS)
            .map(|_| {
                let started = std::time::Instant::now();
                self.infer(input.clone())?;
                Ok(started.elapsed().as_secs_f64() * 1e3)
            })
            .collect::<Result<Vec<_>>>()?;
        times.sort_by(f64::total_cmp);
        Ok(times[RUNS / 2])
    }

    /// Run the network on a preprocessed input and read back the nine head maps.
    fn infer(&self, input: Vec<f32>) -> Result<Vec<(Vec<f32>, [usize; 4])>> {
        let side = INPUT_SIZE as usize;
        match &self.backend {
            Backend::Gpu(boxed) => {
                let (ops, weights) = boxed.as_ref();
                let uploaded =
                    ops.upload_tensor(vec![1, 3, side, side], &input, Some("scrfd input"))?;
                gpu::run(ops, &uploaded, weights)?
                    .iter()
                    .map(|tensor| {
                        let dims = tensor.shape().dims();
                        Ok((tensor.to_vec()?, [dims[0], dims[1], dims[2], dims[3]]))
                    })
                    .collect()
            }
            Backend::Cpu(weights) => {
                let tensor = crate::cpu::tensor::Tensor::new(1, 3, side, side, input)?;
                Ok(plan::run(tensor, weights)?
                    .into_iter()
                    .map(|t| {
                        let dims = [t.batch(), t.channels(), t.height(), t.width()];
                        (t.into_data(), dims)
                    })
                    .collect())
            }
        }
    }

    /// Which engine is running, for logs.
    pub fn engine(&self) -> &'static str {
        match self.backend {
            Backend::Gpu(_) => "wgsl-gpu",
            Backend::Cpu(_) => "cpu-graph",
        }
    }

    /// Detect faces, in the source image's own pixel coordinates.
    ///
    /// `score_threshold` is the operating point: 0.5 was chosen by eye on a corpus neither this
    /// model nor YuNet had seen, where 0.4 kept 32 false positives among the 60 largest
    /// disagreements and 0.5 kept 7 of them while losing 2 of 25 real faces (SCRFD_80K.md).
    pub fn detect(
        &self,
        image: &DynamicImage,
        score_threshold: f32,
        nms_threshold: f32,
    ) -> Result<Vec<Detection>> {
        let (input, letterbox) = preprocess(image, INPUT_SIZE);
        let mut detections =
            decode_maps(&self.infer(input)?, letterbox, score_threshold, INPUT_SIZE)?;
        detections.sort_by(|a, b| b.score.total_cmp(&a.score));
        apply_nms_in_place(&mut detections, nms_threshold);
        Ok(detections)
    }
}

/// Whether the GPU should run the network, given each engine's time at load.
///
/// The GPU keeps the job unless the CPU graph is more than 25% faster. The probe runs on idle
/// cores, but in a batch those cores are decoding and cropping other images, and GPU
/// inference leaves them free for it; a near tie measured idle is a GPU win under load.
fn prefer_gpu(gpu_ms: f64, cpu_ms: f64) -> bool {
    cpu_ms * 1.25 >= gpu_ms
}

/// Resize into the top-left of a square canvas and normalise, as the model was trained.
///
/// Three details that are all load-bearing, and all silent when wrong: the source goes to the
/// **top-left** rather than being centred the way YuNet's preprocessing letterboxed; channels
/// are **RGB**, not the BGR it produced; and the padding is normalised along with everything
/// else. The Python builds a zeroed `uint8` canvas and then subtracts, so the padding is
/// `(0 - 127.5) / 128`, not zero, and a model fed zeroed padding sees a border it never met.
pub fn preprocess(image: &DynamicImage, size: u32) -> (Vec<f32>, Letterbox) {
    let (width, height) = (image.width().max(1), image.height().max(1));
    let ratio = height as f32 / width as f32;
    let (new_width, new_height) = if ratio > 1.0 {
        (((size as f32 / ratio) as u32).max(1), size)
    } else {
        (size, ((size as f32 * ratio) as u32).max(1))
    };
    let scale = new_height as f32 / height as f32;

    let resized = fcs_utils::resize_image(
        image,
        new_width,
        new_height,
        image::imageops::FilterType::Triangle,
    );

    let plane = (size * size) as usize;
    // The padding value, not 0.0: see the note above.
    let mut tensor = vec![(0.0 - 127.5) / 128.0; 3 * plane];
    for y in 0..new_height.min(size) {
        for x in 0..new_width.min(size) {
            let pixel = resized.get_pixel(x, y);
            let index = (y * size + x) as usize;
            for channel in 0..3 {
                tensor[channel * plane + index] = (pixel[channel] as f32 - 127.5) / 128.0;
            }
        }
    }
    (tensor, Letterbox { scale })
}

/// Turn the nine raw tensors into detections in source-image coordinates.
///
/// Outputs are matched by shape rather than by position: the last dimension says which head a
/// tensor came from (1 = score, 4 = box distance, 10 = keypoint distance) and the row count says
/// which stride (640/8 squared times two anchors = 12,800, then 3,200, then 800). Matching by
/// index would work today and break silently the first time the exporter reorders anything.
#[cfg(test)]
fn decode(
    outputs: &[fcs_ort::OutputTensor],
    letterbox: Letterbox,
    score_threshold: f32,
    size: u32,
) -> Result<Vec<Detection>> {
    let mut detections = Vec::new();

    for (level, stride) in STRIDES.iter().enumerate() {
        let cells = (size / stride) as usize;
        let rows = cells * cells * ANCHORS_PER_CELL;
        let scores = find(outputs, rows, 1, "scores", level)?;
        let boxes = find(outputs, rows, 4, "boxes", level)?;
        let points = find(outputs, rows, LANDMARKS * 2, "keypoints", level)?;
        decode_level(
            scores,
            boxes,
            points,
            cells,
            *stride as f32,
            letterbox,
            score_threshold,
            &mut detections,
        );
    }
    Ok(detections)
}

/// One stride's worth of rows, in the deployment layout: spatial order, anchors interleaved.
///
/// Shared by both callers so the geometry exists once. `decode` feeds it ONNX Runtime's
/// tensors; `decode_maps` feeds it the built-in engines' raw maps, transposed to match.
#[allow(clippy::too_many_arguments)]
fn decode_level(
    scores: &[f32],
    boxes: &[f32],
    points: &[f32],
    cells: usize,
    stride: f32,
    letterbox: Letterbox,
    score_threshold: f32,
    detections: &mut Vec<Detection>,
) {
    {
        let rows = cells * cells * ANCHORS_PER_CELL;
        for row in 0..rows {
            let score = scores[row];
            if score < score_threshold {
                continue;
            }
            // Rows run spatially with the anchors interleaved, so a cell's anchors are adjacent.
            let cell = row / ANCHORS_PER_CELL;
            let centre_x = (cell % cells) as f32 * stride;
            let centre_y = (cell / cells) as f32 * stride;

            // distance2bbox: left, top, right, bottom distances outward from the centre.
            let d = &boxes[row * 4..row * 4 + 4];
            let x1 = centre_x - d[0] * stride;
            let y1 = centre_y - d[1] * stride;
            let x2 = centre_x + d[2] * stride;
            let y2 = centre_y + d[3] * stride;

            // Only the two eyes are real. This model's landmark head was trained on eye pairs
            // clicked for this project, with nose and mouth corners written `-1` and weighted
            // to zero (`tools/dataset/to_labelv2.py`), so those three outputs received no
            // gradient at all: they emit near-zero distances that decode to the anchor centre,
            // which lands *above* the face and is identical for all three. Left in, they look
            // like landmarks and are drawn as landmarks, so they are reported absent.
            let mut landmarks = [None; LANDMARKS];
            for (index, landmark) in landmarks.iter_mut().enumerate().take(TRAINED_LANDMARKS) {
                let dx = points[row * LANDMARKS * 2 + index * 2];
                let dy = points[row * LANDMARKS * 2 + index * 2 + 1];
                *landmark = Some(Landmark::new(
                    (centre_x + dx * stride) / letterbox.scale,
                    (centre_y + dy * stride) / letterbox.scale,
                ));
            }

            detections.push(Detection {
                bbox: BoundingBox {
                    x: x1 / letterbox.scale,
                    y: y1 / letterbox.scale,
                    width: (x2 - x1) / letterbox.scale,
                    height: (y2 - y1) / letterbox.scale,
                },
                landmarks,
                score,
            });
        }
    }
}

/// Decode the raw `(1, anchors*channels, height, width)` maps the built-in engines produce.
///
/// They stop at the head's convolutions, because the sigmoid and the deployment reshape that
/// ONNX Runtime's graph carries are cheaper to do here than to add as engine ops. Each map is
/// transposed into the same row order `decode_level` expects: spatial position first, anchors
/// interleaved, channels innermost.
pub(crate) fn decode_maps(
    maps: &[(Vec<f32>, [usize; 4])],
    letterbox: Letterbox,
    score_threshold: f32,
    size: u32,
) -> Result<Vec<Detection>> {
    let mut detections = Vec::new();

    for (level, stride) in STRIDES.iter().enumerate() {
        let cells = (size / stride) as usize;
        let scores = deployment_rows(maps, level, 1, cells, true, "scores")?;
        let boxes = deployment_rows(maps, level + 3, 4, cells, false, "boxes")?;
        let points = deployment_rows(maps, level + 6, LANDMARKS * 2, cells, false, "keypoints")?;
        decode_level(
            &scores,
            &boxes,
            &points,
            cells,
            *stride as f32,
            letterbox,
            score_threshold,
            &mut detections,
        );
    }
    Ok(detections)
}

/// Transpose one map from `(anchors*channels, height, width)` into rows of `channels`.
fn deployment_rows(
    maps: &[(Vec<f32>, [usize; 4])],
    at: usize,
    channels: usize,
    cells: usize,
    sigmoid: bool,
    what: &str,
) -> Result<Vec<f32>> {
    let (data, dims) = maps.get(at).ok_or_else(|| {
        anyhow::anyhow!(
            "no {what} map at index {at}: the engine returned {} maps",
            maps.len()
        )
    })?;
    let (height, width) = (dims[2], dims[3]);
    anyhow::ensure!(
        height == cells && width == cells && dims[1] == ANCHORS_PER_CELL * channels,
        "{what} map is {dims:?}, expected {ANCHORS_PER_CELL}x{channels} channels over {cells}x{cells}"
    );

    let plane = height * width;
    let mut rows = vec![0.0f32; plane * ANCHORS_PER_CELL * channels];
    for y in 0..height {
        for x in 0..width {
            for anchor in 0..ANCHORS_PER_CELL {
                for channel in 0..channels {
                    let value = data[(anchor * channels + channel) * plane + y * width + x];
                    let row = (y * width + x) * ANCHORS_PER_CELL + anchor;
                    rows[row * channels + channel] = if sigmoid {
                        1.0 / (1.0 + (-value).exp())
                    } else {
                        value
                    };
                }
            }
        }
    }
    Ok(rows)
}

/// The tensor with this row count and channel count, or a readable error.
#[cfg(test)]
fn find<'a>(
    outputs: &'a [fcs_ort::OutputTensor],
    rows: usize,
    channels: usize,
    what: &str,
    level: usize,
) -> Result<&'a [f32]> {
    outputs
        .iter()
        .find(|tensor| {
            let shape = &tensor.shape;
            shape.last() == Some(&channels) && tensor.data.len() == rows * channels
        })
        .map(|tensor| tensor.data.as_slice())
        .ok_or_else(|| {
            anyhow::anyhow!(
                "SCRFD output for {what} at stride level {level} is missing: wanted {rows}x{channels}, \
                 the model returned {:?}",
                outputs.iter().map(|t| t.shape.clone()).collect::<Vec<_>>()
            )
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{Rgb, RgbImage};

    fn tensor(shape: Vec<usize>, data: Vec<f32>) -> fcs_ort::OutputTensor {
        fcs_ort::OutputTensor { shape, data }
    }

    #[test]
    fn a_wide_image_is_letterboxed_along_the_top() {
        // 1000x500 is twice as wide as tall, so it fills the width and half the height.
        let image = DynamicImage::ImageRgb8(RgbImage::from_pixel(1000, 500, Rgb([255, 255, 255])));
        let (input, fit) = preprocess(&image, 640);
        assert!((fit.scale - 0.64).abs() < 1e-6, "scale was {}", fit.scale);

        let plane = 640 * 640;
        // Inside the image: white, normalised.
        assert!((input[0] - (255.0 - 127.5) / 128.0).abs() < 1e-6);
        // Below it: padding, which must carry the normalised zero, not zero.
        let below = 500 * 640; // first row past the resized content
        assert!(
            (input[below] - (0.0 - 127.5) / 128.0).abs() < 1e-6,
            "padding was {}",
            input[below]
        );
        assert_eq!(input.len(), 3 * plane);
    }

    #[test]
    fn a_tall_image_fills_the_height() {
        let image = DynamicImage::ImageRgb8(RgbImage::from_pixel(500, 1000, Rgb([10, 20, 30])));
        let (_, fit) = preprocess(&image, 640);
        assert!((fit.scale - 0.64).abs() < 1e-6, "scale was {}", fit.scale);
    }

    /// Non-degenerate raw head maps in the engines' own `(1, A*C, H, W)` layout.
    ///
    /// Every value depends on all four of channel, anchor, y and x, with coefficients chosen so
    /// no two index terms can swap and give the same number -- the fixture-hygiene rule from
    /// `.claude/skills/run-mutants`. A map full of zeros, or one where `y * width` happens to
    /// equal `x * height`, lets an index mistake through however exact the assertion looks.
    ///
    /// Scores are mostly negative so only a handful of anchors clear the threshold after the
    /// sigmoid; `sigmoid(v) > 0.5` exactly when `v > 0`.
    fn raw_maps(size: u32) -> Vec<(Vec<f32>, [usize; 4])> {
        let mut maps = Vec::new();
        // Grouped by kind, then stride -- the order `decode_maps` indexes with `level + 3`.
        for channels in [1usize, 4, LANDMARKS * 2] {
            for stride in STRIDES {
                let cells = (size / stride) as usize;
                let plane = cells * cells;
                let mut data = vec![0.0f32; ANCHORS_PER_CELL * channels * plane];
                for anchor in 0..ANCHORS_PER_CELL {
                    for channel in 0..channels {
                        for y in 0..cells {
                            for x in 0..cells {
                                let at = (anchor * channels + channel) * plane + y * cells + x;
                                let v = 0.37 * (channel as f32 + 1.0)
                                    + 0.13 * x as f32
                                    + 0.29 * y as f32
                                    + 0.71 * anchor as f32;
                                data[at] = if channels == 1 {
                                    // One cell per stride clears the threshold, so every stride
                                    // contributes detections rather than just the coarsest.
                                    if x == 2 && y == 3 { v } else { -v - 0.5 }
                                } else {
                                    v
                                };
                            }
                        }
                    }
                }
                maps.push((data, [1, ANCHORS_PER_CELL * channels, cells, cells]));
            }
        }
        maps
    }

    /// Transpose a raw map into the deployment layout, written out independently here.
    ///
    /// Deliberately not a call to `deployment_rows`: this is the reference that function is
    /// checked against, so it has to be a separate statement of the same intent.
    fn to_deployment(map: &(Vec<f32>, [usize; 4]), channels: usize, sigmoid: bool) -> Vec<f32> {
        let (data, dims) = map;
        let (height, width) = (dims[2], dims[3]);
        let plane = height * width;
        let mut rows = vec![0.0f32; plane * ANCHORS_PER_CELL * channels];
        for y in 0..height {
            for x in 0..width {
                for anchor in 0..ANCHORS_PER_CELL {
                    for channel in 0..channels {
                        let value = data[(anchor * channels + channel) * plane + y * width + x];
                        let row = (y * width + x) * ANCHORS_PER_CELL + anchor;
                        rows[row * channels + channel] = if sigmoid {
                            1.0 / (1.0 + (-value).exp())
                        } else {
                            value
                        };
                    }
                }
            }
        }
        rows
    }

    /// The two decode paths must produce identical detections from identical data.
    ///
    /// This is the test the engine-parity suite cannot be: `tests/scrfd_parity.rs` compares the
    /// nine head tensors *before* decoding. The former ONNX Runtime decoder stays test-only
    /// as an independent layout check. So `decode_maps` and `deployment_rows` -- the decode every machine without
    /// ONNX Runtime uses, which is the whole reason the built-in engines exist -- had no test at
    /// all. A mutation run found 70 survivors in this file, most of them their arithmetic.
    ///
    /// Agreement is worth something because the other side is externally validated: torch
    /// agrees with `decode` (python_parity), so `decode_maps` agreeing with `decode` inherits
    /// that. Without the torch oracle this would only prove the two paths are consistent.
    #[test]
    fn the_two_decode_paths_agree() {
        // 160 rather than 640: cells 20/10/5 keep the three strides' row counts distinct (800,
        // 200, 50) so `find` still matches by shape, at a fraction of the arithmetic.
        const SIZE: u32 = 160;
        let maps = raw_maps(SIZE);

        let mut deployment = Vec::new();
        for (kind, channels) in [(0usize, 1usize), (1, 4), (2, LANDMARKS * 2)] {
            for (level, stride) in STRIDES.iter().enumerate() {
                let cells = (SIZE / stride) as usize;
                let rows = cells * cells * ANCHORS_PER_CELL;
                let map = &maps[kind * STRIDES.len() + level];
                let values = to_deployment(map, channels, channels == 1);
                deployment.push(tensor(vec![1, rows, channels], values));
            }
        }

        let from_maps = decode_maps(&maps, Letterbox { scale: 1.5 }, 0.5, SIZE).expect("maps");
        let from_deployment =
            decode(&deployment, Letterbox { scale: 1.5 }, 0.5, SIZE).expect("deployment");

        // Vacuous agreement is the failure mode to guard against: two empty lists match.
        assert!(
            from_maps.len() >= STRIDES.len(),
            "expected at least one detection per stride, got {}",
            from_maps.len()
        );
        assert_eq!(
            from_maps.len(),
            from_deployment.len(),
            "the two paths found different numbers of faces"
        );
        for (index, (a, b)) in from_maps.iter().zip(&from_deployment).enumerate() {
            assert_eq!(a.score, b.score, "detection {index}: score");
            assert_eq!(a.bbox, b.bbox, "detection {index}: box");
            assert_eq!(a.landmarks, b.landmarks, "detection {index}: landmarks");
        }
    }

    fn strict_tests() -> bool {
        std::env::var("FCS_STRICT_TESTS").is_ok_and(|v| v != "0" && !v.is_empty())
    }

    /// The model the workspace ships, resolved from this crate rather than the working
    /// directory, which for a unit test is the crate root.
    fn default_model_for_test() -> std::path::PathBuf {
        std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .expect("fcs-core sits in the workspace root")
            .join(DEFAULT_MODEL)
    }

    /// `engine` names the backend, and the GPU parity tests *skip* on anything unexpected -- so
    /// a mutant returning `""` made those tests skip and pass. Pinned here instead.
    #[test]
    fn engine_reports_a_builtin_backend() {
        let Some(detector) = ScrfdDetector::load_from(default_model_for_test()) else {
            assert!(
                !strict_tests(),
                "FCS_STRICT_TESTS: the shipped model did not load"
            );
            eprintln!("skipped: no model present");
            return;
        };
        let engine = detector.engine();
        assert!(
            ["wgsl-gpu", "cpu-graph"].contains(&engine),
            "unknown engine name {engine:?}"
        );
    }

    /// Loading the shipped model must succeed.
    ///
    /// Loading returns `Option`, and every other test skipped on `None`, so `-> None` mutants
    /// were invisible: the suite skipped and passed. Under `FCS_STRICT_TESTS` -- which CI sets
    /// -- an absent detector is now a failure, which is what makes those mutants reachable.
    #[test]
    fn the_shipped_model_loads() {
        let model = default_model_for_test();
        if !model.exists() {
            assert!(!strict_tests(), "FCS_STRICT_TESTS: no model at {model:?}");
            eprintln!("skipped: no model present");
            return;
        }
        assert!(
            ScrfdDetector::load_from(&model).is_some(),
            "load_from returned None for a model that exists"
        );
        // And the negative case, which is what the `!path.exists()` guard is for.
        assert!(
            ScrfdDetector::load_from(model.with_file_name("absent.onnx")).is_none(),
            "a missing file must not produce a detector"
        );
    }

    /// The two machines the rule was written for, and the boundary between them.
    #[test]
    fn the_gpu_keeps_the_job_unless_the_cpu_is_clearly_faster() {
        assert!(prefer_gpu(2.0, 5.5), "RTX 4090: the GPU is faster");
        assert!(
            !prefer_gpu(22.0, 7.5),
            "integrated GPU: the CPU is three times faster"
        );
        assert!(prefer_gpu(10.0, 9.0), "a near tie goes to the GPU");
        assert!(prefer_gpu(10.0, 8.0), "exactly 25% faster is still a tie");
        assert!(!prefer_gpu(10.0, 7.9), "past 25% the CPU takes it");
    }

    /// A square image takes either branch of the aspect-ratio test and must come out the same.
    ///
    /// This pins `>` against `>=` in `preprocess` as **equivalent**: at `ratio == 1.0` the
    /// portrait arm computes `(size / 1.0, size)` and the landscape arm `(size, size * 1.0)`,
    /// which are the same pair. The mutant cannot be killed, and this says so rather than
    /// leaving the next reader to rediscover it.
    #[test]
    fn a_square_image_is_unaffected_by_which_aspect_branch_runs() {
        let image = DynamicImage::ImageRgb8(RgbImage::from_pixel(300, 300, Rgb([40, 50, 60])));
        let (tensor, fit) = preprocess(&image, 640);
        assert!(
            (fit.scale - 640.0 / 300.0).abs() < 1e-6,
            "scale {}",
            fit.scale
        );
        assert_eq!(tensor.len(), 3 * 640 * 640);
    }

    /// Every term of `decode_level`, on a fixture where none of them can collapse.
    ///
    /// The 34 mutants that survived the first round of this work were all here, and the reason
    /// is that `the_two_decode_paths_agree` cannot reach shared code: `decode_level` is called
    /// by *both* paths, so mutating it changes both sides identically and they still agree. An
    /// agreement test is blind to the code the two sides have in common.
    ///
    /// So this asserts absolute values, and the fixture is picked so each operator is
    /// observable on its own:
    ///
    /// * a cell that is neither on row 0 nor column 0, so `cell % cells` and `cell / cells` are
    ///   different non-zero numbers and `centre_y` is not zero;
    /// * the second anchor, so `row / ANCHORS_PER_CELL` is not the identity;
    /// * four different box distances, none of them 0 or 1, so no pair of `centre +- d * stride`
    ///   terms agree;
    /// * both landmarks, each with a non-zero dx *and* dy, so `index * 2` is exercised at 1 as
    ///   well as 0 and the y arithmetic is not multiplied by zero;
    /// * a letterbox scale of 1.6, because at 1.0 dividing by it is the identity and `/` can be
    ///   swapped for `*` freely -- which is precisely what survived before.
    #[test]
    fn decode_level_computes_every_term_from_the_anchor() {
        const STRIDE: f32 = 32.0;
        const SCALE: f32 = 1.6;
        let cells = (640 / 32) as usize; // 20
        let rows = cells * cells * ANCHORS_PER_CELL;

        // Cell (column 7, row 3), second anchor.
        let (column, row_index) = (7usize, 3usize);
        let cell = row_index * cells + column;
        let row = cell * ANCHORS_PER_CELL + 1;

        let mut scores = vec![0.0f32; rows];
        let mut boxes = vec![0.0f32; rows * 4];
        let mut points = vec![0.0f32; rows * LANDMARKS * 2];
        scores[row] = 0.8;
        // left, top, right, bottom -- all distinct, none 0 or 1.
        let d = [0.25f32, 0.75, 1.5, 2.25];
        boxes[row * 4..row * 4 + 4].copy_from_slice(&d);
        // Two landmarks, four distinct non-zero distances.
        let lm = [0.5f32, 0.375, -0.25, 0.625];
        points[row * LANDMARKS * 2..row * LANDMARKS * 2 + 4].copy_from_slice(&lm);

        let mut found = Vec::new();
        decode_level(
            &scores,
            &boxes,
            &points,
            cells,
            STRIDE,
            Letterbox { scale: SCALE },
            0.5,
            &mut found,
        );
        assert_eq!(found.len(), 1, "one anchor was over threshold");
        let got = &found[0];

        let centre_x = column as f32 * STRIDE;
        let centre_y = row_index as f32 * STRIDE;
        let x1 = centre_x - d[0] * STRIDE;
        let y1 = centre_y - d[1] * STRIDE;
        let x2 = centre_x + d[2] * STRIDE;
        let y2 = centre_y + d[3] * STRIDE;

        let close = |got: f32, want: f32, what: &str| {
            assert!(
                (got - want).abs() < 1e-3,
                "{what}: got {got}, expected {want}"
            );
        };
        close(got.score, 0.8, "score");
        close(got.bbox.x, x1 / SCALE, "box x");
        close(got.bbox.y, y1 / SCALE, "box y");
        close(got.bbox.width, (x2 - x1) / SCALE, "box width");
        close(got.bbox.height, (y2 - y1) / SCALE, "box height");

        // Nothing in the box may coincide, or a swapped pair would read as correct.
        for (a, b) in [
            (got.bbox.x, got.bbox.y),
            (got.bbox.width, got.bbox.height),
            (got.bbox.x, got.bbox.width),
        ] {
            assert!((a - b).abs() > 1.0, "fixture is degenerate: {a} vs {b}");
        }

        for index in 0..TRAINED_LANDMARKS {
            let point = got.landmarks[index].expect("the two eyes are predicted");
            close(
                point.x,
                (centre_x + lm[index * 2] * STRIDE) / SCALE,
                "landmark x",
            );
            close(
                point.y,
                (centre_y + lm[index * 2 + 1] * STRIDE) / SCALE,
                "landmark y",
            );
        }
        // The two eyes must differ on both axes, or `index * 2` is not being read.
        let (first, second) = (
            got.landmarks[0].expect("eye 0"),
            got.landmarks[1].expect("eye 1"),
        );
        assert!((first.x - second.x).abs() > 1.0, "the eyes share an x");
        assert!((first.y - second.y).abs() > 1.0, "the eyes share a y");
    }

    /// The last anchor of the last cell must be reachable.
    ///
    /// `rows = cells * cells * ANCHORS_PER_CELL` bounds the scan. Shrinking it -- which is what
    /// `cells * cells + ANCHORS_PER_CELL` does -- loses the high-index anchors, meaning faces in
    /// the bottom-right of the frame silently stop being found. Every other fixture here puts
    /// its detection at a low row, so the bound was never exercised.
    #[test]
    fn the_final_anchor_of_the_grid_is_scanned() {
        const STRIDE: f32 = 32.0;
        let cells = (640 / 32) as usize;
        let rows = cells * cells * ANCHORS_PER_CELL;
        let last = rows - 1;

        let mut scores = vec![0.0f32; rows];
        let mut boxes = vec![0.0f32; rows * 4];
        let points = vec![0.0f32; rows * LANDMARKS * 2];
        scores[last] = 0.75;
        boxes[last * 4..last * 4 + 4].copy_from_slice(&[0.5, 0.25, 1.25, 1.75]);

        let mut found = Vec::new();
        decode_level(
            &scores,
            &boxes,
            &points,
            cells,
            STRIDE,
            Letterbox { scale: 1.0 },
            0.5,
            &mut found,
        );
        assert_eq!(
            found.len(),
            1,
            "the last anchor ({last} of {rows}) was not scanned"
        );
        // And it belongs to the last cell, so the centre is at the far corner of the grid.
        let expected_centre = (cells - 1) as f32 * STRIDE;
        let got = &found[0];
        assert!(
            (got.bbox.x + 0.5 * STRIDE - expected_centre).abs() < 1e-3,
            "box x {} does not sit at the last cell ({expected_centre})",
            got.bbox.x
        );
    }

    /// A score exactly equal to the threshold is kept.
    ///
    /// The guard is `score < threshold`, so `<` and `<=` differ only at equality -- and no
    /// fixture hit it, which is the "threshold equal to the fixture's score" row of the
    /// fixture-hygiene table read the other way round.
    #[test]
    fn a_score_exactly_on_the_threshold_is_kept() {
        const STRIDE: f32 = 32.0;
        let cells = (640 / 32) as usize;
        let rows = cells * cells * ANCHORS_PER_CELL;
        let mut scores = vec![0.0f32; rows];
        let mut boxes = vec![0.0f32; rows * 4];
        let points = vec![0.0f32; rows * LANDMARKS * 2];
        scores[7] = 0.625; // exactly the threshold below, and exactly representable in binary
        boxes[28..32].copy_from_slice(&[0.5, 0.25, 1.25, 1.75]);

        let mut found = Vec::new();
        decode_level(
            &scores,
            &boxes,
            &points,
            cells,
            STRIDE,
            Letterbox { scale: 1.0 },
            0.625,
            &mut found,
        );
        assert_eq!(
            found.len(),
            1,
            "a score equal to the threshold must be kept"
        );

        // And one hair below it is dropped, so the comparison is not simply always true.
        let mut below = vec![0.0f32; rows];
        below[7] = 0.624;
        let mut dropped = Vec::new();
        decode_level(
            &below,
            &boxes,
            &points,
            cells,
            STRIDE,
            Letterbox { scale: 1.0 },
            0.625,
            &mut dropped,
        );
        assert!(
            dropped.is_empty(),
            "a score below the threshold must be dropped"
        );
    }

    /// `ScrfdDetector::load` resolves the model relative to the working directory, which under
    /// `cargo test` is the crate root. It therefore returns `None` here whatever it does, so this
    /// only pins the negative half. The positive half -- that `load` finds the model from the
    /// workspace root -- needs `set_current_dir`, which is process-global, so it lives in
    /// `tests/default_model_location.rs`, a binary with nothing else in it to race with.
    #[test]
    fn load_resolves_relative_to_the_working_directory() {
        assert!(
            !std::path::Path::new(DEFAULT_MODEL).exists(),
            "this test's premise is that {DEFAULT_MODEL} is not resolvable from the crate root"
        );
        assert!(ScrfdDetector::load().is_none());
    }

    /// One anchor over threshold at stride 32, decoded by hand.
    fn single_detection_outputs(score: f32) -> Vec<fcs_ort::OutputTensor> {
        let mut outputs = Vec::new();
        for stride in STRIDES {
            let cells = (640 / stride) as usize;
            let rows = cells * cells * ANCHORS_PER_CELL;
            let mut scores = vec![0.0f32; rows];
            let mut boxes = vec![0.0f32; rows * 4];
            let mut points = vec![0.0f32; rows * LANDMARKS * 2];
            if stride == 32 {
                // Cell 5 -> centre (5*32, 0) = (160, 0); first anchor of that cell is row 10.
                scores[10] = score;
                boxes[40..44].copy_from_slice(&[1.0, 0.0, 1.0, 2.0]);
                points[100] = 0.5; // first landmark, x only
            }
            outputs.push(tensor(vec![1, rows, 1], scores));
            outputs.push(tensor(vec![1, rows, 4], boxes));
            outputs.push(tensor(vec![1, rows, LANDMARKS * 2], points));
        }
        outputs
    }

    #[test]
    fn decode_places_a_box_using_stride_scaled_distances() {
        let outputs = single_detection_outputs(0.9);
        let found = decode(&outputs, Letterbox { scale: 1.0 }, 0.5, 640).unwrap();
        assert_eq!(found.len(), 1);
        let detection = &found[0];
        // centre (160, 0); distances (1,0,1,2) * stride 32.
        assert!(
            (detection.bbox.x - 128.0).abs() < 1e-3,
            "x {}",
            detection.bbox.x
        );
        assert!((detection.bbox.y - 0.0).abs() < 1e-3);
        assert!(
            (detection.bbox.width - 64.0).abs() < 1e-3,
            "w {}",
            detection.bbox.width
        );
        assert!(
            (detection.bbox.height - 64.0).abs() < 1e-3,
            "h {}",
            detection.bbox.height
        );
        let eye = detection.landmarks[0].expect("the first eye is predicted");
        assert!((eye.x - 176.0).abs() < 1e-3, "kp {}", eye.x);
    }

    #[test]
    fn untrained_landmarks_are_reported_absent_rather_than_guessed() {
        // The fixture puts a non-zero distance on landmark 0 only; every other landmark reads
        // zero, which is what the real model does for nose and mouth. Passing those through
        // decodes them to the anchor centre -- three identical points above the face, which
        // look like predictions and get drawn like predictions. They must come back as `None`,
        // so a consumer cannot read one by accident.
        let outputs = single_detection_outputs(0.9);
        let found = decode(&outputs, Letterbox { scale: 1.0 }, 0.5, 640).unwrap();
        let landmarks = found[0].landmarks;
        assert!(landmarks[0].is_some(), "the eyes are predicted");
        assert!(landmarks[1].is_some(), "the eyes are predicted");
        for (index, landmark) in landmarks.iter().enumerate().skip(TRAINED_LANDMARKS) {
            assert!(
                landmark.is_none(),
                "landmark {index} was never trained and must read as absent, got {landmark:?}"
            );
        }
    }

    #[test]
    fn decode_undoes_the_letterbox_scale() {
        let outputs = single_detection_outputs(0.9);
        let found = decode(&outputs, Letterbox { scale: 0.5 }, 0.5, 640).unwrap();
        assert!(
            (found[0].bbox.x - 256.0).abs() < 1e-3,
            "x {}",
            found[0].bbox.x
        );
        assert!((found[0].bbox.width - 128.0).abs() < 1e-3);
    }

    #[test]
    fn scores_below_the_threshold_are_dropped() {
        let outputs = single_detection_outputs(0.3);
        assert!(
            decode(&outputs, Letterbox { scale: 1.0 }, 0.5, 640)
                .unwrap()
                .is_empty()
        );
    }

    #[test]
    fn a_missing_output_is_an_error_naming_what_was_wanted() {
        let outputs = vec![tensor(vec![1, 12800, 1], vec![0.0; 12800])];
        let err = decode(&outputs, Letterbox { scale: 1.0 }, 0.5, 640)
            .unwrap_err()
            .to_string();
        assert!(err.contains("boxes"), "{err}");
    }
}
