//! SCRFD: the detector trained on this project's own licence-clean data.
//!
//! It finds 86.0% of the faces in the Open Images test split at 0.11 false positives per image,
//! against YuNet's 71.6% at 0.14, and on a corpus neither model had seen it found 135 faces
//! YuNet missed while missing 13 it found (`tools/dataset/SCRFD_80K.md`). It cost about 10%
//! more per image: 6.4 ms against 5.8 at 640x640 on ONNX Runtime.
//!
//! **It is the only detector.** It runs on all three engines -- ONNX Runtime, the WGSL kernels
//! (`gpu`), or the built-in CPU graph (`plan`) -- which agree to about 1e-05 and produce
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
/// See the note in [`decode`] on why the rest are reported absent rather than passed through.
const TRAINED_LANDMARKS: usize = 2;

/// Workspace-relative location of the exported model.
const DEFAULT_MODEL: &str = "models/scrfd80k_500m_640.onnx";

/// A loaded SCRFD detector, on whichever engine is available.
#[derive(Debug)]
pub struct ScrfdDetector {
    backend: Backend,
}

/// Where the network actually runs.
///
/// ONNX Runtime first because it is the fastest and every release bundles it, then the WGSL
/// engine, then the built-in CPU graph. All three were checked against each other on the same
/// input: 1.1e-05 between GPU and ONNX Runtime, 1.2e-05 between CPU and ONNX Runtime.
#[derive(Debug)]
enum Backend {
    Ort(fcs_ort::Session),
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
    /// `None` is the ordinary outcome without ONNX Runtime, not an error: the caller keeps
    /// whatever detector it already has.
    pub fn load() -> Option<Self> {
        Self::load_from(fcs_utils::resolve_data_path(DEFAULT_MODEL))
    }

    /// Load from an explicit path. See [`Self::load`] for the `None` cases.
    pub fn load_from<P: AsRef<Path>>(path: P) -> Option<Self> {
        let path = path.as_ref();
        if !path.exists() {
            debug!("SCRFD not loaded: no model at {}", path.display());
            return None;
        }
        let Some(environment) = fcs_ort::Environment::shared() else {
            // Not a failure any more: the same network runs on the built-in engines, which is
            // what lets YuNet leave the packages entirely.
            debug!("no ONNX Runtime for SCRFD; using the built-in engines");
            return Self::load_builtin(path);
        };
        match fcs_ort::Session::new(&environment, path, fcs_ort::SessionOptions::default()) {
            Ok(session) => {
                info!("SCRFD detector loaded from {}", path.display());
                Some(Self {
                    backend: Backend::Ort(session),
                })
            }
            Err(err) => {
                warn!(
                    "SCRFD at {} failed to open a session ({err}); using the built-in engines",
                    path.display()
                );
                Self::load_builtin(path)
            }
        }
    }

    /// Load onto the built-in engines, for a machine with no ONNX Runtime.
    ///
    /// The GPU is tried first and the CPU graph is the floor. This is what lets SCRFD ship
    /// everywhere, and so what lets YuNet -- trained on WIDER FACE, "non-commercial academic
    /// research only" -- leave the packages.
    pub fn load_builtin<P: AsRef<Path>>(path: P) -> Option<Self> {
        let path = path.as_ref();
        if !path.exists() {
            debug!("SCRFD not loaded: no model at {}", path.display());
            return None;
        }

        match fcs_utils::GpuContext::init_with_fallback(&Default::default()) {
            fcs_utils::GpuAvailability::Available(context) => {
                match crate::gpu::GpuInferenceOps::new(context, None)
                    .and_then(|ops| Ok((gpu::ScrfdGpuWeights::load(&ops, path)?, ops)))
                {
                    Ok((weights, ops)) => {
                        info!("SCRFD on the WGSL engine, from {}", path.display());
                        return Some(Self {
                            backend: Backend::Gpu(Box::new((ops, weights))),
                        });
                    }
                    Err(err) => warn!("SCRFD on the GPU failed ({err}); trying the CPU graph"),
                }
            }
            other => debug!("no GPU for SCRFD ({other:?}); using the CPU graph"),
        }

        match plan::ScrfdWeights::load(path) {
            Ok(weights) => {
                info!("SCRFD on the built-in CPU graph, from {}", path.display());
                Some(Self {
                    backend: Backend::Cpu(Box::new(weights)),
                })
            }
            Err(err) => {
                warn!("SCRFD at {} would not load ({err})", path.display());
                None
            }
        }
    }

    /// Which engine is running, for logs.
    pub fn engine(&self) -> &'static str {
        match self.backend {
            Backend::Ort(_) => "onnxruntime",
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
        let side = INPUT_SIZE as usize;
        let shape = [1usize, 3, side, side];

        let mut detections = match &self.backend {
            Backend::Ort(session) => {
                let outputs = session.run(&input, &shape)?;
                decode(&outputs, letterbox, score_threshold, INPUT_SIZE)?
            }
            Backend::Gpu(boxed) => {
                let (ops, weights) = boxed.as_ref();
                let uploaded = ops.upload_tensor(shape.to_vec(), &input, Some("scrfd input"))?;
                let outputs = gpu::run(ops, &uploaded, weights)?;
                let maps = outputs
                    .iter()
                    .map(|tensor| {
                        let dims = tensor.shape().dims();
                        Ok((tensor.to_vec()?, [dims[0], dims[1], dims[2], dims[3]]))
                    })
                    .collect::<Result<Vec<_>>>()?;
                decode_maps(&maps, letterbox, score_threshold, INPUT_SIZE)?
            }
            Backend::Cpu(weights) => {
                let tensor = crate::cpu::tensor::Tensor::new(1, 3, side, side, input)?;
                let outputs = plan::run(tensor, weights)?;
                let maps: Vec<(Vec<f32>, [usize; 4])> = outputs
                    .iter()
                    .map(|t| {
                        (
                            t.data().to_vec(),
                            [t.batch(), t.channels(), t.height(), t.width()],
                        )
                    })
                    .collect();
                decode_maps(&maps, letterbox, score_threshold, INPUT_SIZE)?
            }
        };
        detections.sort_by(|a, b| b.score.total_cmp(&a.score));
        apply_nms_in_place(&mut detections, nms_threshold);
        Ok(detections)
    }
}

/// Resize into the top-left of a square canvas and normalise, as the model was trained.
///
/// Three details that are all load-bearing, and all silent when wrong: the source goes to the
/// **top-left** rather than being centred the way `crate::preprocess` letterboxes; channels
/// are **RGB**, not the BGR that module produces; and the padding is normalised along with everything
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
pub(crate) fn decode(
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
            // like landmarks and are drawn as landmarks. The all-zero point is this codebase's
            // existing "absent" marker, which `face_cropper` already tests for.
            let mut landmarks = [Landmark::new(0.0, 0.0); LANDMARKS];
            for (index, landmark) in landmarks.iter_mut().enumerate().take(TRAINED_LANDMARKS) {
                let dx = points[row * LANDMARKS * 2 + index * 2];
                let dy = points[row * LANDMARKS * 2 + index * 2 + 1];
                *landmark = Landmark::new(
                    (centre_x + dx * stride) / letterbox.scale,
                    (centre_y + dy * stride) / letterbox.scale,
                );
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
        assert!(
            (detection.landmarks[0].x - 176.0).abs() < 1e-3,
            "kp {}",
            detection.landmarks[0].x
        );
    }

    #[test]
    fn untrained_landmarks_are_reported_absent_rather_than_guessed() {
        // The fixture puts a non-zero distance on landmark 0 only; every other landmark reads
        // zero, which is what the real model does for nose and mouth. Passing those through
        // decodes them to the anchor centre -- three identical points above the face, which
        // look like predictions and get drawn like predictions.
        let outputs = single_detection_outputs(0.9);
        let found = decode(&outputs, Letterbox { scale: 1.0 }, 0.5, 640).unwrap();
        let landmarks = found[0].landmarks;
        assert!(landmarks[0].x != 0.0, "the eyes are predicted");
        for (index, landmark) in landmarks.iter().enumerate().skip(TRAINED_LANDMARKS) {
            assert_eq!(
                (landmark.x, landmark.y),
                (0.0, 0.0),
                "landmark {index} was never trained and must read as absent"
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
