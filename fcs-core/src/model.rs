use crate::{
    cpu::runtime::CpuYuNet,
    model_config::{
        DETECTION_OUTPUT_COLS, DETECTION_SCORE_INDEX, OUTPUTS_PER_STRIDE, STRIDE_ALIGNMENT, STRIDES,
    },
    ort_backend::OrtBackend,
    preprocess::InputSize,
};

use crate::tensor::Tensor;
use anyhow::{Context, Result};
use log::{info, warn};
use std::path::Path;

#[derive(Clone, Copy)]
struct StrideMeta {
    stride_index: usize,
    stride: usize,
    cols: usize,
    rows: usize,
    cell_count: usize,
    offset: usize,
}

#[derive(Clone)]
struct StrideLayout {
    metas: Vec<StrideMeta>,
    total_capacity: usize,
}

#[derive(Debug)]
struct StrideOutputs<'a> {
    cls: &'a [f32],
    obj: &'a [f32],
    bbox: &'a [f32],
    kps: &'a [f32],
}

#[derive(Clone, Copy)]
struct CellDecodeInput {
    row: usize,
    col: usize,
    stride_f: f32,
    cls_score: f32,
    obj_score: f32,
    bbox: [f32; 4],
    kps: [f32; 10],
}

/// Wrapper around the YuNet ONNX runnable model.
///
/// This struct handles loading the ONNX graph, preparing it for execution, and running inference.
/// Which inference runtime to use. `Auto` is what shipping code wants; the
/// explicit variants exist so the backend-parity test can run both over the
/// same fixtures, and so a user hitting a bad ONNX Runtime can be told to force
/// the built-in graph rather than being stuck.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum InferenceBackend {
    /// ONNX Runtime when a compatible library is present, else the built-in
    /// graph.
    #[default]
    Auto,
    /// ONNX Runtime, failing if no compatible library is present.
    OnnxRuntime,
    /// The built-in pure-Rust graph. Needs nothing installed, but knows only
    /// YuNet's topology, so it fails on any other model.
    CpuGraph,
}

#[derive(Debug)]
enum Backend {
    /// One session, shared. ONNX Runtime allows concurrent runs on it, so
    /// batch export needs neither a lock nor a pool.
    Ort(OrtBackend),
    /// The built-in pure-Rust graph.
    CpuGraph(Box<CpuYuNet>),
}

#[derive(Debug)]
pub struct YuNetModel {
    backend: Backend,
    input_size: InputSize,
}

impl YuNetModel {
    /// Load YuNet, preferring ONNX Runtime when a compatible library is present
    /// and falling back to the built-in graph otherwise — mirroring how GPU
    /// acceleration is already selected by availability rather than by
    /// configuration. See `fcs_ort` for why the probe has to happen before any
    /// ONNX Runtime call.
    pub fn load<P: AsRef<Path>>(model_path: P, input_size: InputSize) -> Result<Self> {
        Self::load_with(model_path, input_size, InferenceBackend::Auto)
    }

    /// Load with an explicit backend choice. See [`InferenceBackend`].
    pub fn load_with<P: AsRef<Path>>(
        model_path: P,
        input_size: InputSize,
        backend: InferenceBackend,
    ) -> Result<Self> {
        let path = model_path.as_ref();
        anyhow::ensure!(path.exists(), "model file not found: {}", path.display());

        let wants_ort = matches!(
            backend,
            InferenceBackend::Auto | InferenceBackend::OnnxRuntime
        );
        let environment = wants_ort.then(fcs_ort::Environment::shared).flatten();

        if backend == InferenceBackend::OnnxRuntime && environment.is_none() {
            anyhow::bail!(
                "InferenceBackend::OnnxRuntime requested but no compatible ONNX Runtime                  (1.{}+) was found; set ORT_DYLIB_PATH or place {} beside the executable",
                fcs_ort::REQUIRED_API_VERSION,
                fcs_ort::library_name(),
            );
        }

        if backend == InferenceBackend::CpuGraph {
            let graph = CpuYuNet::load(path, input_size)
                .context("InferenceBackend::CpuGraph requested but the model would not load")?;
            return Ok(Self {
                backend: Backend::CpuGraph(Box::new(graph)),
                input_size,
            });
        }

        if let Some(environment) = environment {
            match OrtBackend::load(&environment, path) {
                Ok(backend) => {
                    info!(
                        "YuNet inference backend: ONNX Runtime {} ({})",
                        environment.runtime().version(),
                        environment.runtime().path().display()
                    );
                    return Ok(Self {
                        backend: Backend::Ort(backend),
                        input_size,
                    });
                }
                Err(err) => warn!(
                    "ONNX Runtime {} at {} failed to open a session ({err}); using the built-in graph",
                    environment.runtime().version(),
                    environment.runtime().path().display()
                ),
            }
        }
        // Nothing else is available: the built-in graph needs no runtime and is
        // the only remaining backend. It knows YuNet's topology and nothing
        // else, so a model whose initializers do not match is a hard error
        // rather than a fallback — which is the honest outcome now that no
        // general ONNX interpreter is bundled.
        let graph = CpuYuNet::load(path, input_size).with_context(|| {
            format!(
                "cannot run {}: no compatible ONNX Runtime was found, and the built-in graph                  accepts only a YuNet export matching the bundled model",
                path.display()
            )
        })?;
        info!("YuNet inference backend: built-in CPU graph");
        Ok(Self {
            backend: Backend::CpuGraph(Box::new(graph)),
            input_size,
        })
    }

    /// Execute YuNet with a preprocessed tensor and return decoded detections.
    ///
    /// The resulting tensor has shape `[N, 15]` where each row is
    /// `[x, y, w, h, re_x, re_y, le_x, le_y, nt_x, nt_y, rcm_x, rcm_y, lcm_x, lcm_y, score]`
    /// in the resized input coordinate space.
    pub fn run(&self, input: Tensor) -> Result<Tensor> {
        let mut tensors: Vec<Tensor> = match &self.backend {
            Backend::Ort(session) => session.run(&input, self.input_size)?,
            Backend::CpuGraph(graph) => graph.head_tensors(&input)?,
        };

        match tensors.len() {
            0 => anyhow::bail!("YuNet model produced no outputs"),
            1 => Ok(tensors
                .pop()
                .ok_or_else(|| anyhow::anyhow!("YuNet model produced no outputs"))?),
            len if len == STRIDES.len() * OUTPUTS_PER_STRIDE => {
                decode_yunet_outputs(&tensors, self.input_size)
            }
            other => anyhow::bail!(
                "unexpected number of YuNet outputs: expected 1 or {}, got {}",
                STRIDES.len() * OUTPUTS_PER_STRIDE,
                other
            ),
        }
    }

    /// Name of the runtime actually in use, for logging and telemetry.
    pub fn backend_name(&self) -> &'static str {
        match &self.backend {
            Backend::Ort(_) => "onnxruntime",
            Backend::CpuGraph(_) => "cpu-graph",
        }
    }

    pub fn input_size(&self) -> InputSize {
        self.input_size
    }
}

#[inline]
pub(crate) fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Decode YuNet's twelve raw head tensors into `[N, 15]` detection rows.
///
/// Public because a backend that produces head tensors — including one outside
/// this crate — still needs the shared decoder; having two decoders is how the
/// backends would drift apart.
pub fn decode_yunet_outputs(outputs: &[Tensor], input_size: InputSize) -> Result<Tensor> {
    decode_yunet_outputs_with(outputs, input_size, HeadLayout::CellMajorActivated, None)
}

/// The score below which a cell cannot survive postprocessing, if the caller knows it.
///
/// Decoding one cell costs four exponentials and a square root, and
/// `examples/decode_cost.rs` puts that at 0.053 ms of the decode's 0.074 -- it is
/// arithmetic-bound, not traffic-bound, so skipping the arithmetic for cells that cannot
/// survive is most of the stage. `apply_postprocess` then discards them anyway.
///
/// The test is exact rather than approximate. Both scores pass through a sigmoid and the
/// pair through a square root, all monotonic, and `score^2 = s(cls) * s(obj) <=
/// min(s(cls), s(obj))` because neither factor exceeds one. So a cell whose *smaller* logit
/// is below `logit(threshold^2)` cannot reach the threshold, whatever the other one is.
/// A `NaN` fails the comparison and takes the full path, which is where the existing
/// non-finite handling lives.
#[derive(Clone, Copy, Debug)]
struct ScoreGate {
    /// Compared against whichever of cls/obj is smaller, in the units of that layout.
    floor: f32,
}

impl ScoreGate {
    /// `None` when no threshold can prune anything: outside `(0, 1)` the bound is vacuous
    /// or rejects everything, and neither is worth a special case here.
    fn new(min_score: Option<f32>, layout: HeadLayout) -> Option<Self> {
        let threshold = min_score?;
        if !(threshold > 0.0 && threshold < 1.0) {
            return None;
        }
        let squared = threshold * threshold;
        Some(Self {
            floor: match layout {
                // Already through sigmoid, so compare probabilities directly.
                HeadLayout::CellMajorActivated => squared,
                // Raw logits, so compare against the logit of the squared threshold.
                HeadLayout::ChannelMajorLogits => (squared / (1.0 - squared)).ln(),
            },
        })
    }

    fn rejects(&self, cls: f32, obj: f32) -> bool {
        cls < self.floor || obj < self.floor
    }
}

/// How one stride's four head tensors are laid out, and whether their scores are activated.
///
/// The two variants are the two producers, not a general matrix: layout and activation
/// travel together because each backend fixes both.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum HeadLayout {
    /// One row per cell -- `bbox[cell * 4 + channel]` -- with `cls` and `obj` already
    /// through sigmoid. What ONNX Runtime, tract and the CPU graph produce.
    CellMajorActivated,
    /// One plane per channel -- `bbox[channel * cells + cell]` -- with `cls` and `obj`
    /// still raw logits. What the GPU heads write, so the GPU path decodes straight from
    /// the downloaded buffers instead of transposing them first.
    ChannelMajorLogits,
}

/// [`decode_yunet_outputs`] for a caller whose heads are not cell-major and activated.
pub fn decode_yunet_outputs_with(
    outputs: &[Tensor],
    input_size: InputSize,
    layout: HeadLayout,
    min_score: Option<f32>,
) -> Result<Tensor> {
    anyhow::ensure!(
        outputs.len() == STRIDES.len() * OUTPUTS_PER_STRIDE,
        "YuNet decode expects {} tensors, got {}",
        STRIDES.len() * OUTPUTS_PER_STRIDE,
        outputs.len()
    );

    let strides = build_stride_layout(input_size)?;
    let mut fused = vec![0f32; strides.total_capacity];

    let gate = ScoreGate::new(min_score, layout);
    for meta in strides.metas.iter() {
        let dst = &mut fused[meta.offset..meta.offset + meta.cell_count * DETECTION_OUTPUT_COLS];
        decode_stride_outputs(outputs, meta, layout, gate, dst)?;
    }

    let rows = fused.len() / DETECTION_OUTPUT_COLS;
    // Every element of `fused` was written above, so hand the buffer over rather than
    // copying all 8400x15 of it into a second allocation.
    Tensor::from_vec(&[rows, DETECTION_OUTPUT_COLS], fused)
        .map_err(|e| anyhow::anyhow!("failed to build fused YuNet tensor: {e}"))
}

fn build_stride_layout(input_size: InputSize) -> Result<StrideLayout> {
    let input_w = input_size.width as usize;
    let input_h = input_size.height as usize;
    let pad_w = align_to(input_w, STRIDE_ALIGNMENT);
    let pad_h = align_to(input_h, STRIDE_ALIGNMENT);

    let mut metas = Vec::with_capacity(STRIDES.len());
    let mut total_cells = 0usize;
    let mut offset = 0usize;

    for (stride_index, &stride) in STRIDES.iter().enumerate() {
        anyhow::ensure!(
            pad_w
                .checked_rem(stride)
                .is_some_and(|remainder| remainder == 0),
            "input width not divisible by stride {}",
            stride
        );
        anyhow::ensure!(
            pad_h
                .checked_rem(stride)
                .is_some_and(|remainder| remainder == 0),
            "input height not divisible by stride {}",
            stride
        );

        let cols = pad_w / stride;
        let rows = pad_h / stride;
        let cell_count = rows * cols;
        let len = cell_count * DETECTION_OUTPUT_COLS;
        metas.push(StrideMeta {
            stride_index,
            stride,
            cols,
            rows,
            cell_count,
            offset,
        });
        total_cells += cell_count;
        offset += len;
    }

    let total_capacity = total_cells * DETECTION_OUTPUT_COLS;
    debug_assert_eq!(offset, total_capacity);

    Ok(StrideLayout {
        metas,
        total_capacity,
    })
}

fn validate_stride_outputs<'a>(
    outputs: &'a [Tensor],
    meta: &StrideMeta,
) -> Result<StrideOutputs<'a>> {
    let cls_slice = outputs[meta.stride_index].as_slice();
    let obj_slice = outputs[meta.stride_index + STRIDES.len()].as_slice();
    let bbox_slice = outputs[meta.stride_index + STRIDES.len() * 2].as_slice();
    let kps_slice = outputs[meta.stride_index + STRIDES.len() * 3].as_slice();

    anyhow::ensure!(
        cls_slice.len() == meta.cell_count,
        "cls length mismatch: expected {}, got {}",
        meta.cell_count,
        cls_slice.len()
    );
    anyhow::ensure!(
        obj_slice.len() == meta.cell_count,
        "obj length mismatch: expected {}, got {}",
        meta.cell_count,
        obj_slice.len()
    );
    anyhow::ensure!(
        bbox_slice.len() == meta.cell_count * 4,
        "bbox length mismatch: expected {}, got {}",
        meta.cell_count * 4,
        bbox_slice.len()
    );
    anyhow::ensure!(
        kps_slice.len() == meta.cell_count * 10,
        "kps length mismatch: expected {}, got {}",
        meta.cell_count * 10,
        kps_slice.len()
    );

    Ok(StrideOutputs {
        cls: cls_slice,
        obj: obj_slice,
        bbox: bbox_slice,
        kps: kps_slice,
    })
}

fn decode_stride_cell(input: CellDecodeInput) -> [f32; DETECTION_OUTPUT_COLS] {
    let CellDecodeInput {
        row,
        col,
        stride_f,
        cls_score,
        obj_score,
        bbox,
        kps,
    } = input;

    let cls_score = cls_score.clamp(0.0, 1.0);
    let obj_score = obj_score.clamp(0.0, 1.0);
    let mut score = (cls_score * obj_score).sqrt();
    if !score.is_finite() {
        score = 0.0;
    }

    let cx = (col as f32 + bbox[0]) * stride_f;
    let cy = (row as f32 + bbox[1]) * stride_f;
    let w = bbox[2].exp() * stride_f;
    let h = bbox[3].exp() * stride_f;
    let x = (-0.5f32).mul_add(w, cx);
    let y = (-0.5f32).mul_add(h, cy);

    let mut row_out = [0.0f32; DETECTION_OUTPUT_COLS];
    row_out[0] = x;
    row_out[1] = y;
    row_out[2] = w;
    row_out[3] = h;

    let mut write_kps = 4;
    for lm in 0..5 {
        row_out[write_kps] = (kps[lm * 2] + col as f32) * stride_f;
        row_out[write_kps + 1] = (kps[lm * 2 + 1] + row as f32) * stride_f;
        write_kps += 2;
    }

    row_out[DETECTION_SCORE_INDEX] = score;
    row_out
}

fn decode_stride_outputs(
    outputs: &[Tensor],
    meta: &StrideMeta,
    layout: HeadLayout,
    gate: Option<ScoreGate>,
    dst: &mut [f32],
) -> Result<()> {
    let s = validate_stride_outputs(outputs, meta)?;
    // Both arms are the same loop over the same cells; only where each channel lives in
    // the buffer differs, and whether the scores still need sigmoid. Branching here rather
    // than inside the loop keeps the indexing constant-folded for 8400 cells.
    match layout {
        HeadLayout::CellMajorActivated => decode_cells(
            meta,
            dst,
            |c, cell, channels| cell * channels + c,
            false,
            gate,
            &s,
        ),
        HeadLayout::ChannelMajorLogits => decode_cells(
            meta,
            dst,
            |c, cell, _| c * meta.cell_count + cell,
            true,
            gate,
            &s,
        ),
    }
    Ok(())
}

/// Walk every cell of one stride, gathering its channels through `index`.
///
/// `index(channel, cell, channels)` is where that channel's value for that cell lives,
/// which is the only thing the two head layouts disagree about.
fn decode_cells(
    meta: &StrideMeta,
    dst: &mut [f32],
    index: impl Fn(usize, usize, usize) -> usize,
    activate: bool,
    gate: Option<ScoreGate>,
    s: &StrideOutputs<'_>,
) {
    let stride_f = meta.stride as f32;
    let score = |v: f32| if activate { sigmoid(v) } else { v };
    let mut write = 0;

    for row in 0..meta.rows {
        for col in 0..meta.cols {
            let cell = row * meta.cols + col;
            // A rejected cell leaves its row zeroed: score 0 fails the threshold and a
            // zero width fails the size check, so postprocessing drops it on either.
            if gate.is_some_and(|g| g.rejects(s.cls[cell], s.obj[cell])) {
                dst[write..write + DETECTION_OUTPUT_COLS].fill(0.0);
                write += DETECTION_OUTPUT_COLS;
                continue;
            }
            let decoded = decode_stride_cell(CellDecodeInput {
                row,
                col,
                stride_f,
                cls_score: score(s.cls[cell]),
                obj_score: score(s.obj[cell]),
                bbox: std::array::from_fn(|c| s.bbox[index(c, cell, 4)]),
                kps: std::array::from_fn(|c| s.kps[index(c, cell, 10)]),
            });
            dst[write..write + DETECTION_OUTPUT_COLS].copy_from_slice(&decoded);
            write += DETECTION_OUTPUT_COLS;
        }
    }
}

fn align_to(value: usize, divisor: usize) -> usize {
    assert!(divisor > 0, "divisor must be non-zero");
    value.div_ceil(divisor) * divisor
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn loading_missing_model_fails() {
        let result = YuNetModel::load("missing.onnx", InputSize::default());
        assert!(result.is_err());
    }

    #[test]
    fn invalid_model_produces_useful_error() {
        let mut temp = NamedTempFile::new().expect("temp file");
        temp.write_all(b"not a real onnx file")
            .expect("write mock model");

        let err = YuNetModel::load(temp.path(), InputSize::default())
            .expect_err("invalid ONNX should fail");

        // Assert the intent rather than one backend's wording: the failure must
        // name the file and explain what was expected, and the chain must still
        // carry the underlying decode error so "corrupt file" is
        // distinguishable from "valid ONNX that is not YuNet".
        let top = format!("{err}");
        assert!(
            top.contains(&temp.path().display().to_string()),
            "error should name the offending file: {top}"
        );
        let chain = format!("{err:#}");
        assert!(
            chain.to_lowercase().contains("onnx") || chain.to_lowercase().contains("decode"),
            "error chain should explain what failed to load: {chain}"
        );
    }

    // --- align_to ---

    #[test]
    fn align_to_already_aligned_is_unchanged() {
        assert_eq!(align_to(32, 32), 32);
        assert_eq!(align_to(64, 32), 64);
        assert_eq!(align_to(640, 32), 640);
    }

    #[test]
    fn align_to_rounds_up_to_next_multiple() {
        assert_eq!(align_to(1, 32), 32);
        assert_eq!(align_to(33, 32), 64);
        assert_eq!(align_to(31, 32), 32);
    }

    #[test]
    fn build_stride_layout_totals_match_expected_grid_sizes() {
        let layout = build_stride_layout(InputSize {
            width: 64,
            height: 64,
        })
        .expect("layout should be valid");

        assert_eq!(layout.metas.len(), STRIDES.len());
        assert_eq!(layout.total_capacity, 84 * DETECTION_OUTPUT_COLS);

        assert_eq!(layout.metas[0].stride, 8);
        assert_eq!(layout.metas[0].cell_count, 64);
        assert_eq!(layout.metas[1].cell_count, 16);
        assert_eq!(layout.metas[2].cell_count, 4);
    }

    // --- head layouts ---

    /// The two head layouts must decode to exactly the same rows.
    ///
    /// The GPU path decodes straight from channel-major planes with raw logits, where
    /// every other backend hands over cell-major rows with activated scores. Transposing
    /// and activating the same data by hand and decoding it the other way is the check
    /// that the channel-major indexing -- `channel * cells + cell`, easy to get backwards
    /// and silent when wrong -- still lines up.
    #[test]
    fn channel_major_logits_decode_like_cell_major_activated() {
        let size = InputSize {
            width: 64,
            height: 64,
        };
        let strides = build_stride_layout(size).expect("build stride layout");

        // Channel-major, raw logits: what the GPU heads produce.
        let mut chw: Vec<Tensor> = Vec::new();
        for channels in [1usize, 1, 4, 10] {
            for meta in strides.metas.iter() {
                let n = meta.cell_count * channels;
                // Spread over a range where sigmoid is not saturated, so an activation
                // applied to the wrong element changes the result.
                let data: Vec<f32> = (0..n).map(|i| ((i % 23) as f32 * 0.3) - 3.0).collect();
                chw.push(Tensor::from_vec(&[channels, meta.cell_count], data).expect("chw head"));
            }
        }

        // The same values transposed to cell-major, with cls and obj pre-activated.
        let cell_major: Vec<Tensor> = chw
            .iter()
            .enumerate()
            .map(|(i, plane)| {
                let channels = plane.shape()[0];
                let cells = plane.shape()[1];
                let src = plane.as_slice();
                let activate = i < STRIDES.len() * 2; // cls and obj come first
                let mut out = vec![0f32; channels * cells];
                for c in 0..channels {
                    for cell in 0..cells {
                        let v = src[c * cells + cell];
                        out[cell * channels + c] = if activate { sigmoid(v) } else { v };
                    }
                }
                Tensor::from_vec(&[cells, channels], out).expect("cell-major head")
            })
            .collect();

        let from_chw = decode_yunet_outputs_with(&chw, size, HeadLayout::ChannelMajorLogits, None)
            .expect("decode channel-major");
        let from_rows =
            decode_yunet_outputs_with(&cell_major, size, HeadLayout::CellMajorActivated, None)
                .expect("decode cell-major");

        assert_eq!(from_chw.shape(), from_rows.shape());
        for (i, (a, b)) in from_chw
            .as_slice()
            .iter()
            .zip(from_rows.as_slice())
            .enumerate()
        {
            assert!(
                (a - b).abs() < 1e-6,
                "element {i} differs between head layouts: {a} vs {b}"
            );
        }
    }

    // --- sigmoid ---

    #[test]
    fn sigmoid_zero_is_half() {
        assert!((sigmoid(0.0) - 0.5).abs() < 1e-6);
    }

    #[test]
    fn sigmoid_large_positive_approaches_one() {
        assert!(sigmoid(100.0) > 0.9999);
    }

    #[test]
    fn sigmoid_large_negative_approaches_zero() {
        assert!(sigmoid(-100.0) < 0.0001);
    }

    // --- decode_yunet_outputs ---

    fn mock_outputs(input_size: InputSize) -> Vec<Tensor> {
        let input_w = align_to(input_size.width as usize, 32);
        let input_h = align_to(input_size.height as usize, 32);
        let mut cls_t = Vec::new();
        let mut obj_t = Vec::new();
        let mut bbox_t = Vec::new();
        let mut kps_t = Vec::new();

        for &stride in STRIDES.iter() {
            let cols = input_w / stride;
            let rows = input_h / stride;
            let n = rows * cols;
            cls_t.push(Tensor::from_shape(&[n], &vec![0.9f32; n]).expect("cls tensor"));
            obj_t.push(Tensor::from_shape(&[n], &vec![0.8f32; n]).expect("obj tensor"));
            bbox_t.push(Tensor::from_shape(&[n, 4], &vec![0.0f32; n * 4]).expect("bbox tensor"));
            kps_t.push(Tensor::from_shape(&[n, 10], &vec![0.0f32; n * 10]).expect("kps tensor"));
        }

        cls_t
            .into_iter()
            .chain(obj_t)
            .chain(bbox_t)
            .chain(kps_t)
            .collect()
    }

    #[test]
    fn decode_yunet_outputs_produces_n_by_15_tensor() {
        let size = InputSize {
            width: 128,
            height: 128,
        };
        let outputs = mock_outputs(size);
        let result = decode_yunet_outputs(&outputs, size).expect("decode should succeed");
        assert_eq!(result.shape().len(), 2);
        assert_eq!(result.shape()[1], DETECTION_OUTPUT_COLS);
        // Verify scores are in [0, 1]
        let data = result.as_slice();
        for score in data
            .iter()
            .skip(DETECTION_SCORE_INDEX)
            .step_by(DETECTION_OUTPUT_COLS)
        {
            assert!(
                *score >= 0.0 && *score <= 1.0,
                "score out of range: {score}"
            );
        }
    }

    #[test]
    fn decode_yunet_outputs_errors_for_wrong_tensor_count() {
        let size = InputSize {
            width: 128,
            height: 128,
        };
        assert!(decode_yunet_outputs(&[], size).is_err());
        // Partial count also errors
        let outputs = mock_outputs(size);
        assert!(decode_yunet_outputs(&outputs[..3], size).is_err());
    }

    #[test]
    fn decode_yunet_outputs_cell_count_matches_grid_dimensions() {
        let size = InputSize {
            width: 64,
            height: 64,
        };
        let outputs = mock_outputs(size);
        let result = decode_yunet_outputs(&outputs, size).expect("decode should succeed");
        // stride 8 → 8×8=64 cells, stride 16 → 4×4=16, stride 32 → 2×2=4 → total 84 rows
        assert_eq!(result.shape()[0], 84);
    }

    #[test]
    fn validate_stride_outputs_rejects_malformed_tensor_lengths() {
        let size = InputSize {
            width: 64,
            height: 64,
        };
        let layout = build_stride_layout(size).expect("build stride layout");
        let mut outputs = mock_outputs(size);
        outputs[0] = Tensor::from_shape(&[layout.metas[0].cell_count - 1], &vec![0.0f32; 63])
            .expect("build malformed cls tensor");

        let err = validate_stride_outputs(&outputs, &layout.metas[0])
            .unwrap_err()
            .to_string();
        assert!(err.contains("cls length mismatch"));
    }

    /// The gate must never change a row that postprocessing would have kept. It is allowed
    /// to zero rows below the threshold -- that is the whole point -- so this asserts the
    /// exact invariant rather than equality of the whole tensor.
    #[test]
    fn score_gate_preserves_every_row_above_the_threshold() {
        let size = InputSize {
            width: 640,
            height: 640,
        };
        let threshold = 0.6f32;
        for layout in [
            HeadLayout::ChannelMajorLogits,
            HeadLayout::CellMajorActivated,
        ] {
            let outputs = gate_test_outputs(size, layout);
            let full = decode_yunet_outputs_with(&outputs, size, layout, None).expect("full");
            let gated =
                decode_yunet_outputs_with(&outputs, size, layout, Some(threshold)).expect("gated");
            let full = full.as_slice();
            let gated = gated.as_slice();
            assert_eq!(full.len(), gated.len());

            let mut kept = 0usize;
            for (a, b) in full
                .chunks_exact(DETECTION_OUTPUT_COLS)
                .zip(gated.chunks_exact(DETECTION_OUTPUT_COLS))
            {
                if a[DETECTION_SCORE_INDEX] >= threshold {
                    kept += 1;
                    assert_eq!(a, b, "gate changed a row that would have been kept");
                } else {
                    assert!(
                        b[DETECTION_SCORE_INDEX] < threshold,
                        "gate produced a keepable row where the full decode had none"
                    );
                }
            }
            assert!(
                kept > 0,
                "{layout:?}: no row cleared the threshold, so the test proved nothing"
            );
        }
    }

    /// Scores spread either side of the threshold so the test sees both branches of the gate.
    fn gate_test_outputs(size: InputSize, layout: HeadLayout) -> Vec<Tensor> {
        let strides = build_stride_layout(size).expect("layout");
        let mut outputs = Vec::with_capacity(STRIDES.len() * OUTPUTS_PER_STRIDE);
        for channels in [1usize, 1, 4, 10] {
            for meta in strides.metas.iter() {
                let n = meta.cell_count;
                let values: Vec<f32> = (0..channels * n)
                    .map(|i| {
                        let spread = ((i * 61 % 97) as f32 - 40.0) / 12.0;
                        match layout {
                            // Probabilities for the activated layout, logits for the other.
                            HeadLayout::CellMajorActivated if channels == 1 => sigmoid(spread),
                            _ => spread,
                        }
                    })
                    .collect();
                outputs.push(
                    Tensor::from_vec(&[channels, n], values).expect("build gate test tensor"),
                );
            }
        }
        outputs
    }

    #[test]
    fn decode_stride_cell_nan_score_becomes_zero() {
        let decoded = decode_stride_cell(CellDecodeInput {
            row: 0,
            col: 0,
            stride_f: 8.0,
            cls_score: f32::NAN,
            obj_score: 0.9,
            bbox: [0.0; 4],
            kps: [0.0; 10],
        });
        assert_eq!(decoded[DETECTION_SCORE_INDEX], 0.0);
    }

    #[test]
    fn decode_stride_cell_scores_above_one_are_clamped() {
        let decoded = decode_stride_cell(CellDecodeInput {
            row: 0,
            col: 0,
            stride_f: 8.0,
            cls_score: 2.0,
            obj_score: 2.0,
            bbox: [0.0; 4],
            kps: [0.0; 10],
        });
        // clamped to 1.0 each → sqrt(1.0 * 1.0) = 1.0
        assert!((decoded[DETECTION_SCORE_INDEX] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn validate_stride_outputs_rejects_wrong_bbox_length() {
        let size = InputSize {
            width: 64,
            height: 64,
        };
        let layout = build_stride_layout(size).expect("build stride layout");
        let mut outputs = mock_outputs(size);
        let n = layout.metas[0].cell_count;
        // bbox should be n*4 elements; give n*3 instead
        outputs[STRIDES.len() * 2] =
            Tensor::from_shape(&[n, 3], &vec![0.0f32; n * 3]).expect("bad bbox tensor");

        let err = validate_stride_outputs(&outputs, &layout.metas[0])
            .unwrap_err()
            .to_string();
        assert!(err.contains("bbox length mismatch"), "got: {err}");
    }

    #[test]
    fn validate_stride_outputs_rejects_wrong_kps_length() {
        let size = InputSize {
            width: 64,
            height: 64,
        };
        let layout = build_stride_layout(size).expect("build stride layout");
        let mut outputs = mock_outputs(size);
        let n = layout.metas[0].cell_count;
        // kps should be n*10 elements; give n*5 instead
        outputs[STRIDES.len() * 3] =
            Tensor::from_shape(&[n, 5], &vec![0.0f32; n * 5]).expect("bad kps tensor");

        let err = validate_stride_outputs(&outputs, &layout.metas[0])
            .unwrap_err()
            .to_string();
        assert!(err.contains("kps length mismatch"), "got: {err}");
    }

    #[test]
    fn validate_stride_outputs_rejects_wrong_obj_length() {
        let size = InputSize {
            width: 64,
            height: 64,
        };
        let layout = build_stride_layout(size).expect("build stride layout");
        let mut outputs = mock_outputs(size);
        let n = layout.metas[0].cell_count;
        outputs[STRIDES.len()] =
            Tensor::from_shape(&[n - 1], &vec![0.0f32; n - 1]).expect("bad obj tensor");

        let err = validate_stride_outputs(&outputs, &layout.metas[0])
            .unwrap_err()
            .to_string();
        assert!(err.contains("obj length mismatch"), "got: {err}");
    }

    #[test]
    fn decode_stride_cell_matches_expected_row_values() {
        let decoded = decode_stride_cell(CellDecodeInput {
            row: 1,
            col: 2,
            stride_f: 8.0,
            cls_score: 1.2,
            obj_score: 0.64,
            bbox: [0.25, 0.5, 0.0, 0.0],
            kps: [0.0; 10],
        });

        assert!((decoded[0] - 14.0).abs() < f32::EPSILON);
        assert!((decoded[1] - 8.0).abs() < f32::EPSILON);
        assert!((decoded[2] - 8.0).abs() < f32::EPSILON);
        assert!((decoded[3] - 8.0).abs() < f32::EPSILON);
        assert!((decoded[4] - 16.0).abs() < f32::EPSILON);
        assert!((decoded[5] - 8.0).abs() < f32::EPSILON);
        assert!((decoded[DETECTION_SCORE_INDEX] - 0.8).abs() < f32::EPSILON);
    }
}

#[cfg(test)]
mod benches {
    use super::*;
    use anyhow::Result;
    use rayon::prelude::*;
    use std::time::Instant;

    fn decode_yunet_outputs_baseline(outputs: &[Tensor], input_size: InputSize) -> Result<Tensor> {
        let input_w = input_size.width as usize;
        let input_h = input_size.height as usize;
        let pad_w = align_to(input_w, STRIDE_ALIGNMENT);
        let pad_h = align_to(input_h, STRIDE_ALIGNMENT);

        let mut total_cells = 0;
        for &stride in STRIDES.iter() {
            anyhow::ensure!(
                pad_w
                    .checked_rem(stride)
                    .is_some_and(|remainder| remainder == 0),
                "input width not divisible by stride {}",
                stride
            );
            anyhow::ensure!(
                pad_h
                    .checked_rem(stride)
                    .is_some_and(|remainder| remainder == 0),
                "input height not divisible by stride {}",
                stride
            );
            total_cells += (pad_w * pad_h) / (stride * stride);
        }
        let total_capacity = total_cells * DETECTION_OUTPUT_COLS;

        let stride_results: Result<Vec<Vec<f32>>> = STRIDES
            .par_iter()
            .enumerate()
            .map(|(stride_index, &stride)| -> Result<Vec<f32>> {
                let cols = pad_w / stride;
                let rows = pad_h / stride;
                let cell_count = rows * cols;
                let stride_f = stride as f32;

                let cls_slice = outputs[stride_index].as_slice();
                let obj_slice = outputs[stride_index + STRIDES.len()].as_slice();
                let bbox_slice = outputs[stride_index + STRIDES.len() * 2].as_slice();
                let kps_slice = outputs[stride_index + STRIDES.len() * 3].as_slice();

                anyhow::ensure!(
                    cls_slice.len() == cell_count,
                    "cls length mismatch: expected {}, got {}",
                    cell_count,
                    cls_slice.len()
                );
                anyhow::ensure!(
                    obj_slice.len() == cell_count,
                    "obj length mismatch: expected {}, got {}",
                    cell_count,
                    obj_slice.len()
                );
                anyhow::ensure!(
                    bbox_slice.len() == cell_count * 4,
                    "bbox length mismatch: expected {}, got {}",
                    cell_count * 4,
                    bbox_slice.len()
                );
                anyhow::ensure!(
                    kps_slice.len() == cell_count * 10,
                    "kps length mismatch: expected {}, got {}",
                    cell_count * 10,
                    kps_slice.len()
                );

                let mut stride_data = Vec::with_capacity(cell_count * DETECTION_OUTPUT_COLS);

                for row in 0..rows {
                    for col in 0..cols {
                        let idx = row * cols + col;
                        let cls_score = cls_slice[idx].clamp(0.0, 1.0);
                        let obj_score = obj_slice[idx].clamp(0.0, 1.0);
                        let mut score = (cls_score * obj_score).sqrt();
                        if !score.is_finite() {
                            score = 0.0;
                        }

                        let bbox_offset = idx * 4;
                        let dx = bbox_slice[bbox_offset];
                        let dy = bbox_slice[bbox_offset + 1];
                        let dw = bbox_slice[bbox_offset + 2];
                        let dh = bbox_slice[bbox_offset + 3];

                        let cx = (col as f32 + dx) * stride_f;
                        let cy = (row as f32 + dy) * stride_f;
                        let w = dw.exp() * stride_f;
                        let h = dh.exp() * stride_f;
                        let x = (-0.5f32).mul_add(w, cx);
                        let y = (-0.5f32).mul_add(h, cy);

                        stride_data.push(x);
                        stride_data.push(y);
                        stride_data.push(w);
                        stride_data.push(h);

                        let kps_offset = idx * 10;
                        for lm in 0..5 {
                            let lx = (kps_slice[kps_offset + lm * 2] + col as f32) * stride_f;
                            let ly = (kps_slice[kps_offset + lm * 2 + 1] + row as f32) * stride_f;
                            stride_data.push(lx);
                            stride_data.push(ly);
                        }

                        stride_data.push(score);
                    }
                }

                Ok(stride_data)
            })
            .collect();

        let stride_vecs = stride_results?;
        let mut fused = Vec::with_capacity(total_capacity);
        for vec in stride_vecs {
            fused.extend_from_slice(&vec);
        }

        let rows = fused.len() / DETECTION_OUTPUT_COLS;
        Tensor::from_shape(&[rows, DETECTION_OUTPUT_COLS], &fused)
            .map_err(|e| anyhow::anyhow!("failed to build fused YuNet tensor: {e}"))
    }

    fn mock_outputs(input_size: InputSize) -> Vec<Tensor> {
        let input_w = align_to(input_size.width as usize, 32);
        let input_h = align_to(input_size.height as usize, 32);
        let mut cls_tensors = Vec::new();
        let mut obj_tensors = Vec::new();
        let mut bbox_tensors = Vec::new();
        let mut kps_tensors = Vec::new();

        for &stride in STRIDES.iter() {
            let cols = input_w / stride;
            let rows = input_h / stride;
            let cell_count = cols * rows;

            let cls = vec![0.9f32; cell_count];
            let obj = vec![0.8f32; cell_count];
            let bbox: Vec<f32> = (0..cell_count * 4)
                .map(|idx| ((idx % 11) as f32 * 0.01) - 0.05)
                .collect();
            let kps: Vec<f32> = (0..cell_count * 10)
                .map(|idx| ((idx % 7) as f32 * 0.015) - 0.05)
                .collect();

            cls_tensors.push(Tensor::from_shape(&[cell_count], &cls).expect("cls tensor"));
            obj_tensors.push(Tensor::from_shape(&[cell_count], &obj).expect("obj tensor"));
            bbox_tensors.push(Tensor::from_shape(&[cell_count, 4], &bbox).expect("bbox tensor"));
            kps_tensors.push(Tensor::from_shape(&[cell_count, 10], &kps).expect("kps tensor"));
        }

        cls_tensors
            .into_iter()
            .chain(obj_tensors)
            .chain(bbox_tensors)
            .chain(kps_tensors)
            .collect()
    }

    #[test]
    #[ignore]
    fn bench_decode_yunet_outputs() {
        let input_size = InputSize {
            width: 320,
            height: 320,
        };
        let tensors = mock_outputs(input_size);

        for _ in 0..3 {
            let _ = decode_yunet_outputs(&tensors, input_size).expect("warmup decode");
            let _ = decode_yunet_outputs_baseline(&tensors, input_size).expect("warmup baseline");
        }

        let iterations = 20;
        let start_new = Instant::now();
        for _ in 0..iterations {
            let _ = decode_yunet_outputs(&tensors, input_size).expect("bench decode");
        }
        let new_time = start_new.elapsed();

        let start_old = Instant::now();
        for _ in 0..iterations {
            let _ = decode_yunet_outputs_baseline(&tensors, input_size).expect("bench baseline");
        }
        let old_time = start_old.elapsed();

        println!(
            "decode_yunet_outputs optimized avg: {:?}, baseline avg: {:?}",
            new_time / iterations,
            old_time / iterations
        );
    }
}
