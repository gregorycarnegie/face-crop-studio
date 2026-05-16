use crate::{
    model_config::{
        DETECTION_OUTPUT_COLS, DETECTION_SCORE_INDEX, OUTPUTS_PER_STRIDE, STRIDE_ALIGNMENT, STRIDES,
    },
    preprocess::InputSize,
};

use anyhow::{Context, Result};
use log::{debug, warn};
use std::{fmt::Write, path::Path};
use tract_onnx::prelude::{
    Framework, Graph, InferenceModelExt, IntoTensor, SimplePlan, Tensor, TypedFact, TypedOp, tvec,
};

type RunnableModel = SimplePlan<TypedFact, Box<dyn TypedOp>, Graph<TypedFact, Box<dyn TypedOp>>>;

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
#[derive(Debug)]
pub struct YuNetModel {
    runnable: RunnableModel,
    input_size: InputSize,
}

impl YuNetModel {
    /// Load and optimize the YuNet ONNX graph for a specific input size.
    pub fn load<P: AsRef<Path>>(model_path: P, input_size: InputSize) -> Result<Self> {
        let path = model_path.as_ref();
        anyhow::ensure!(path.exists(), "model file not found: {}", path.display());

        let runnable = match load_runnable_model(path, input_size, true) {
            Ok(model) => {
                debug!(
                    "YuNet model {} optimized successfully ({}x{})",
                    path.display(),
                    input_size.width,
                    input_size.height
                );
                model
            }
            Err(opt_err) => {
                let optimize_msg = format!("{opt_err}");
                let mut chain_msg = String::new();
                for cause in opt_err.chain() {
                    let _ = writeln!(&mut chain_msg, "  • {cause}");
                }
                warn!(
                    "YuNet model {} failed optimized load ({}); falling back to decluttered graph (~2x slower).\nError chain:\n{}",
                    path.display(),
                    optimize_msg,
                    chain_msg.trim_end()
                );
                let decluttered =
                    load_runnable_model(path, input_size, false).with_context(|| {
                    format!(
                        "fallback to decluttered YuNet graph failed after optimize error: {optimize_msg}"
                    )
                })?;
                debug!(
                    "YuNet model {} running in decluttered mode ({}x{})",
                    path.display(),
                    input_size.width,
                    input_size.height
                );
                decluttered
            }
        };

        Ok(Self {
            runnable,
            input_size,
        })
    }

    /// Execute YuNet with a preprocessed tensor and return decoded detections.
    ///
    /// The resulting tensor has shape `[N, 15]` where each row is
    /// `[x, y, w, h, re_x, re_y, le_x, le_y, nt_x, nt_y, rcm_x, rcm_y, lcm_x, lcm_y, score]`
    /// in the resized input coordinate space.
    pub fn run(&self, input: Tensor) -> Result<Tensor> {
        let outputs = self
            .runnable
            .run(tvec![input.into()])
            .map_err(|e| anyhow::anyhow!("YuNet execution failed: {e}"))?;

        let mut tensors: Vec<Tensor> = outputs
            .into_iter()
            .map(|value| value.into_tensor())
            .collect();

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

    pub fn input_size(&self) -> InputSize {
        self.input_size
    }
}

fn load_runnable_model(
    path: &Path,
    _input_size: InputSize,
    optimized: bool,
) -> Result<RunnableModel> {
    // Load model and let it infer shape from ONNX file
    // The input_size parameter is used for preprocessing and coordinate scaling,
    // but the model itself should match the ONNX file's declared input shape
    let model = tract_onnx::onnx()
        .model_for_path(path)
        .with_context(|| format!("failed to parse ONNX graph from {}", path.display()))?;

    if optimized {
        model
            .into_optimized()
            .map_err(|e| anyhow::anyhow!("unable to optimize YuNet graph: {e}"))?
            .into_runnable()
            .map_err(|e| anyhow::anyhow!("unable to make YuNet graph runnable: {e}"))
    } else {
        model
            .into_typed()
            .map_err(|e| anyhow::anyhow!("unable to type-check YuNet graph: {e}"))?
            .into_decluttered()
            .map_err(|e| anyhow::anyhow!("unable to declutter YuNet graph: {e}"))?
            .into_runnable()
            .map_err(|e| anyhow::anyhow!("unable to make YuNet graph runnable: {e}"))
    }
}

pub(crate) fn decode_yunet_outputs(outputs: &[Tensor], input_size: InputSize) -> Result<Tensor> {
    anyhow::ensure!(
        outputs.len() == STRIDES.len() * OUTPUTS_PER_STRIDE,
        "YuNet decode expects {} tensors, got {}",
        STRIDES.len() * OUTPUTS_PER_STRIDE,
        outputs.len()
    );

    let layout = build_stride_layout(input_size)?;
    let mut fused = vec![0f32; layout.total_capacity];

    for meta in layout.metas.iter() {
        let dst = &mut fused[meta.offset..meta.offset + meta.cell_count * DETECTION_OUTPUT_COLS];
        decode_stride_outputs(outputs, meta, dst)?;
    }

    let rows = fused.len() / DETECTION_OUTPUT_COLS;
    Tensor::from_shape(&[rows, DETECTION_OUTPUT_COLS], &fused)
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
    let cls_slice = outputs[meta.stride_index]
        .as_slice::<f32>()
        .map_err(|e| anyhow::anyhow!("cls output not f32: {e}"))?;
    let obj_slice = outputs[meta.stride_index + STRIDES.len()]
        .as_slice::<f32>()
        .map_err(|e| anyhow::anyhow!("obj output not f32: {e}"))?;
    let bbox_slice = outputs[meta.stride_index + STRIDES.len() * 2]
        .as_slice::<f32>()
        .map_err(|e| anyhow::anyhow!("bbox output not f32: {e}"))?;
    let kps_slice = outputs[meta.stride_index + STRIDES.len() * 3]
        .as_slice::<f32>()
        .map_err(|e| anyhow::anyhow!("kps output not f32: {e}"))?;

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

fn decode_stride_outputs(outputs: &[Tensor], meta: &StrideMeta, dst: &mut [f32]) -> Result<()> {
    let stride_outputs = validate_stride_outputs(outputs, meta)?;
    let stride_f = meta.stride as f32;
    let mut write = 0;

    for row in 0..meta.rows {
        for col in 0..meta.cols {
            let idx = row * meta.cols + col;
            let bbox_offset = idx * 4;
            let kps_offset = idx * 10;
            let decoded = decode_stride_cell(CellDecodeInput {
                row,
                col,
                stride_f,
                cls_score: stride_outputs.cls[idx],
                obj_score: stride_outputs.obj[idx],
                bbox: [
                    stride_outputs.bbox[bbox_offset],
                    stride_outputs.bbox[bbox_offset + 1],
                    stride_outputs.bbox[bbox_offset + 2],
                    stride_outputs.bbox[bbox_offset + 3],
                ],
                kps: [
                    stride_outputs.kps[kps_offset],
                    stride_outputs.kps[kps_offset + 1],
                    stride_outputs.kps[kps_offset + 2],
                    stride_outputs.kps[kps_offset + 3],
                    stride_outputs.kps[kps_offset + 4],
                    stride_outputs.kps[kps_offset + 5],
                    stride_outputs.kps[kps_offset + 6],
                    stride_outputs.kps[kps_offset + 7],
                    stride_outputs.kps[kps_offset + 8],
                    stride_outputs.kps[kps_offset + 9],
                ],
            });
            dst[write..write + DETECTION_OUTPUT_COLS].copy_from_slice(&decoded);
            write += DETECTION_OUTPUT_COLS;
        }
    }

    Ok(())
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
        let message = format!("{err}");
        assert!(
            message.contains("failed to parse ONNX") || message.contains("unable to optimize"),
            "Unexpected error message: {message}"
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
        let data = result.as_slice::<f32>().expect("output tensor is f32");
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

                let cls_slice = outputs[stride_index]
                    .as_slice::<f32>()
                    .map_err(|e| anyhow::anyhow!("cls output not f32: {e}"))?;
                let obj_slice = outputs[stride_index + STRIDES.len()]
                    .as_slice::<f32>()
                    .map_err(|e| anyhow::anyhow!("obj output not f32: {e}"))?;
                let bbox_slice = outputs[stride_index + STRIDES.len() * 2]
                    .as_slice::<f32>()
                    .map_err(|e| anyhow::anyhow!("bbox output not f32: {e}"))?;
                let kps_slice = outputs[stride_index + STRIDES.len() * 3]
                    .as_slice::<f32>()
                    .map_err(|e| anyhow::anyhow!("kps output not f32: {e}"))?;

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
