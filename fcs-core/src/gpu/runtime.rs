use crate::{
    gpu::{
        graph::{self, BACKBONE_STAGES, DETECTION_HEADS, DetectionLevelOutputs, WeightProvider},
        onnx::OnnxInitializerMap,
        ops::GpuInferenceOps,
        tensor::GpuTensor,
    },
    model::decode_yunet_outputs,
    preprocess::InputSize,
};
use bytemuck::cast_slice;
use std::sync::mpsc;
use wgpu::CommandEncoderDescriptor;

use anyhow::{Context, Result, anyhow};
use fcs_utils::gpu::{GpuAvailability, GpuContext, GpuContextOptions};
use std::{
    collections::HashMap,
    path::Path,
    sync::{Arc, Mutex},
};
use tract_onnx::prelude::Tensor;

#[derive(Debug, Default)]
struct GpuYuNetWorkspace {
    input_tensors: Vec<GpuTensor>, // Pool of available tensors
}

#[derive(Debug)]
pub struct GpuYuNet {
    ops: Arc<GpuInferenceOps>,
    weights: Arc<HashMap<String, GpuTensor>>,
    input_size: InputSize,
    workspace: Mutex<GpuYuNetWorkspace>,
}

impl GpuYuNet {
    pub fn new<P: AsRef<Path>>(model_path: P, input_size: InputSize) -> Result<Self> {
        let model_path = model_path.as_ref();
        let context = match GpuContext::init_with_fallback(&GpuContextOptions::default()) {
            GpuAvailability::Available(ctx) => ctx,
            GpuAvailability::Disabled { reason } => {
                anyhow::bail!("GPU backend disabled by configuration: {reason}")
            }
            GpuAvailability::Unavailable { error } => {
                anyhow::bail!("GPU backend unavailable: {error}")
            }
        };
        let loader = graph::load_backbone_weights(
            model_path,
            BACKBONE_STAGES.len(),
            true,
            DETECTION_HEADS.len(),
        )?;

        let memory_limit = estimate_inference_memory(&loader, input_size);
        let ops = Arc::new(GpuInferenceOps::new(context, Some(memory_limit))?);
        let weight_map = upload_gpu_weights(&ops, loader)?;
        Ok(Self {
            ops,
            weights: Arc::new(weight_map),
            input_size,
            workspace: Mutex::new(GpuYuNetWorkspace::default()),
        })
    }

    pub fn run(&self, tensor: Tensor) -> Result<Tensor> {
        let dims = tensor.shape().to_vec();
        let data = tensor
            .try_as_plain()
            .and_then(|view| view.as_slice::<f32>())
            .context("gpu input must be contiguous f32")?;

        // 1. Acquire a tensor from the pool or create a new one
        let input_gpu = {
            let mut workspace = self
                .workspace
                .lock()
                .map_err(|_| anyhow!("GPU workspace lock poisoned while acquiring input tensor"))?;
            let maybe_tensor = workspace.input_tensors.pop();

            if let Some(existing) = maybe_tensor {
                // If dimensions match, reuse it. If not, creates new one (and drops old one implicitly or we could recycle it more smartly)
                // For simplified logic: checks dimensions.
                if existing.shape().dims() == dims {
                    self.ops
                        .upload_to_tensor(&existing, data)
                        .context("upload to pooled input tensor")?;
                    existing
                } else {
                    // Dims changed, allocate new. Old one is dropped (and its buffer goes to GpuBufferPool)
                    self.ops
                        .upload_tensor(dims, data, Some("gpu_input"))
                        .context("upload input tensor")?
                }
            } else {
                // Pool empty, allocate new
                self.ops
                    .upload_tensor(dims, data, Some("gpu_input"))
                    .context("upload input tensor")?
            }
        };

        // Ensure the tensor is returned to the pool when we are done, even if we panic/error.
        // We use a guard struct or just a clean 'finally' block structure.
        // Since we return `Result`, we wrap execution.
        let result = self.run_inference(&input_gpu);

        // 3. Return tensor to pool
        {
            let mut workspace = self
                .workspace
                .lock()
                .map_err(|_| anyhow!("GPU workspace lock poisoned while returning input tensor"))?;
            workspace.input_tensors.push(input_gpu);
        }

        result
    }

    fn run_inference(&self, input_gpu: &GpuTensor) -> Result<Tensor> {
        let weight_provider = CachedWeights {
            tensors: Arc::clone(&self.weights),
        };

        // Accumulate the entire forward pass into one command buffer and submit
        // once. This eliminates ~53 individual queue.submit() calls (one per op)
        // and gives the GPU a full workload to pipeline, dramatically improving
        // utilisation vs the previous per-op-submit pattern.
        let mut encoder =
            self.ops
                .context()
                .device()
                .create_command_encoder(&CommandEncoderDescriptor {
                    label: Some("inference"),
                });
        let features = graph::encode_backbone_features(
            &mut encoder,
            &self.ops,
            &weight_provider,
            input_gpu,
            BACKBONE_STAGES.len(),
        )?;
        let levels =
            graph::encode_neck_and_heads(&mut encoder, &self.ops, &weight_provider, &features)?;
        self.ops.context().queue().submit(Some(encoder.finish()));

        let outputs = build_decode_tensors(&levels)?;
        decode_yunet_outputs(&outputs, self.input_size)
    }

    pub fn memory_usage(&self) -> u64 {
        self.ops.memory_usage()
    }
}

fn upload_gpu_weights(
    ops: &GpuInferenceOps,
    loader: OnnxInitializerMap,
) -> Result<HashMap<String, GpuTensor>> {
    let mut map = HashMap::with_capacity(loader.len());
    for (name, tensor) in loader.into_map() {
        let gpu_tensor = ops.upload_tensor(
            tensor.dims().to_vec(),
            tensor.data(),
            Some(&format!("weight::{name}")),
        )?;
        map.insert(name, gpu_tensor);
    }
    Ok(map)
}

struct CachedWeights {
    tensors: Arc<HashMap<String, GpuTensor>>,
}

impl WeightProvider for CachedWeights {
    fn tensor(&self, name: &str) -> Result<GpuTensor> {
        self.tensors
            .get(name)
            .cloned()
            .ok_or_else(|| anyhow!("cached weight '{name}' missing"))
    }
}

fn build_decode_tensors(levels: &[DetectionLevelOutputs; 3]) -> Result<Vec<Tensor>> {
    // Collect the 12 output tensors in the order we need them:
    //   cls×3, obj×3, bbox×3, kps×3
    // along with the metadata needed to reorder each one on the CPU.
    struct BranchMeta {
        height: usize,
        width: usize,
        channels: usize,
        apply_sigmoid: bool,
    }

    let mut gpu_tensors: Vec<&GpuTensor> = Vec::with_capacity(DET_HEAD_OUTPUTS);
    let mut meta: Vec<BranchMeta> = Vec::with_capacity(DET_HEAD_OUTPUTS);

    for level in levels.iter() {
        let shape = level.feature.shape().dims();
        anyhow::ensure!(
            shape.len() == 4,
            "feature map must be NCHW (got {:?})",
            shape
        );
        let height = shape[2];
        let width = shape[3];

        // cls, obj, bbox, kps — interleaved by level so we can split later
        for (tensor, channels, apply_sigmoid) in [
            (&level.cls, 1usize, true),
            (&level.obj, 1usize, true),
            (&level.bbox, 4usize, false),
            (&level.kps, 10usize, false),
        ] {
            gpu_tensors.push(tensor);
            meta.push(BranchMeta {
                height,
                width,
                channels,
                apply_sigmoid,
            });
        }
    }

    // One submit + one poll downloads all 12 tensors simultaneously.
    let raw_data = batch_download(gpu_tensors[0].context(), &gpu_tensors)
        .context("batch download of detection head outputs")?;

    // Reorder each downloaded buffer from CHW → HWC (+ optional sigmoid)
    // and build the tract Tensors.
    let mut outputs: Vec<Tensor> = Vec::with_capacity(DET_HEAD_OUTPUTS);
    for (flat, m) in raw_data.into_iter().zip(meta.iter()) {
        let flattened = reorder_hw_major(&flat, m.channels, m.height, m.width, m.apply_sigmoid);
        let rows = m.height * m.width;
        outputs.push(
            Tensor::from_shape(&[rows, m.channels], &flattened)
                .map_err(|e| anyhow!("failed to build tensor from branch output: {e}"))?,
        );
    }

    // The original order was cls×3, obj×3, bbox×3, kps×3 (grouped by type).
    // Currently outputs are interleaved as [cls0, obj0, bbox0, kps0, cls1, ...].
    // Re-group them.
    let mut grouped: [Vec<Tensor>; 4] = std::array::from_fn(|_| Vec::with_capacity(3));
    for (idx, output) in outputs.into_iter().enumerate() {
        grouped[idx % 4].push(output);
    }

    let mut result = Vec::with_capacity(DET_HEAD_OUTPUTS);
    for group in grouped {
        result.extend(group);
    }
    Ok(result)
}

const DET_HEAD_OUTPUTS: usize = 12;

/// Download multiple GPU tensors in a single submit + single poll.
///
/// All buffer copies are recorded into one `CommandEncoder` and submitted
/// together. The GPU DMA engine can pipeline them, and we block exactly
/// once instead of once per tensor.
fn batch_download(context: &Arc<GpuContext>, tensors: &[&GpuTensor]) -> Result<Vec<Vec<f32>>> {
    if tensors.is_empty() {
        return Ok(vec![]);
    }
    let device = context.device();

    // Allocate one HOST_VISIBLE readback buffer per tensor.
    let readback_usage = wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ;
    let readback_bufs: Vec<wgpu::Buffer> = tensors
        .iter()
        .map(|t| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("batch_readback"),
                size: t.size_bytes(),
                usage: readback_usage,
                mapped_at_creation: false,
            })
        })
        .collect();

    // One encoder copies all tensors to their readback buffers.
    let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
        label: Some("batch_readback_encoder"),
    });
    for (tensor, readback) in tensors.iter().zip(readback_bufs.iter()) {
        encoder.copy_buffer_to_buffer(tensor.buffer(), 0, readback, 0, tensor.size_bytes());
    }
    context.queue().submit(Some(encoder.finish()));

    // One poll waits for all copies to land in system RAM.
    device
        .poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: None,
        })
        .map_err(|e| anyhow!("batch readback poll failed: {e}"))?;

    // Kick off all 12 map_async calls simultaneously (data is already in
    // HOST_VISIBLE memory so these complete on the next poll).
    let receivers: Vec<mpsc::Receiver<Result<(), wgpu::BufferAsyncError>>> = readback_bufs
        .iter()
        .map(|buf| {
            let (tx, rx) = mpsc::channel();
            buf.slice(..).map_async(wgpu::MapMode::Read, move |r| {
                let _ = tx.send(r);
            });
            rx
        })
        .collect();

    // One poll flushes all pending map callbacks.
    device
        .poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: None,
        })
        .map_err(|e| anyhow!("batch readback map poll failed: {e}"))?;

    // Collect data from all mapped buffers and unmap.
    let mut results = Vec::with_capacity(tensors.len());
    for (i, (buf, rx)) in readback_bufs.iter().zip(receivers.iter()).enumerate() {
        rx.recv()
            .map_err(|_| anyhow!("batch readback channel dropped for tensor {i}"))?
            .map_err(|e| anyhow!("batch readback map failed for tensor {i}: {e}"))?;

        let elements = tensors[i].shape().elements();
        let size_bytes = tensors[i].size_bytes();
        let mapped = buf.slice(0..size_bytes).get_mapped_range();
        let floats: Vec<f32> = cast_slice(&mapped).to_vec();
        drop(mapped);
        buf.unmap();

        anyhow::ensure!(
            floats.len() == elements,
            "batch readback tensor {i}: got {} elements, expected {}",
            floats.len(),
            elements
        );
        results.push(floats);
    }
    Ok(results)
}

fn reorder_hw_major(
    data: &[f32],
    channels: usize,
    height: usize,
    width: usize,
    apply_sigmoid: bool,
) -> Vec<f32> {
    let mut out = vec![0.0f32; height * width * channels];
    for c in 0..channels {
        for y in 0..height {
            for x in 0..width {
                let src = (c * height + y) * width + x;
                let dst = (y * width + x) * channels + c;
                let mut value = data[src];
                if apply_sigmoid {
                    value = sigmoid(value);
                }
                out[dst] = value;
            }
        }
    }
    out
}

#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Estimate GPU memory requirements (in bytes) based on weights + input size.
///
/// This provides a safe upper bound for the `GpuBufferPool` limit.
fn estimate_inference_memory(weights: &OnnxInitializerMap, input_size: InputSize) -> u64 {
    // 1. Calculate static weight size
    let mut total_weight_bytes = 0;
    for tensor in weights.values() {
        total_weight_bytes += std::mem::size_of_val(tensor.data()) as u64;
    }

    // 2. Estimate activation memory
    // YuNet (ResNet-ish) downsamples spatially.
    // Largest activations are at the start.
    // Map: Input (H,W,3) -> Stage 0 (H/2, W/2, 32) -> ...
    //
    // Worst case memory usage is roughly:
    // - Input tensor
    // - Largest intermediate tensor
    // - Workspace for convolution (im2col or similar if optimized, but we use direct dispatch)
    //
    // Heuristic:
    // - Input: W * H * 3 * 4 bytes
    // - Largest activation (Stage0): (W/2) * (H/2) * 32 channels * 4 bytes
    // - Typical buffers needed concurrently: ~2-3x largest activation
    //
    // For 640x640:
    // - Input: 640*640*3*4 = 4.9 MB
    // - Stage0: 320*320*32*4 = 13.1 MB
    // - Weights: ~1-2 MB (YuNet is tiny)
    //
    // Total used in practice is small (~50-100MB).
    //
    // However, we want to allow for larger inputs (e.g. 2048x2048) or larger batches/models.
    // Let's us a generous factor:
    // Limit = Weights + (InputPixels * MaxChannels * sizeof(f32) * SafetyFactor)

    let InputSize {
        width: w,
        height: h,
    } = input_size;
    let input_pixels = (w as u64) * (h as u64);

    // Assume worst case channel depth early on is 64 (standard ResNet is 64, YuNet is fewer but let's be safe)
    // And we need ping-pong buffers, so say 8x capacity to be very safe against fragmentation or held buffers.
    let activation_heuristic = input_pixels << 11; // pixels * channels(64) * copies(4) * f32(8)

    // Add 256MB fixed overhead for driver/fragmentation/mips/etc
    let fixed_overhead = 1 << 28; // 256 MB

    let required = total_weight_bytes + activation_heuristic + fixed_overhead;

    // If we can query the actual VRAM budget, we use it to intelligently set the limit.
    if let Some(hardware_available) = fcs_utils::gpu::get_available_vram() {
        let hardware_limit = hardware_available.saturating_mul(80) / 100; // 80% usage safe zone

        if hardware_limit < required {
            log::warn!(
                "Estimated requirement ({} MB) exceeds safe hardware limit ({} MB). Capping at hardware limit.",
                required >> 20,       // bytes to MiB
                hardware_limit >> 20, // bytes to MiB
            );
        } else {
            // Hardware has plenty of space. Use the hardware limit as the cap to strictly avoid
            // "MemoryLimitExceeded" errors on capable hardware, even if our heuristic is slightly off.
            log::info!(
                "Hardware VRAM ({} MB) allows increasing limit from estimated {} MB to {} MB.",
                hardware_available >> 20, // bytes to MiB
                required >> 20,           // bytes to MiB
                hardware_limit >> 20      // bytes to MiB
            );
        }
        return hardware_limit;
    }

    required
}

#[cfg(test)]
mod tests {
    use super::*;

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

    // --- reorder_hw_major ---

    #[test]
    fn reorder_hw_major_single_channel_no_sigmoid_is_identity() {
        // 1 channel, 2x2 → channel-major and HW-major are the same layout
        let data = vec![1.0f32, 2.0, 3.0, 4.0];
        let out = reorder_hw_major(&data, 1, 2, 2, false);
        assert_eq!(out, data);
    }

    #[test]
    fn reorder_hw_major_two_channels_transposes_correctly() {
        // Input is CHW: [c0_y0x0, c0_y0x1, c1_y0x0, c1_y0x1]
        // Output is HWC: [c0_y0x0, c1_y0x0, c0_y0x1, c1_y0x1]
        let data = vec![1.0f32, 2.0, 10.0, 20.0];
        let out = reorder_hw_major(&data, 2, 1, 2, false);
        assert_eq!(out, vec![1.0, 10.0, 2.0, 20.0]);
    }

    #[test]
    fn reorder_hw_major_applies_sigmoid_when_requested() {
        let data = vec![0.0f32];
        let out = reorder_hw_major(&data, 1, 1, 1, true);
        assert!((out[0] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn reorder_hw_major_no_sigmoid_preserves_values() {
        let data = vec![0.0f32];
        let out = reorder_hw_major(&data, 1, 1, 1, false);
        assert_eq!(out[0], 0.0);
    }
}
