use crate::{
    gpu::{
        graph::{self, DetectionLevelOutputs, HEAD_BRANCH_CHANNELS},
        ops::GpuInferenceOps,
        tensor::GpuTensor,
    },
    model::{HeadLayout, decode_yunet_outputs_with},
    preprocess::InputSize,
};
use bytemuck::cast_slice;
use std::sync::mpsc;
use wgpu::CommandEncoderDescriptor;

use crate::tensor::Tensor;
use anyhow::{Context, Result, anyhow};
use fcs_utils::gpu::{GpuAvailability, GpuContext, GpuContextOptions};
use fcs_utils::timing_guard;
use std::{
    collections::{HashMap, HashSet},
    path::Path,
    sync::{Arc, Condvar, Mutex},
};

use crate::yunet::{BACKBONE_STAGES, DETECTION_HEADS, onnx::OnnxInitializerMap};

#[derive(Debug, Default)]
struct GpuYuNetWorkspace {
    input_tensors: Vec<GpuTensor>, // Pool of available tensors
}

/// Inferences allowed to hold intermediates on one model at the same time (experiment 21).
///
/// Every in-flight inference parks its intermediates in its own execution scope, so the pool
/// grows about 36 MB per concurrent caller: 42 MB at one, 1.2 GB at 32 rayon workers, on the
/// 4090 and on the Radeon iGPU alike. On the iGPU the extra callers also cost throughput --
/// 80.7 detections/s at one in flight, 57 at sixteen -- while on the 4090 four in flight already
/// reach 945/s, far past what a folder job's CPU work can feed. `FCS_MAX_IN_FLIGHT` overrides it.
// ponytail: one constant for every adapter; size it from the device if a GPU ever wants more.
const MAX_IN_FLIGHT: usize = 4;

fn max_in_flight() -> usize {
    crate::model_config::positive_count(std::env::var("FCS_MAX_IN_FLIGHT").ok().as_deref())
        .unwrap_or(MAX_IN_FLIGHT)
}

/// A counting gate: `enter` blocks while the limit is reached, the guard leaves on drop.
#[derive(Debug, Default)]
struct InFlight {
    count: Mutex<usize>,
    freed: Condvar,
}

struct InFlightSlot<'a>(&'a InFlight);

impl InFlight {
    fn enter(&self) -> Result<InFlightSlot<'_>> {
        let limit = max_in_flight();
        let mut count = self
            .count
            .lock()
            .map_err(|_| anyhow!("in-flight gate poisoned"))?;
        while *count >= limit {
            count = self
                .freed
                .wait(count)
                .map_err(|_| anyhow!("in-flight gate poisoned"))?;
        }
        *count += 1;
        Ok(InFlightSlot(self))
    }
}

impl Drop for InFlightSlot<'_> {
    fn drop(&mut self) {
        // A poisoned count still has to be released, or every other caller waits forever.
        let mut count = self.0.count.lock().unwrap_or_else(|e| e.into_inner());
        *count -= 1;
        self.0.freed.notify_one();
    }
}

/// Reusable YuNet GPU runtime with resident weights and pooled working buffers.
///
/// Use [`Self::with_context`] to share a device with GPU preprocessing.
#[derive(Debug)]
pub struct GpuYuNet {
    ops: Arc<GpuInferenceOps>,
    weights: graph::GpuWeights,
    input_size: InputSize,
    workspace: Mutex<GpuYuNetWorkspace>,
    in_flight: InFlight,
}

impl GpuYuNet {
    /// Initialize a GPU context and load YuNet weights for the configured input size.
    /// Returns an error if GPU initialization, model loading, or pipeline setup fails.
    pub fn new<P: AsRef<Path>>(model_path: P, input_size: InputSize) -> Result<Self> {
        let context = match GpuContext::init_with_fallback(&GpuContextOptions::default()) {
            GpuAvailability::Available(ctx) => ctx,
            GpuAvailability::Disabled { reason } => {
                anyhow::bail!("GPU backend disabled by configuration: {reason}")
            }
            GpuAvailability::Unavailable { error } => {
                anyhow::bail!("GPU backend unavailable: {error}")
            }
        };
        Self::with_context(context, model_path, input_size)
    }

    /// Build the model on an existing device.
    ///
    /// Sharing a context with the preprocessor is what makes an end-to-end GPU path possible:
    /// tensors cannot cross `wgpu::Device` boundaries, so with separate devices the preprocessed
    /// tensor has to be downloaded and uploaded again. It also avoids initialising a second
    /// adapter, which `new` does implicitly.
    pub fn with_context<P: AsRef<Path>>(
        context: Arc<GpuContext>,
        model_path: P,
        input_size: InputSize,
    ) -> Result<Self> {
        let model_path = model_path.as_ref();
        let loader = {
            let _guard = timing_guard("fcs_core::load_onnx_weights", log::Level::Trace);
            crate::yunet::load_backbone_weights(
                model_path,
                BACKBONE_STAGES.len(),
                true,
                DETECTION_HEADS.len(),
            )?
        };

        let memory_limit = estimate_inference_memory(&loader, input_size);
        let ops = {
            let _guard = timing_guard("fcs_core::compile_pipelines", log::Level::Trace);
            Arc::new(GpuInferenceOps::new(context, Some(memory_limit))?)
        };
        let weight_map = {
            let _guard = timing_guard("fcs_core::upload_weights", log::Level::Trace);
            upload_gpu_weights(&ops, loader)?
        };
        Ok(Self {
            ops,
            weights: weight_map,
            input_size,
            workspace: Mutex::new(GpuYuNetWorkspace::default()),
            in_flight: InFlight::default(),
        })
    }

    /// Upload a preprocessed BGR NCHW tensor, run inference, and download decoded rows.
    ///
    /// The input must match `[1, 3, input_height, input_width]`. The result is
    /// `[N, 15]` in model-input coordinates, as for [`crate::YuNetModel::run`],
    /// before score filtering and non-maximum suppression. Returns an error on
    /// incompatible input, GPU execution, or readback failure.
    pub fn run(&self, tensor: Tensor) -> Result<Tensor> {
        let dims = tensor.shape().to_vec();
        let data = tensor.as_slice();

        // 1. Acquire a tensor from the pool or create a new one
        let input_gpu = {
            let _guard = timing_guard("fcs_core::gpu_upload", log::Level::Trace);
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
        let result = self.run_inference(&input_gpu, None);

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

    fn run_inference(&self, input_gpu: &GpuTensor, min_score: Option<f32>) -> Result<Tensor> {
        // Hold an execution scope for the whole encode/submit/readback cycle. Intermediates are
        // dropped while the encoder is still being built — before anything is submitted — so
        // without this they would return to the shared pool and a concurrently encoding thread
        // could acquire a buffer this pass already references. Declared first so it outlives
        // every tensor below and is dropped last, once the readback has completed.
        // Taken before the scope, so a caller waiting here holds no intermediates.
        let _slot = self.in_flight.enter()?;
        let _scope = self.ops.buffer_pool().execution_scope();

        // Accumulate the entire forward pass into one command buffer and submit
        // once. This eliminates ~53 individual queue.submit() calls (one per op)
        // and gives the GPU a full workload to pipeline, dramatically improving
        // utilisation vs the previous per-op-submit pattern.
        // Split into three guards because the whole call is dominated by what happens
        // around the dispatches rather than by the dispatches: GPU timestamps put the
        // forward pass at ~0.9 ms against several ms of wall time, so knowing which of
        // encode, readback and decode owns the rest is what makes the gap actionable.
        let levels = {
            let _guard = timing_guard("fcs_core::gpu_encode", log::Level::Trace);
            let mut encoder =
                self.ops
                    .context()
                    .device()
                    .create_command_encoder(&CommandEncoderDescriptor {
                        label: Some("inference"),
                    });
            let levels = {
                let _record = timing_guard("fcs_core::gpu_record", log::Level::Trace);
                if self.context().profiler().is_some() {
                    self.encode_inference(&mut encoder, input_gpu)?
                } else {
                    // Compute dispatches have separate usage scopes even inside one pass:
                    // wgpu inserts the dependencies needed for pooled-buffer reuse.
                    // Keep separate passes only when per-op timestamps are requested.
                    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("yunet_forward"),
                        timestamp_writes: None,
                    });
                    self.encode_inference(&mut pass, input_gpu)?
                }
            };
            {
                // Split because the two are not the same kind of cost: `finish` is wgpu
                // turning the recorded pass into backend commands, which scales with what
                // was recorded, while `submit` is handing the result to the queue.
                let _submit = timing_guard("fcs_core::gpu_submit", log::Level::Trace);
                let commands = {
                    let _finish = timing_guard("fcs_core::gpu_finish", log::Level::Trace);
                    encoder.finish()
                };
                self.ops.context().queue().submit(Some(commands));
            }
            levels
        };

        // Blocks until the GPU has finished, so it absorbs the forward pass itself as
        // well as the download of the 12 head outputs.
        let outputs = {
            let _guard = timing_guard("fcs_core::gpu_readback", log::Level::Trace);
            build_decode_tensors(&levels)?
        };

        let _guard = timing_guard("fcs_core::gpu_decode", log::Level::Trace);
        decode_yunet_outputs_with(
            &outputs,
            self.input_size,
            HeadLayout::ChannelMajorLogits,
            min_score,
        )
    }

    fn encode_inference(
        &self,
        encoder: &mut impl super::utils::ComputeDispatch,
        input: &GpuTensor,
    ) -> Result<[DetectionLevelOutputs; 3]> {
        let features = graph::encode_backbone_features(
            encoder,
            &self.ops,
            &self.weights,
            input,
            BACKBONE_STAGES.len(),
        )?;
        graph::encode_neck_and_heads(encoder, &self.ops, &self.weights, features)
    }

    /// Return the number of bytes tracked by the inference buffer pool.
    pub fn memory_usage(&self) -> u64 {
        self.ops.memory_usage()
    }

    /// Convolution bind-group cache hits and misses, for the test that guards the hit rate.
    #[cfg(test)]
    pub(crate) fn bind_cache_stats(&self) -> (u64, u64) {
        self.ops.bind_cache_stats()
    }

    /// The device this model runs on.
    pub fn context(&self) -> &Arc<GpuContext> {
        self.ops.context()
    }

    /// Allocate an input tensor on this model's device, for a caller that wants to write into it
    /// directly (GPU preprocessing) rather than upload host data.
    pub fn allocate_input(&self, input_size: InputSize) -> Result<GpuTensor> {
        GpuTensor::uninitialized_with_pool(
            self.ops.context().clone(),
            Some(self.ops.buffer_pool().clone()),
            vec![1, 3, input_size.height as usize, input_size.width as usize],
            Some("gpu_input"),
        )
    }

    /// Run inference on a tensor that is already on this device.
    ///
    /// The counterpart to [`GpuYuNet::run`], which takes host data and uploads it. Submissions on
    /// one queue execute in order, so a preprocess dispatch submitted before this call is
    /// guaranteed to have written `input` by the time the graph reads it -- no host
    /// synchronisation, no round trip.
    pub fn run_on_device(&self, input: &GpuTensor) -> Result<Tensor> {
        self.run_on_device_filtered(input, None)
    }

    /// Run on a device-resident tensor, skipping the decode of cells that cannot reach
    /// `min_score`.
    ///
    /// Every skipped row is zeroed, so postprocessing drops it on either the score or the
    /// zero width -- identical detections, four fewer exponentials and a square root for the
    /// cells that were never going to survive. `None` decodes everything, which is what the
    /// parity probes and tests use so their fingerprints stay a full check of the decode.
    pub fn run_on_device_filtered(
        &self,
        input: &GpuTensor,
        min_score: Option<f32>,
    ) -> Result<Tensor> {
        anyhow::ensure!(
            Arc::ptr_eq(input.context(), self.ops.context()),
            "input tensor belongs to a different GPU context than the model"
        );
        self.run_inference(input, min_score)
    }
}

fn upload_gpu_weights(
    ops: &GpuInferenceOps,
    loader: OnnxInitializerMap,
) -> Result<HashMap<String, GpuTensor>> {
    let mut map = HashMap::with_capacity(loader.len() + DETECTION_HEADS.len() * 4);
    let mut superseded: HashSet<&'static str> = HashSet::new();
    for fused in fuse_head_weights(&loader)? {
        superseded.extend(fused.sources);
        let gpu_tensor = ops.upload_tensor(fused.dims, &fused.data, Some(&fused.name))?;
        map.insert(fused.name, gpu_tensor);
    }
    for (name, tensor) in loader.into_map() {
        // The four per-branch head initializers are already in the fused tensor; uploading
        // them again would leave twelve buffers per level that nothing binds.
        if superseded.contains(name.as_str()) {
            continue;
        }
        let gpu_tensor = ops.upload_tensor(
            tensor.dims().to_vec(),
            tensor.data(),
            Some(&format!("weight::{name}")),
        )?;
        map.insert(name, gpu_tensor);
    }
    Ok(map)
}

struct FusedWeight {
    name: String,
    dims: Vec<usize>,
    data: Vec<f32>,
    /// The initializers this tensor replaces, so they are not uploaded separately.
    sources: [&'static str; 4],
}

/// Concatenate each level's four head branches along the output-channel axis.
///
/// The cls, obj, bbox and kps branches read the same feature map and are the same
/// 1x1-then-depthwise shape, so one convolution over the concatenated weights computes all
/// four. Both halves concatenate: a pointwise output channel depends only on its own row of
/// weights, and a depthwise channel only on its own 3x3 kernel, so nothing crosses a branch
/// boundary and the arithmetic per channel is unchanged.
///
/// Levels whose initializers were not requested are skipped rather than failing, because
/// `load_backbone_weights` takes a head-level count and the tests load partial graphs.
fn fuse_head_weights(loader: &OnnxInitializerMap) -> Result<Vec<FusedWeight>> {
    let mut fused = Vec::with_capacity(DETECTION_HEADS.len() * 4);
    for (level, head) in DETECTION_HEADS.iter().enumerate() {
        let branches = [&head.cls, &head.obj, &head.bbox, &head.kps];
        let parts = [
            (
                "point_weight",
                [
                    branches[0].conv1_weight,
                    branches[1].conv1_weight,
                    branches[2].conv1_weight,
                    branches[3].conv1_weight,
                ],
            ),
            (
                "point_bias",
                [
                    branches[0].conv1_bias,
                    branches[1].conv1_bias,
                    branches[2].conv1_bias,
                    branches[3].conv1_bias,
                ],
            ),
            (
                "depth_weight",
                [
                    branches[0].conv2_weight,
                    branches[1].conv2_weight,
                    branches[2].conv2_weight,
                    branches[3].conv2_weight,
                ],
            ),
            (
                "depth_bias",
                [
                    branches[0].conv2_bias,
                    branches[1].conv2_bias,
                    branches[2].conv2_bias,
                    branches[3].conv2_bias,
                ],
            ),
        ];
        if parts
            .iter()
            .any(|(_, names)| names.iter().any(|n| loader.tensor(n).is_err()))
        {
            continue;
        }
        for (part, names) in parts {
            let tensors: Vec<_> = names
                .iter()
                .map(|n| loader.tensor(n))
                .collect::<Result<_>>()?;
            // Output channels add; every other dimension has to agree, or the branches were
            // not the same shape and concatenating them would silently compute nonsense.
            let mut dims = tensors[0].dims().to_vec();
            for tensor in &tensors[1..] {
                anyhow::ensure!(
                    tensor.dims()[1..] == dims[1..],
                    "head branch {part} shapes differ beyond the channel axis: {:?} vs {:?}",
                    tensor.dims(),
                    dims
                );
                dims[0] += tensor.dims()[0];
            }
            let mut data = Vec::with_capacity(tensors.iter().map(|t| t.data().len()).sum());
            for tensor in &tensors {
                data.extend_from_slice(tensor.data());
            }
            fused.push(FusedWeight {
                name: graph::fused_head_key(level, part),
                dims,
                data,
                sources: names,
            });
        }
    }
    Ok(fused)
}

fn build_decode_tensors(levels: &[DetectionLevelOutputs; 3]) -> Result<Vec<Tensor>> {
    // Collect the 12 output tensors in the order we need them:
    //   cls×3, obj×3, bbox×3, kps×3
    // along with the metadata needed to reorder each one on the CPU.
    struct BranchMeta {
        height: usize,
        width: usize,
        channels: usize,
    }

    let mut gpu_tensors: Vec<&GpuTensor> = Vec::with_capacity(levels.len());
    let mut meta: Vec<BranchMeta> = Vec::with_capacity(levels.len());

    for level in levels.iter() {
        let shape = level.feature.shape().dims();
        anyhow::ensure!(
            shape.len() == 4,
            "feature map must be NCHW (got {:?})",
            shape
        );
        gpu_tensors.push(&level.heads);
        meta.push(BranchMeta {
            height: shape[2],
            width: shape[3],
            channels: HEAD_BRANCH_CHANNELS.iter().sum(),
        });
    }

    // One submit + one poll downloads one buffer per level, and each level's four
    // branches are taken straight off the mapped view. Experiment 55: transposing all
    // twelve heads to HWC first, only for the decoder to read them cell by cell, was pure
    // rearrangement of data nothing else looked at, so each level arrives channel-major
    // with cls, obj, bbox and kps in that order and splitting it is taking the channel
    // ranges off the front in turn.
    //
    // Experiment 17: the branches used to be cut out of a `Vec` that `readback_collect`
    // had already copied out of the mapped range, so 525 KB was copied once to own it and
    // most of it again to divide it. The ranges are contiguous and disjoint, so copying
    // each branch directly out of the mapped view does the whole job in one pass, worth
    // 0.011 ms at 0.8 MP and 0.021 at 10. Not copying at all -- decoding from the mapping
    // -- loses: reads out of a mapped allocation measured 55% slower per access than the
    // same reads out of a `Vec`, which is more than the copy costs. `readback_bytes
    // --reads` is that measurement.
    let per_level = batch_download_with(gpu_tensors[0].context(), &gpu_tensors, |i, flat| {
        let m = &meta[i];
        let rows = m.height * m.width;
        anyhow::ensure!(
            flat.len() == m.channels * rows,
            "fused head buffer is {} floats, expected {}",
            flat.len(),
            m.channels * rows
        );
        let mut start = 0;
        let mut branches = Vec::with_capacity(HEAD_BRANCH_CHANNELS.len());
        for channels in HEAD_BRANCH_CHANNELS {
            let end = start + channels * rows;
            branches.push(
                Tensor::from_vec(&[channels, rows], flat[start..end].to_vec())
                    .context("failed to build tensor from branch output")?,
            );
            start = end;
        }
        Ok(branches)
    })
    .context("batch download of detection head outputs")?;

    let outputs: Vec<Tensor> = per_level.into_iter().flatten().collect();

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

/// Download multiple GPU tensors in a single submit + single poll, calling `split`
/// on each one's mapped view.
///
/// All buffer copies are recorded into one `CommandEncoder` and submitted
/// together. The GPU DMA engine can pipeline them, and we block exactly
/// once instead of once per tensor.
///
/// `split` runs while the buffer is still mapped, so a caller that only wants part of a
/// download -- or wants it cut up -- copies once instead of owning the whole buffer
/// first and dividing it afterwards (experiment 17). It must not keep the slice: the
/// buffer is unmapped as soon as it returns. Passing `|_, f| Ok(f.to_vec())` recovers
/// the plain "give me the buffer" download.
fn batch_download_with<T>(
    context: &Arc<GpuContext>,
    tensors: &[&GpuTensor],
    split: impl Fn(usize, &[f32]) -> Result<T>,
) -> Result<Vec<T>> {
    if tensors.is_empty() {
        return Ok(vec![]);
    }
    let device = context.device();

    // Allocate one HOST_VISIBLE readback buffer per tensor.
    let readback_usage = wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ;
    let readback_bufs: Vec<wgpu::Buffer> = {
        let _guard = timing_guard("fcs_core::readback_alloc", log::Level::Trace);
        tensors
            .iter()
            .map(|t| {
                device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("batch_readback"),
                    size: t.size_bytes(),
                    usage: readback_usage,
                    mapped_at_creation: false,
                })
            })
            .collect()
    };

    {
        // One encoder copies all tensors to their readback buffers.
        let _guard = timing_guard("fcs_core::readback_copy", log::Level::Trace);
        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("batch_readback_encoder"),
        });
        for (tensor, readback) in tensors.iter().zip(readback_bufs.iter()) {
            encoder.copy_buffer_to_buffer(tensor.buffer(), 0, readback, 0, tensor.size_bytes());
        }
        context.queue().submit(Some(encoder.finish()));
    }

    // Request every map before waiting for anything. `map_async` on a buffer with a
    // pending submission is already deferred until that submission completes, so one
    // wait drives the copies and the map callbacks together; polling for the copies
    // first only added a second blocking call that had nothing left to wait for.
    let receivers: Vec<mpsc::Receiver<Result<(), wgpu::BufferAsyncError>>> = {
        let _guard = timing_guard("fcs_core::readback_map", log::Level::Trace);
        readback_bufs
            .iter()
            .map(|buf| {
                let (tx, rx) = mpsc::channel();
                buf.slice(..).map_async(wgpu::MapMode::Read, move |r| {
                    let _ = tx.send(r);
                });
                rx
            })
            .collect()
    };

    {
        // The one blocking wait: the copies land and every map callback fires. It
        // absorbs the forward pass itself, so this is GPU execution plus the copies,
        // not idle cost, and must not be added to a GPU timestamp total.
        //
        // Device-wide rather than on this submission's index, which is not an oversight:
        // experiment 16 measured both. Waiting on the index does shorten this phase 24%
        // under 32 workers, because a device-wide wait also waits for whatever they
        // submitted afterwards -- and it changes nothing end to end at any thread count
        // from 2 to 32, so the simpler call stays.
        let _guard = timing_guard("fcs_core::readback_wait", log::Level::Trace);
        fcs_utils::gpu::wait_for_gpu(device, "batch readback")?;
    }

    // Collect data from all mapped buffers and unmap.
    let _collect = timing_guard("fcs_core::readback_collect", log::Level::Trace);
    let mut results = Vec::with_capacity(tensors.len());
    for (i, (buf, rx)) in readback_bufs.iter().zip(receivers.iter()).enumerate() {
        // The wait above drives the map callbacks, so this should already hold a result.
        // Bounded anyway, and for the same reason 94 bounded the wait: a `recv` that can
        // only ever block forever is the wrong shape for a callback that might not fire.
        rx.recv_timeout(fcs_utils::gpu::GPU_WAIT_TIMEOUT)
            .map_err(|_| anyhow!("batch readback callback for tensor {i} never arrived"))?
            .map_err(|e| anyhow!("batch readback map failed for tensor {i}: {e}"))?;

        let elements = tensors[i].shape().elements();
        let size_bytes = tensors[i].size_bytes();
        let mapped = buf
            .slice(0..size_bytes)
            .get_mapped_range()
            .map_err(|e| anyhow!("batch readback mapped range failed for tensor {i}: {e}"))?;
        let floats: &[f32] = cast_slice(&mapped);
        // Checked before `split` sees it, so a caller indexing by its own shape cannot
        // read past a buffer that came back short.
        let checked = if floats.len() == elements {
            split(i, floats)
        } else {
            Err(anyhow!(
                "batch readback tensor {i}: got {} elements, expected {elements}",
                floats.len()
            ))
        };
        // Unmap before propagating: a buffer left mapped by an early return cannot be
        // reused or dropped cleanly.
        drop(mapped);
        buf.unmap();
        results.push(checked?);
    }

    Ok(results)
}

/// The heuristic requirement in bytes, before any hardware cap is applied.
///
/// Split out of [`estimate_inference_memory`] to make it observable. That
/// function returns the hardware-derived limit whenever `get_available_vram`
/// succeeds — which is on any machine with a working adapter — so this
/// arithmetic never reaches its return value there and no test could pin it
/// down. Taking the weight total as a plain `u64` also means the formula can be
/// checked without constructing an ONNX initializer map.
fn estimate_required_bytes(total_weight_bytes: u64, input_size: InputSize) -> u64 {
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

    total_weight_bytes + activation_heuristic + fixed_overhead
}

/// Estimate GPU memory requirements (in bytes) based on weights + input size.
///
/// This provides a safe upper bound for the `GpuBufferPool` limit.
fn estimate_inference_memory(weights: &OnnxInitializerMap, input_size: InputSize) -> u64 {
    // 1. Calculate static weight size
    let total_weight_bytes: u64 = weights
        .values()
        .map(|tensor| std::mem::size_of_val(tensor.data()) as u64)
        .sum();

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

    let required = estimate_required_bytes(total_weight_bytes, input_size);

    // If we can query the actual VRAM budget, we use it to intelligently set the limit.
    if let Some(hardware_available) = fcs_utils::gpu::get_available_vram() {
        let hardware_limit = hardware_pool_limit(hardware_available);

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

/// The share of reported VRAM the buffer pool may use: 80%, leaving room for the driver and
/// other applications.
fn hardware_pool_limit(available: u64) -> u64 {
    available.saturating_mul(80) / 100
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_pool_may_use_four_fifths_of_the_reported_vram() {
        assert_eq!(hardware_pool_limit(1_000), 800);
        // Saturates rather than overflowing.
        assert_eq!(hardware_pool_limit(u64::MAX), u64::MAX / 100);
    }

    #[test]
    fn the_in_flight_limit_defaults_when_not_overridden() {
        if std::env::var_os("FCS_MAX_IN_FLIGHT").is_none() {
            assert_eq!(max_in_flight(), MAX_IN_FLIGHT);
        }
    }

    // --- memory estimate ---

    /// The pool limit derives from this, and getting it wrong either starves
    /// inference or lets it over-allocate, so the formula is asserted exactly
    /// rather than by inequality.
    #[test]
    fn required_bytes_is_weights_plus_activations_plus_fixed_overhead() {
        let fixed = 1u64 << 28; // 256 MiB
        let size = InputSize {
            width: 640,
            height: 640,
        };
        // 640 * 640 pixels, shifted left 11 (channels * copies * f32).
        let activations = 640u64 * 640 * 2048;

        assert_eq!(
            estimate_required_bytes(0, size),
            activations + fixed,
            "with no weights the estimate is activations plus the fixed overhead"
        );

        // Weights are added, not scaled or ignored.
        assert_eq!(
            estimate_required_bytes(1_000_000, size),
            1_000_000 + activations + fixed
        );
        assert_eq!(
            estimate_required_bytes(1_000_000, size) - estimate_required_bytes(0, size),
            1_000_000,
            "weight bytes must pass through one-for-one"
        );
    }

    #[test]
    fn required_bytes_scales_with_pixel_count_not_with_a_single_dimension() {
        let square = InputSize {
            width: 640,
            height: 640,
        };
        let double_width = InputSize {
            width: 1280,
            height: 640,
        };
        let fixed = 1u64 << 28;

        let a = estimate_required_bytes(0, square) - fixed;
        let b = estimate_required_bytes(0, double_width) - fixed;
        assert_eq!(
            b,
            a * 2,
            "doubling one dimension doubles the pixel count and so the activation estimate"
        );

        // A degenerate size still yields the overhead rather than zero, so the
        // pool is never given a limit it cannot work with.
        let zero = InputSize {
            width: 0,
            height: 0,
        };
        assert_eq!(estimate_required_bytes(0, zero), fixed);

        // Non-square dimensions are symmetric: only the product matters.
        let transposed = InputSize {
            width: 640,
            height: 1280,
        };
        assert_eq!(
            estimate_required_bytes(0, double_width),
            estimate_required_bytes(0, transposed)
        );
    }

    #[test]
    fn required_bytes_is_monotonic_in_both_inputs() {
        let small = InputSize {
            width: 320,
            height: 320,
        };
        let large = InputSize {
            width: 2048,
            height: 2048,
        };
        assert!(estimate_required_bytes(0, large) > estimate_required_bytes(0, small));
        assert!(estimate_required_bytes(10_000, small) > estimate_required_bytes(0, small));
    }
}
