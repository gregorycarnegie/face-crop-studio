//! GPU-side timing for compute passes, via wgpu timestamp queries.
//!
//! Wall-clock benchmarks around a dispatch measure buffer upload, readback and driver
//! overhead as much as the shader, and on a desktop GPU they swing far more than the
//! shader changes worth making — enough that identical code can measure 40% apart run
//! to run. These timestamps come from the GPU's own clock at pass boundaries, so a
//! shader change shows up as a change in the pass that contains it.
//!
//! Off unless [`super::GpuContextOptions::profiling`] is set, and absent entirely when
//! the adapter lacks `TIMESTAMP_QUERY`, so the normal path allocates nothing.

use std::sync::{
    Mutex,
    atomic::{AtomicU32, Ordering},
};

use anyhow::{Result, anyhow};
use wgpu::{
    Buffer, BufferAddress, BufferDescriptor, BufferUsages, CommandEncoderDescriptor,
    ComputePassTimestampWrites, Device, Features, QuerySet, QuerySetDescriptor, QueryType, Queue,
};

/// How long one compute pass took on the GPU.
#[derive(Clone, Debug, PartialEq)]
pub struct PassTiming {
    pub label: String,
    pub duration_ns: f64,
}

/// Bytes per resolved timestamp.
const TIMESTAMP_BYTES: BufferAddress = 8;
/// Resolve destinations must be aligned to this; see `wgpu::QUERY_RESOLVE_BUFFER_ALIGNMENT`.
const RESOLVE_ALIGNMENT: BufferAddress = 256;

/// Records GPU timestamps at compute-pass boundaries.
#[derive(Debug)]
pub struct GpuProfiler {
    query_set: QuerySet,
    /// Total query slots, i.e. two per pass.
    capacity: u32,
    /// Next free query slot. Passes claim two at a time.
    next: AtomicU32,
    /// Label per pass slot, indexed by `query index / 2` so it stays correct when
    /// several threads encode passes against one context.
    labels: Mutex<Vec<Option<String>>>,
    resolve: Buffer,
    readback: Buffer,
    /// Nanoseconds per timestamp tick.
    period_ns: f32,
}

impl GpuProfiler {
    /// Returns `None` when the adapter cannot timestamp, so callers can stay unconditional.
    pub fn new(
        device: &Device,
        queue: &Queue,
        features: Features,
        max_passes: u32,
    ) -> Option<Self> {
        if !features.contains(Features::TIMESTAMP_QUERY) {
            return None;
        }
        let max_passes = max_passes.max(1);
        let capacity = max_passes.checked_mul(2)?;
        let byte_len = (capacity as BufferAddress) * TIMESTAMP_BYTES;
        let byte_len = byte_len.next_multiple_of(RESOLVE_ALIGNMENT);

        Some(Self {
            query_set: device.create_query_set(&QuerySetDescriptor {
                label: Some("gpu_profiler_timestamps"),
                ty: QueryType::Timestamp,
                count: capacity,
            }),
            capacity,
            next: AtomicU32::new(0),
            labels: Mutex::new(vec![None; max_passes as usize]),
            resolve: device.create_buffer(&BufferDescriptor {
                label: Some("gpu_profiler_resolve"),
                size: byte_len,
                usage: BufferUsages::QUERY_RESOLVE | BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            }),
            readback: device.create_buffer(&BufferDescriptor {
                label: Some("gpu_profiler_readback"),
                size: byte_len,
                usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            period_ns: queue.get_timestamp_period(),
        })
    }

    /// Claims two query slots for one compute pass.
    ///
    /// Returns `None` once the profiler is full rather than failing the dispatch: a
    /// missing timing is a better outcome than a broken render.
    pub fn timestamp_writes(&self, label: &str) -> Option<ComputePassTimestampWrites<'_>> {
        let base = self.next.fetch_add(2, Ordering::Relaxed);
        if base >= self.capacity {
            // Undo so a long-running session cannot wrap the counter back into range.
            self.next.store(self.capacity, Ordering::Relaxed);
            return None;
        }
        self.labels.lock().ok()?[(base / 2) as usize] = Some(label.to_string());
        Some(ComputePassTimestampWrites {
            query_set: &self.query_set,
            beginning_of_pass_write_index: Some(base),
            end_of_pass_write_index: Some(base + 1),
        })
    }

    /// Resolves and returns every pass recorded since the last call, then clears.
    ///
    /// Resolving here rather than in each caller's encoder is why instrumenting a pass
    /// is a one-line change: timestamps live in the query set until read, so this can
    /// use an encoder of its own. Every pass being measured must already have been
    /// submitted, or its slot resolves to a stale value.
    pub fn take(&self, device: &Device, queue: &Queue) -> Result<Vec<PassTiming>> {
        let used = self.next.load(Ordering::Relaxed).min(self.capacity);
        if used == 0 {
            return Ok(Vec::new());
        }
        let byte_len = (used as BufferAddress) * TIMESTAMP_BYTES;

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("gpu_profiler_resolve_encoder"),
        });
        encoder.resolve_query_set(&self.query_set, 0..used, &self.resolve, 0);
        encoder.copy_buffer_to_buffer(&self.resolve, 0, &self.readback, 0, byte_len);
        queue.submit(Some(encoder.finish()));

        let ticks = self.read_back(device, byte_len)?;
        let labels = self
            .labels
            .lock()
            .map_err(|_| anyhow!("GPU profiler labels poisoned"))?;

        // Each pass owns an adjacent (begin, end) pair.
        let timings = ticks
            .as_chunks::<2>()
            .0
            .iter()
            .enumerate()
            .filter_map(|(pass, pair)| {
                let label = labels.get(pass)?.clone()?;
                // Saturating: a pass whose timestamps land out of order (some drivers
                // reorder across a submit boundary) reports zero rather than underflows.
                let elapsed = pair[1].saturating_sub(pair[0]);
                Some(PassTiming {
                    label,
                    duration_ns: elapsed as f64 * f64::from(self.period_ns),
                })
            })
            .collect();

        drop(labels);
        self.reset()?;
        Ok(timings)
    }

    /// Drops recorded passes without reading them.
    pub fn reset(&self) -> Result<()> {
        self.labels
            .lock()
            .map_err(|_| anyhow!("GPU profiler labels poisoned"))?
            .iter_mut()
            .for_each(|slot| *slot = None);
        self.next.store(0, Ordering::Relaxed);
        Ok(())
    }

    fn read_back(&self, device: &Device, byte_len: BufferAddress) -> Result<Vec<u64>> {
        use std::sync::mpsc;

        let slice = self.readback.slice(0..byte_len);
        let (sender, receiver) = mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |res| {
            let _ = sender.send(res);
        });
        device
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: None,
            })
            .map_err(|err| anyhow!("device poll failed during profiler readback: {err}"))?;
        receiver
            .recv()
            .map_err(|_| anyhow!("GPU profiler map callback dropped"))?
            .map_err(|err| anyhow!("GPU profiler map error: {err}"))?;

        let mapped = slice
            .get_mapped_range()
            .map_err(|err| anyhow!("GPU profiler mapped range failed: {err}"))?;
        let ticks: Vec<u64> = bytemuck::cast_slice(&mapped).to_vec();
        drop(mapped);
        self.readback.unmap();
        Ok(ticks)
    }
}

/// Sums timings that share a label, so a graph with 50 conv passes reads as one row.
pub fn total_by_label(timings: &[PassTiming]) -> Vec<(String, usize, f64)> {
    let mut order: Vec<String> = Vec::new();
    let mut totals: std::collections::HashMap<String, (usize, f64)> =
        std::collections::HashMap::new();
    for timing in timings {
        let entry = totals.entry(timing.label.clone()).or_insert_with(|| {
            order.push(timing.label.clone());
            (0, 0.0)
        });
        entry.0 += 1;
        entry.1 += timing.duration_ns;
    }
    let mut rows: Vec<(String, usize, f64)> = order
        .into_iter()
        .map(|label| {
            let (count, ns) = totals[&label];
            (label, count, ns)
        })
        .collect();
    rows.sort_by(|a, b| b.2.total_cmp(&a.2));
    rows
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn total_by_label_groups_and_sorts_by_time() {
        let timings = vec![
            PassTiming {
                label: "conv2d".into(),
                duration_ns: 100.0,
            },
            PassTiming {
                label: "pool".into(),
                duration_ns: 500.0,
            },
            PassTiming {
                label: "conv2d".into(),
                duration_ns: 900.0,
            },
        ];
        let rows = total_by_label(&timings);
        assert_eq!(rows[0], ("conv2d".to_string(), 2, 1000.0));
        assert_eq!(rows[1], ("pool".to_string(), 1, 500.0));
    }

    #[test]
    fn total_by_label_on_empty_input_is_empty() {
        assert!(total_by_label(&[]).is_empty());
    }

    #[test]
    fn profiling_off_by_default_records_nothing() {
        let Some(ctx) = crate::gpu::test_support::test_context() else {
            eprintln!("Skipping profiler test: no GPU");
            return;
        };
        assert!(ctx.profiler().is_none(), "profiling must be opt-in");
        assert!(ctx.timestamp_writes("noop").is_none());
        assert!(
            ctx.take_pass_timings().expect("timings").is_empty(),
            "a context without a profiler reports no passes"
        );
    }

    #[test]
    fn a_real_dispatch_is_timed() {
        let Some(ctx) = crate::gpu::test_support::profiling_context() else {
            eprintln!("Skipping profiler test: no GPU");
            return;
        };
        let Some(_) = ctx.profiler() else {
            eprintln!("Skipping profiler test: adapter has no TIMESTAMP_QUERY");
            return;
        };

        // A real op through the normal API: this is the path the instrumentation has to
        // work on, and it proves the label plumbed through from the pass descriptor.
        let equalizer =
            crate::gpu::GpuHistogramEqualizer::new(ctx.clone()).expect("histogram equalizer");
        let image = crate::gpu::test_support::gradient_image(256, 256);
        equalizer.equalize(&image).expect("equalize");

        let timings = ctx.take_pass_timings().expect("timings");
        let labels: Vec<&str> = timings.iter().map(|t| t.label.as_str()).collect();
        assert_eq!(
            labels,
            vec!["hist", "hist_cdf", "hist_apply"],
            "every pass of the op should be recorded, in submission order"
        );
        assert!(
            timings.iter().all(|t| t.duration_ns > 0.0),
            "a dispatch that ran cannot take zero GPU time: {timings:?}"
        );
        // Loose upper bound: catches a period/unit mistake (ticks read as nanoseconds,
        // or a missing timestamp_period multiply) without being flaky on a slow adapter.
        assert!(
            timings.iter().all(|t| t.duration_ns < 1e9),
            "a 256x256 equalize pass taking over a second means the units are wrong: {timings:?}"
        );

        assert!(
            ctx.take_pass_timings().expect("timings").is_empty(),
            "taking the timings should clear them"
        );
    }

    #[test]
    fn the_profiler_stops_recording_when_full_instead_of_failing() {
        let Some(ctx) = crate::gpu::test_support::profiling_context() else {
            eprintln!("Skipping profiler test: no GPU");
            return;
        };
        let Some(profiler) = ctx.profiler() else {
            eprintln!("Skipping profiler test: adapter has no TIMESTAMP_QUERY");
            return;
        };

        // Claim every slot, then keep going. A dispatch past the limit must still be
        // encodable — losing a timing is fine, failing the render is not.
        let mut claimed = 0;
        while profiler.timestamp_writes("filler").is_some() {
            claimed += 1;
            assert!(claimed <= 100_000, "counter never saturated");
        }
        assert!(claimed > 0, "should have claimed at least one slot");
        assert!(
            profiler.timestamp_writes("overflow").is_none(),
            "a full profiler keeps returning None rather than wrapping"
        );
        profiler.reset().expect("reset");
        assert!(
            profiler.timestamp_writes("after_reset").is_some(),
            "reset should free the slots again"
        );
        profiler.reset().expect("reset");
    }
}
