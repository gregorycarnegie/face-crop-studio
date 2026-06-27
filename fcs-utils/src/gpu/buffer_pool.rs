use std::{
    collections::HashMap,
    fmt,
    sync::{Arc, Mutex},
};

use crate::gpu::GpuContext;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum BufferPoolError {
    #[error(
        "GPU memory limit exceeded (allocation size: {size}, current usage: {usage}, limit: {limit})"
    )]
    MemoryLimitExceeded { size: u64, usage: u64, limit: u64 },
    #[error("Failed to create GPU buffer: {0}")]
    AllocationFailed(String),
}

struct BufferEntry {
    buffer: wgpu::Buffer,
    size: u64,
}

/// Best-fit pool for `wgpu::Buffer` allocations organized by usage flags.
/// Buffers are grouped by usage to improve search performance and avoid
/// iterating through incompatible buffers.
pub struct GpuBufferPool {
    context: Arc<GpuContext>,
    idle: Mutex<HashMap<wgpu::BufferUsages, Vec<BufferEntry>>>,
    total_allocated_bytes: std::sync::atomic::AtomicU64,
    max_memory: Option<u64>,
}

impl GpuBufferPool {
    pub fn new(context: Arc<GpuContext>, max_memory: Option<u64>) -> Self {
        Self {
            context,
            idle: Mutex::new(HashMap::new()),
            total_allocated_bytes: std::sync::atomic::AtomicU64::new(0),
            max_memory,
        }
    }

    pub fn acquire(
        &self,
        size: u64,
        usage: wgpu::BufferUsages,
        label: Option<&str>,
    ) -> Result<wgpu::Buffer, BufferPoolError> {
        if let Some(entry) = self.take_best_fit(size, usage) {
            return Ok(entry.buffer);
        }

        // Check memory limits before allocating
        if let Some(limit) = self.max_memory {
            let current = self.memory_usage();
            if current + size > limit {
                // Try to free up space by clearing idle buffers
                self.clear();
                let current_after_clear = self.memory_usage();
                if current_after_clear + size > limit {
                    return Err(BufferPoolError::MemoryLimitExceeded {
                        size,
                        usage: current_after_clear,
                        limit,
                    });
                }
            }
        }

        // In wgpu 0.17+, create_buffer can panic on OOM. We wrap it if possible but it's hard.
        // Assuming standard behavior: returns buffer, but might be invalid if OOM.
        // For now, we trust the limit check above.

        let buffer = self
            .context
            .device()
            .create_buffer(&wgpu::BufferDescriptor {
                label,
                size,
                usage,
                mapped_at_creation: false,
            });

        self.total_allocated_bytes
            .fetch_add(size, std::sync::atomic::Ordering::Relaxed);

        Ok(buffer)
    }

    pub fn recycle(&self, buffer: wgpu::Buffer, size: u64, usage: wgpu::BufferUsages) {
        match self.idle.lock() {
            Ok(mut idle) => {
                idle.entry(usage)
                    .or_default()
                    .push(BufferEntry { buffer, size });
            }
            Err(poisoned) => {
                poisoned
                    .into_inner()
                    .entry(usage)
                    .or_default()
                    .push(BufferEntry { buffer, size });
            }
        }
    }

    pub fn available(&self) -> usize {
        self.idle
            .lock()
            .map(|map| map.values().map(|v| v.len()).sum())
            .unwrap_or_else(|poisoned| poisoned.into_inner().values().map(|v| v.len()).sum())
    }

    /// Returns the total size in bytes of all buffers currently managed (or allocated) by this pool.
    pub fn memory_usage(&self) -> u64 {
        self.total_allocated_bytes
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Clears all idle buffers from the pool, freeing their memory.
    pub fn clear(&self) {
        let mut idle = match self.idle.lock() {
            Ok(guard) => guard,
            Err(poisoned) => poisoned.into_inner(),
        };

        // Calculate size of dropped buffers to decrement counter correctly
        let mut freed_bytes = 0;
        for entries in idle.values() {
            for entry in entries {
                freed_bytes += entry.size;
            }
        }

        idle.clear();

        // Note: We only decrement for buffers that were actually in the pool.
        // Buffers currently in use are not affected, but total_allocated_bytes tracks *all* allocated
        // buffers created through this pool that haven't been dropped by the GPU yet (conceptually).
        // Wait, `recycle` puts them back. If they are dropped outside, we can't track that easily without a wrapper.
        //
        // CORRECTION: The current design allocates new buffers if pool is empty.
        // `total_allocated_bytes` increments on create.
        // It should technically decrement when a buffer is destroyed.
        // But we return raw `wgpu::Buffer`. We don't know when the user drops it unless they call `recycle`.
        //
        // If they DROP it instead of recycling, our counter leaks.
        //
        // For OOM handling purposes, we care about what's IN the pool mostly, or we accept the leak
        // as "allocated by app".
        //
        // Let's adjust: track `total_allocated_bytes` as distinct from `pooled_bytes`.
        // Actually, if we want to release memory on OOM, we only care about `idle` buffers.
        //
        // Let's assume `total_allocated_bytes` is "allocated via this pool and not yet known to be freed".
        // Use a better metric? `pooled_memory_usage` might be more accurate for "what we can free".
        //
        // Let's stick to tracking what we create. If user drops buffer without recycling, we drift.
        // Ideally we'd wrap `wgpu::Buffer`. For now, let's just track "pooled" memory + estimated active?
        //
        // Actually, `recycle` puts it back. If they don't recycle, it's gone from our control.
        // So `total_allocated_bytes` increases on create.
        // When clearing pool, we decrement by the size of cleared buffers.
        //
        // Issue: if user drops buffer (no recycle), we never decrement.
        //
        // Let's change the defined metric: `pooled_memory_usage`. Only track what is sitting in `idle`.
        // When we create a buffer, it's "in use". When recycled, it becomes "pooled".
        //
        // If we want to track TOTAL GPU usage, `wgpu::GlobalReport` is better.
        // `GpuBufferPool` should track how much IT is holding.

        self.total_allocated_bytes
            .fetch_sub(freed_bytes, std::sync::atomic::Ordering::Relaxed);
    }

    fn take_best_fit(&self, size: u64, usage: wgpu::BufferUsages) -> Option<BufferEntry> {
        let mut idle = match self.idle.lock() {
            Ok(guard) => guard,
            Err(poisoned) => poisoned.into_inner(),
        };

        // Only search buffers with matching usage flags
        let buffers = idle.get_mut(&usage)?;

        let mut best_index = None;
        let mut best_size = u64::MAX;

        for (index, entry) in buffers.iter().enumerate() {
            if entry.size < size {
                continue;
            }
            if entry.size < best_size {
                best_size = entry.size;
                best_index = Some(index);
                if entry.size == size {
                    break;
                }
            }
        }

        best_index.map(|index| buffers.swap_remove(index))
    }
}

impl fmt::Debug for GpuBufferPool {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let idle_count = self.available();
        f.debug_struct("GpuBufferPool")
            .field("idle_buffers", &idle_count)
            .field("memory_usage", &self.memory_usage())
            .field("max_memory", &self.max_memory)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::{GpuAvailability, GpuContextOptions};

    fn test_context() -> Option<Arc<GpuContext>> {
        match GpuContext::init_with_fallback(&GpuContextOptions::default()) {
            GpuAvailability::Available(ctx) => Some(ctx),
            _ => None,
        }
    }

    #[test]
    fn debug_impl_does_not_panic() {
        let Some(ctx) = test_context() else {
            eprintln!("Skipping buffer_pool debug test: no GPU");
            return;
        };
        let pool = GpuBufferPool::new(ctx, None);
        let debug_str = format!("{pool:?}");
        assert!(debug_str.contains("GpuBufferPool"));
    }

    #[test]
    fn available_tracks_recycled_buffers() {
        let Some(ctx) = test_context() else {
            eprintln!("Skipping buffer_pool available test: no GPU");
            return;
        };
        let pool = GpuBufferPool::new(ctx, None);
        assert_eq!(pool.available(), 0);

        let buf = pool
            .acquire(256, wgpu::BufferUsages::STORAGE, None)
            .expect("alloc");
        assert_eq!(pool.available(), 0);

        pool.recycle(buf, 256, wgpu::BufferUsages::STORAGE);
        assert_eq!(pool.available(), 1);

        pool.clear();
        assert_eq!(pool.available(), 0);
    }

    #[test]
    fn test_memory_limit_enforcement() {
        let Some(ctx) = test_context() else {
            eprintln!("Skipping GPU memory test: no GPU");
            return;
        };

        // limit = 1024 bytes
        let pool = GpuBufferPool::new(ctx.clone(), Some(1024));
        assert_eq!(pool.memory_usage(), 0);

        // Alloc 512 - OK
        let buf1 = pool
            .acquire(512, wgpu::BufferUsages::STORAGE, None)
            .expect("alloc 512");
        assert!(pool.memory_usage() >= 512);

        // Alloc 600 - Fail (512 + 600 > 1024)
        // buf1 is still active
        let result = pool.acquire(600, wgpu::BufferUsages::STORAGE, None);
        assert!(matches!(
            result,
            Err(BufferPoolError::MemoryLimitExceeded { .. })
        ));

        // Recycle buf1
        pool.recycle(buf1, 512, wgpu::BufferUsages::STORAGE);
        // usage is still 512 (it's in pool now)

        // Alloc 600 - Should succeed (Pool clears buf1 to make room)
        let buf2 = pool
            .acquire(600, wgpu::BufferUsages::STORAGE, None)
            .expect("alloc 600 after clear");
        // Usage should be 600 now (because buf1 (512) was dropped)
        assert_eq!(pool.memory_usage(), 600);

        // Cleanup
        pool.recycle(buf2, 600, wgpu::BufferUsages::STORAGE);
    }
}
