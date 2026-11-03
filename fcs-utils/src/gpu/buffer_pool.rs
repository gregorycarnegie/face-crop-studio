use std::{
    collections::HashMap,
    fmt,
    sync::{Arc, Mutex},
};

use crate::gpu::GpuContext;
use thiserror::Error;

/// Failure to acquire a buffer from a GPU pool.
#[derive(Debug, Error)]
pub enum BufferPoolError {
    /// A new allocation would exceed the configured pool budget after clearing idle buffers.
    #[error(
        "GPU memory limit exceeded (allocation size: {size}, current usage: {usage}, limit: {limit})"
    )]
    MemoryLimitExceeded {
        /// Requested allocation size in bytes.
        size: u64,
        /// Bytes tracked by the pool after clearing idle buffers.
        usage: u64,
        /// Configured total allocation budget in bytes.
        limit: u64,
    },
    /// A GPU buffer allocation failed with the supplied diagnostic.
    #[error("Failed to create GPU buffer: {0}")]
    AllocationFailed(String),
}

/// Generic over the buffer so the idle accounting can be tested without a GPU.
struct BufferEntry<B = wgpu::Buffer> {
    buffer: B,
    size: u64,
    /// Kept alongside the buffer so a parked entry can be filed back under the right
    /// usage bucket when its execution scope ends.
    usage: wgpu::BufferUsages,
}

/// Idle buffers by usage, with their total size held under the same lock so the two cannot
/// disagree: every change to the lists goes through these methods, which update `bytes` too.
struct IdleBuffers<B = wgpu::Buffer> {
    by_usage: HashMap<wgpu::BufferUsages, Vec<BufferEntry<B>>>,
    bytes: u64,
}

impl<B> Default for IdleBuffers<B> {
    fn default() -> Self {
        Self {
            by_usage: HashMap::new(),
            bytes: 0,
        }
    }
}

impl<B> IdleBuffers<B> {
    fn push(&mut self, entry: BufferEntry<B>) {
        self.bytes += entry.size;
        self.by_usage.entry(entry.usage).or_default().push(entry);
    }

    fn take_best_fit(&mut self, size: u64, usage: wgpu::BufferUsages) -> Option<BufferEntry<B>> {
        let buffers = self.by_usage.get_mut(&usage)?;
        let entry = buffers.swap_remove(best_fit_index(buffers, size, None)?);
        self.bytes -= entry.size;
        Some(entry)
    }

    /// Drop buffers, smallest first, until at most `cap` bytes remain; returns the bytes freed.
    fn evict_to(&mut self, cap: u64) -> u64 {
        let mut freed = 0;
        while self.bytes > cap {
            let Some((usage, index)) = self
                .by_usage
                .iter()
                .flat_map(|(usage, entries)| {
                    entries
                        .iter()
                        .enumerate()
                        .map(move |(i, e)| (*usage, i, e.size))
                })
                .min_by_key(|&(_, _, size)| size)
                .map(|(usage, index, _)| (usage, index))
            else {
                break;
            };
            let entries = self.by_usage.get_mut(&usage).expect("usage was just found");
            let entry = entries.swap_remove(index);
            self.bytes -= entry.size;
            freed += entry.size;
        }
        freed
    }

    /// Drop every buffer; returns the bytes freed.
    fn clear(&mut self) -> u64 {
        self.by_usage.clear();
        std::mem::take(&mut self.bytes)
    }

    fn len(&self) -> usize {
        self.by_usage.values().map(Vec::len).sum()
    }
}

/// A poisoned pool mutex only means some other thread panicked while holding it; the buffer
/// lists themselves are still consistent, so recovering is better than propagating a panic
/// into every later GPU call.
fn lock<T>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
}

thread_local! {
    /// `(pool address, scope id)` for the execution scope active on this thread, if any.
    /// The address disambiguates nested scopes belonging to different pools, since a scope
    /// id is only meaningful to the pool that issued it.
    static ACTIVE_SCOPE: std::cell::Cell<Option<(usize, u64)>> = const { std::cell::Cell::new(None) };
}

/// Ceiling on the bytes held in `idle` when the pool has no explicit `max_memory`.
///
/// Without a bound the pool only ever grows: `take_best_fit` needs a buffer at least as large as
/// the request, so every new high-water size allocates a fresh buffer and the smaller ones are
/// retained forever. A small request also consumes a larger idle buffer, so the larger size
/// re-allocates the next time it appears -- on a folder of mixed image sizes that ratchets upward
/// indefinitely, and every pipeline in this crate builds its pool with `max_memory: None`, so
/// nothing ever triggered `clear`.
///
/// ponytail: one fixed ceiling for every device. Deriving it from actual VRAM would suit a 24 GB
/// discrete card and a shared-memory integrated GPU better, if this ever proves too tight or too
/// slack in practice.
const DEFAULT_MAX_IDLE_BYTES: u64 = 512 << 20;

/// Pixel buffers: uploaded to, read and written by a shader, and copied out for readback.
///
/// One definition for every filter, because the pool is keyed by usage: an acquire and a
/// recycle that spelled the flags differently would file the buffer where it is never reused.
pub(crate) const STORAGE_RW: wgpu::BufferUsages = wgpu::BufferUsages::STORAGE
    .union(wgpu::BufferUsages::COPY_SRC)
    .union(wgpu::BufferUsages::COPY_DST);

/// Host-mappable copy targets for reading results back.
pub(crate) const READBACK: wgpu::BufferUsages =
    wgpu::BufferUsages::MAP_READ.union(wgpu::BufferUsages::COPY_DST);

/// Best-fit GPU buffer pool, grouped by usage flags.
///
/// Return acquired buffers with [`Self::recycle`]. For concurrent encoding,
/// keep an [`Self::execution_scope`] alive until the submitted work completes.
pub struct GpuBufferPool {
    context: Arc<GpuContext>,
    /// Idle buffers and their total, kept under `max_idle_bytes`.
    idle: Mutex<IdleBuffers>,
    /// Buffers released inside an execution scope, keyed by scope id. They are held back from
    /// `idle` until the scope ends, so a buffer whose GPU work is still in flight cannot be
    /// handed to another thread. See [`GpuBufferPool::execution_scope`].
    in_flight: Mutex<HashMap<u64, Vec<BufferEntry>>>,
    next_scope: std::sync::atomic::AtomicU64,
    total_allocated_bytes: std::sync::atomic::AtomicU64,
    max_idle_bytes: u64,
    max_memory: Option<u64>,
}

impl GpuBufferPool {
    /// Create a pool with an optional total allocation budget in bytes.
    /// With `None`, total allocation is uncapped but retained idle buffers are
    /// limited to 512 MiB. Use [`Self::with_idle_limit`] to tune idle retention.
    pub fn new(context: Arc<GpuContext>, max_memory: Option<u64>) -> Self {
        Self {
            context,
            idle: Mutex::new(IdleBuffers::default()),
            in_flight: Mutex::new(HashMap::new()),
            next_scope: std::sync::atomic::AtomicU64::new(0),
            total_allocated_bytes: std::sync::atomic::AtomicU64::new(0),
            // Pools given an explicit `max_memory` are already bounded: `acquire` releases
            // idle buffers once the budget is reached. Imposing an idle ceiling as well would
            // fight that, evicting at the very threshold the pool is meant to work at and
            // freeing buffers the next call immediately re-allocates. The ceiling is the safety
            // net for pools with no budget at all -- which is every pipeline in this crate.
            max_idle_bytes: match max_memory {
                Some(_) => u64::MAX,
                None => DEFAULT_MAX_IDLE_BYTES,
            },
            max_memory,
        }
    }

    /// Build a pool with an explicit ceiling on retained idle bytes.
    ///
    /// Distinct from `max_memory`, which is a hard budget on everything this pool has allocated
    /// and makes `acquire` fail once exceeded. The idle ceiling never fails a request; it only
    /// decides how much is kept for reuse rather than released.
    pub fn with_idle_limit(
        context: Arc<GpuContext>,
        max_memory: Option<u64>,
        max_idle_bytes: u64,
    ) -> Self {
        Self {
            max_idle_bytes,
            ..Self::new(context, max_memory)
        }
    }

    /// Open an execution scope for the calling thread, covering one unit of GPU work from
    /// encoding through to the readback that completes it.
    ///
    /// Buffers are recycled by `Drop` on the host, which happens while a command encoder is
    /// still being built — before anything is submitted. Without a scope those buffers go
    /// straight back to `idle`, and a second thread encoding its own pass can acquire a buffer
    /// that the first thread's not-yet-submitted command buffer already references. Both
    /// submissions then write the same memory, which corrupts detections nondeterministically.
    ///
    /// Inside a scope, released buffers are parked in `in_flight` and only rejoin `idle` when
    /// the scope ends. Reuse *within* the scope is still allowed, and is still correct: the
    /// release and the reuse are encoded into the same command buffer in program order, so the
    /// GPU runs them in that order. Keeping that reuse is what stops peak memory growing by one
    /// allocation per layer.
    ///
    /// The caller must not end the scope until the work is known to have completed. Inference
    /// satisfies this by reading its outputs back before returning, which waits on the queue.
    pub fn execution_scope(&self) -> ExecutionScope<'_> {
        let id = self
            .next_scope
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let key = (self as *const Self as usize, id);
        let previous = ACTIVE_SCOPE.with(|slot| slot.replace(Some(key)));
        ExecutionScope {
            pool: self,
            id,
            previous,
        }
    }

    /// The scope id active on this thread for *this* pool, if any.
    fn active_scope(&self) -> Option<u64> {
        let this = self as *const Self as usize;
        ACTIVE_SCOPE.with(|slot| match slot.get() {
            Some((pool, id)) if pool == this => Some(id),
            _ => None,
        })
    }

    /// Move a finished scope's buffers back into the shared idle pool.
    fn end_scope(&self, id: u64) {
        let parked = {
            let mut in_flight = lock(&self.in_flight);
            in_flight.remove(&id)
        };
        let Some(parked) = parked else { return };
        for entry in parked {
            self.push_idle(entry);
        }
        self.evict_to_cap();
    }

    /// Reuse a matching buffer of at least `size` bytes, or allocate one.
    ///
    /// The buffer has the requested usage flags; reused contents are not cleared.
    /// Return it with [`Self::recycle`] when safe to reuse. Returns an error if a
    /// new allocation would exceed the memory budget after idle buffers are cleared.
    ///
    /// # Panics
    ///
    /// Invalid descriptors or device allocation failures may panic through wgpu.
    pub fn acquire(
        &self,
        size: u64,
        usage: wgpu::BufferUsages,
        label: Option<&str>,
    ) -> Result<wgpu::Buffer, BufferPoolError> {
        if let Some(entry) = self.take_best_fit(size, usage) {
            return Ok(entry.buffer);
        }

        // Reserve before allocating, in one atomic step: a separate check and add let two
        // threads both pass the check and together overshoot the budget.
        let limit = self.max_memory.unwrap_or(u64::MAX);
        if self.try_reserve(size, limit).is_err() {
            // Try to free up space by clearing idle buffers
            self.clear();
            if let Err(usage) = self.try_reserve(size, limit) {
                return Err(BufferPoolError::MemoryLimitExceeded { size, usage, limit });
            }
        }

        // wgpu's default error handler panics on a failed allocation, and the GUI survives that
        // per file, so a reservation left behind would shrink the budget for the whole session.
        let reservation = Reservation {
            counter: &self.total_allocated_bytes,
            size,
        };
        let buffer = self
            .context
            .device()
            .create_buffer(&wgpu::BufferDescriptor {
                label,
                size,
                usage,
                mapped_at_creation: false,
            });
        std::mem::forget(reservation);

        Ok(buffer)
    }

    /// Atomically add `size` to the allocated total if `limit` allows; on refusal, the usage
    /// that refused it.
    fn try_reserve(&self, size: u64, limit: u64) -> Result<u64, u64> {
        use std::sync::atomic::Ordering::Relaxed;
        self.total_allocated_bytes
            .try_update(Relaxed, Relaxed, |current| {
                super::buffer_math::reserve(current, size, limit)
            })
    }

    /// Return a buffer to the pool.
    ///
    /// `size` is what the caller *asked* for, which is not necessarily what it got: `acquire`
    /// satisfies a request from any buffer at least as large, so a 4 MB buffer is routinely
    /// handed out for a 256 KB request. Filing it back under the requested size would relabel a
    /// large buffer as a small one -- it would then stop matching large requests, which allocate
    /// a fresh buffer instead, and the pool grows without ever reusing what it already holds.
    /// The buffer knows its own size, so use that and treat `size` as advisory.
    pub fn recycle(&self, buffer: wgpu::Buffer, size: u64, usage: wgpu::BufferUsages) {
        debug_assert!(
            buffer.size() >= size,
            "recycled buffer is smaller than the size it was acquired for"
        );
        let entry = BufferEntry {
            size: buffer.size(),
            buffer,
            usage,
        };
        // Inside a scope the GPU may still be reading this buffer, so park it until the scope
        // ends rather than offering it to other threads.
        match self.active_scope() {
            Some(scope) => lock(&self.in_flight).entry(scope).or_default().push(entry),
            None => {
                self.push_idle(entry);
                self.evict_to_cap();
            }
        }
    }

    fn push_idle(&self, entry: BufferEntry) {
        lock(&self.idle).push(entry);
    }

    /// Drop idle buffers, smallest first, until the pool holds no more than `max_idle_bytes`.
    ///
    /// Smallest first because `take_best_fit` will serve a small request from a large buffer but
    /// never the reverse, so the large ones are the reusable ones -- evicting those would just
    /// force them to be allocated again.
    fn evict_to_cap(&self) {
        let freed = lock(&self.idle).evict_to(self.max_idle_bytes);
        release(&self.total_allocated_bytes, freed);
    }

    /// Return the number of idle buffers, excluding buffers parked in execution scopes.
    pub fn available(&self) -> usize {
        lock(&self.idle).len()
    }

    /// Returns the total size in bytes of all buffers currently managed (or allocated) by this pool.
    pub fn memory_usage(&self) -> u64 {
        self.total_allocated_bytes
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Clears all idle buffers from the pool, freeing their memory.
    pub fn clear(&self) {
        let freed_bytes = lock(&self.idle).clear();
        // `total_allocated_bytes` counts bytes created and not yet freed through `clear` or
        // eviction. A buffer dropped without `recycle` stays counted, so the budget errs
        // toward refusing, never toward overshooting.
        release(&self.total_allocated_bytes, freed_bytes);
    }

    fn take_best_fit(&self, size: u64, usage: wgpu::BufferUsages) -> Option<BufferEntry> {
        // Prefer this scope's own parked buffers. They are the ones this thread just released,
        // so reusing them is both safe (same command buffer, program order) and what keeps a
        // single inference from allocating a fresh buffer per layer.
        if let Some(scope) = self.active_scope() {
            let mut in_flight = lock(&self.in_flight);
            if let Some(parked) = in_flight.get_mut(&scope)
                && let Some(index) = best_fit_index(parked, size, Some(usage))
            {
                return Some(parked.swap_remove(index));
            }
        }

        lock(&self.idle).take_best_fit(size, usage)
    }
}

/// Subtract `freed` from a byte counter in one atomic step, saturating at zero.
fn release(counter: &std::sync::atomic::AtomicU64, freed: u64) {
    use std::sync::atomic::Ordering::Relaxed;
    counter.update(Relaxed, Relaxed, |current| {
        super::buffer_math::release(current, freed)
    });
}

/// A budget reservation, given back on drop unless forgotten once the allocation succeeds.
struct Reservation<'a> {
    counter: &'a std::sync::atomic::AtomicU64,
    size: u64,
}

impl Drop for Reservation<'_> {
    fn drop(&mut self) {
        release(self.counter, self.size);
    }
}

/// Index of the smallest entry that is at least `size`, optionally restricted to one usage.
/// The usage filter is needed for the scope-parked list, which is keyed by scope rather than
/// by usage and so holds mixed usages.
fn best_fit_index<B>(
    entries: &[BufferEntry<B>],
    size: u64,
    usage: Option<wgpu::BufferUsages>,
) -> Option<usize> {
    let mut best = None;

    for (index, entry) in entries.iter().enumerate() {
        if usage.is_some_and(|u| entry.usage != u) {
            continue;
        }
        best = super::buffer_math::choose_fit(best, index, entry.size, size);
        if entry.size == size {
            break;
        }
    }

    best.map(|(index, _)| index)
}

/// Guard returned by [`GpuBufferPool::execution_scope`]. Ending it releases the buffers the
/// scope parked back into the shared pool.
pub struct ExecutionScope<'a> {
    pool: &'a GpuBufferPool,
    id: u64,
    previous: Option<(usize, u64)>,
}

impl Drop for ExecutionScope<'_> {
    fn drop(&mut self) {
        ACTIVE_SCOPE.with(|slot| slot.set(self.previous));
        self.pool.end_scope(self.id);
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

    use crate::gpu::test_support::test_context;

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

    /// A buffer handed out for a small request must go back under its own size.
    ///
    /// `acquire` satisfies a request from any buffer at least as large, so callers routinely
    /// recycle a big buffer citing the small size they asked for. Filing it under that size
    /// relabels it as small, so it stops matching large requests and the pool allocates a fresh
    /// large buffer every time one comes round -- unbounded growth on a folder of mixed sizes.
    #[test]
    fn a_reused_buffer_keeps_its_real_size() {
        let Some(ctx) = test_context() else {
            eprintln!("Skipping buffer_pool test: no GPU");
            return;
        };
        let pool = GpuBufferPool::new(ctx, None);

        let big = pool.acquire(8192, STORAGE, None).expect("acquire");
        pool.recycle(big, 8192, STORAGE);

        // Ask for far less than is pooled; best fit hands over the 8192 buffer.
        let reused = pool.acquire(512, STORAGE, None).expect("acquire");
        assert_eq!(
            reused.size(),
            8192,
            "the pooled buffer should have been reused"
        );
        // Recycled citing the *requested* size, as every caller in this crate does.
        pool.recycle(reused, 512, STORAGE);

        // If it had been filed as 512 this would allocate a second buffer.
        let big_again = pool.acquire(8192, STORAGE, None).expect("acquire");
        assert_eq!(
            pool.available(),
            0,
            "the 8192 buffer should still satisfy an 8192 request after a small reuse"
        );
        pool.recycle(big_again, 8192, STORAGE);
    }

    /// Retained buffers stay under the idle ceiling however many sizes go through the pool.
    #[test]
    fn idle_memory_stays_under_the_ceiling() {
        let Some(ctx) = test_context() else {
            eprintln!("Skipping buffer_pool test: no GPU");
            return;
        };
        // The sequence below retains up to 49 KiB, so this ceiling makes eviction actually run.
        let cap = 32 * 1024;
        let pool = GpuBufferPool::with_idle_limit(ctx, None, cap);

        // Mixed sizes in a repeating order, which is what ratcheted the pool upward before.
        for _ in 0..6 {
            for multiple in [3u64, 17, 5, 29, 8, 21, 2, 13] {
                let size = multiple * 1024;
                let buffer = pool.acquire(size, STORAGE, None).expect("acquire");
                pool.recycle(buffer, size, STORAGE);
            }
        }

        assert!(
            pool.memory_usage() <= cap,
            "retained {} bytes, ceiling is {cap}",
            pool.memory_usage()
        );
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

    /// Racing acquires must never jointly exceed the budget: the check and the add are one step.
    #[test]
    fn concurrent_acquires_stay_within_the_budget() {
        let Some(ctx) = test_context() else {
            eprintln!("Skipping GPU memory test: no GPU");
            return;
        };
        let pool = GpuBufferPool::new(ctx, Some(4 * 256));
        let barrier = std::sync::Barrier::new(16);
        let granted: Vec<_> = std::thread::scope(|s| {
            let handles: Vec<_> = (0..16)
                .map(|_| {
                    s.spawn(|| {
                        barrier.wait();
                        pool.acquire(256, STORAGE, None).ok()
                    })
                })
                .collect();
            handles
                .into_iter()
                .filter_map(|h| h.join().unwrap())
                .collect()
        });
        assert_eq!(granted.len(), 4, "exactly the budget's worth is granted");
        assert_eq!(pool.memory_usage(), 4 * 256);
    }

    /// A buffer wgpu refuses (larger than the device allows) panics inside `create_buffer`;
    /// once that panic is caught, as the GUI does per file, the reservation must be gone.
    #[test]
    fn a_panicking_allocation_gives_its_reservation_back() {
        let Some(ctx) = test_context() else {
            eprintln!("Skipping GPU memory test: no GPU");
            return;
        };
        let too_big = ctx.device().limits().max_buffer_size + 4;
        let pool = GpuBufferPool::new(ctx, Some(u64::MAX));
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            pool.acquire(too_big, STORAGE, None)
        }));
        if result.is_ok() {
            eprintln!("Skipping: this backend did not panic on an oversized buffer");
            return;
        }
        assert_eq!(pool.memory_usage(), 0);
    }

    // ------------------------------------------------------------------
    // `take_best_fit` decides which idle buffer a request reuses. A wrong choice
    // never shows up in output pixels — only as extra allocations, or a buffer
    // too small for the request — so these assert on the pool's own accounting.
    //
    // `available()`, the count of idle buffers, is the observable that matters.
    // `memory_usage()` cannot distinguish a pool hit from an allocation: it
    // counts bytes ever created minus bytes cleared, and `recycle` leaves it
    // untouched. The long comment in `clear()` wrestles with the same ambiguity.
    // ------------------------------------------------------------------

    const STORAGE: wgpu::BufferUsages = wgpu::BufferUsages::STORAGE;

    #[test]
    fn a_request_reuses_an_idle_buffer_rather_than_allocating() {
        let Some(ctx) = test_context() else {
            eprintln!("Skipping buffer_pool test: no GPU");
            return;
        };
        let pool = GpuBufferPool::new(ctx, None);

        let buf = pool.acquire(1024, STORAGE, Some("first")).expect("acquire");
        pool.recycle(buf, 1024, STORAGE);
        assert_eq!(pool.available(), 1, "recycled buffer should be pooled");

        let reused = pool
            .acquire(1024, STORAGE, Some("reused"))
            .expect("acquire");
        assert_eq!(
            pool.available(),
            0,
            "an exact-size match must come from the pool, not a fresh allocation"
        );
        pool.recycle(reused, 1024, STORAGE);
    }

    #[test]
    fn an_idle_buffer_smaller_than_the_request_is_not_reused() {
        let Some(ctx) = test_context() else {
            eprintln!("Skipping buffer_pool test: no GPU");
            return;
        };
        let pool = GpuBufferPool::new(ctx, None);

        let small = pool.acquire(256, STORAGE, Some("small")).expect("acquire");
        pool.recycle(small, 256, STORAGE);

        // Handing back the 256-byte buffer would under-run every write into it.
        let big = pool.acquire(4096, STORAGE, Some("big")).expect("acquire");
        assert_eq!(
            pool.available(),
            1,
            "the too-small buffer must stay in the pool"
        );
        pool.recycle(big, 4096, STORAGE);
    }

    /// Which buffer was taken is not directly observable, so it is inferred from
    /// a follow-up request: after asking for 1024 from {512, 2048, 8192}, a
    /// request for 8192 can only be served from the pool if 2048 was consumed.
    #[test]
    fn the_smallest_sufficient_buffer_wins() {
        let Some(ctx) = test_context() else {
            eprintln!("Skipping buffer_pool test: no GPU");
            return;
        };
        let pool = GpuBufferPool::new(ctx, None);

        // Held together so each is a fresh allocation, then recycled largest first: stopping at
        // the first sufficient buffer would then take 8192 rather than 2048.
        let held: Vec<_> = [8192u64, 2048, 512]
            .map(|size| (pool.acquire(size, STORAGE, None).expect("acquire"), size))
            .into();
        for (b, size) in held {
            pool.recycle(b, size, STORAGE);
        }
        assert_eq!(pool.available(), 3);

        // 512 is too small, so the best fit is 2048 — not 8192.
        let got = pool.acquire(1024, STORAGE, None).expect("acquire");
        assert_eq!(pool.available(), 2, "one buffer should have been consumed");

        let big = pool.acquire(8192, STORAGE, None).expect("acquire");
        assert_eq!(
            pool.available(),
            1,
            "8192 should still have been pooled, proving 2048 was the best fit"
        );

        pool.recycle(got, 2048, STORAGE);
        pool.recycle(big, 8192, STORAGE);
    }

    #[test]
    fn buffers_are_only_reused_for_a_matching_usage() {
        let Some(ctx) = test_context() else {
            eprintln!("Skipping buffer_pool test: no GPU");
            return;
        };
        let pool = GpuBufferPool::new(ctx, None);

        let storage = pool.acquire(1024, STORAGE, None).expect("acquire");
        pool.recycle(storage, 1024, STORAGE);

        // A different usage cannot bind the same buffer.
        let other = READBACK;
        let mapped = pool.acquire(1024, other, None).expect("acquire");
        assert_eq!(
            pool.available(),
            1,
            "a STORAGE buffer must not satisfy a MAP_READ request"
        );
        pool.recycle(mapped, 1024, other);
    }

    #[test]
    fn an_empty_pool_allocates_and_clear_empties_it() {
        let Some(ctx) = test_context() else {
            eprintln!("Skipping buffer_pool test: no GPU");
            return;
        };
        let pool = GpuBufferPool::new(ctx, None);

        assert_eq!(pool.available(), 0);
        let buf = pool.acquire(1024, STORAGE, None).expect("acquire");
        assert_eq!(pool.available(), 0, "nothing was pooled to reuse");
        assert_eq!(
            pool.memory_usage(),
            1024,
            "creating a buffer should be accounted"
        );

        pool.recycle(buf, 1024, STORAGE);
        assert_eq!(pool.available(), 1);
        pool.clear();
        assert_eq!(pool.available(), 0, "clear must drop pooled buffers");
        assert_eq!(
            pool.memory_usage(),
            0,
            "clear must also release their accounted bytes"
        );
    }

    /// The limit check is `current + size > limit`, so a request landing exactly
    /// on the limit is allowed and one byte over is refused.
    #[test]
    fn the_memory_limit_admits_an_exact_fit_and_rejects_an_overshoot() {
        let Some(ctx) = test_context() else {
            eprintln!("Skipping buffer_pool test: no GPU");
            return;
        };
        let pool = GpuBufferPool::new(ctx, Some(2048));

        let exact = pool
            .acquire(2048, STORAGE, None)
            .expect("an exact fit is allowed");
        pool.recycle(exact, 2048, STORAGE);

        assert!(
            pool.acquire(4096, STORAGE, None).is_err(),
            "a request larger than the whole limit must be refused"
        );
    }

    /// Three 4096-byte buffers against an 8192 ceiling: exactly one is evicted, not zero or two.
    #[test]
    fn eviction_stops_as_soon_as_idle_bytes_fit_the_ceiling() {
        let Some(ctx) = test_context() else {
            return;
        };
        let pool = GpuBufferPool::with_idle_limit(ctx, None, 8192);
        // Held together so each is a fresh allocation, then returned together.
        let held: Vec<_> = (0..3)
            .map(|_| pool.acquire(4096, STORAGE, None).expect("acquire"))
            .collect();
        for buffer in held {
            pool.recycle(buffer, 4096, STORAGE);
        }
        assert_eq!(
            pool.available(),
            2,
            "12288 idle bytes over an 8192 ceiling evicts exactly one buffer"
        );
        assert_eq!(
            pool.memory_usage(),
            8192,
            "the evicted buffer's bytes are released too"
        );
    }

    /// A request landing exactly on the budget keeps idle buffers; one that overshoots clears
    /// them first and is then allowed if it fits exactly.
    #[test]
    fn the_budget_clears_idle_buffers_only_on_an_overshoot() {
        let Some(ctx) = test_context() else {
            return;
        };
        let pool = GpuBufferPool::new(ctx, Some(2048));
        // Parked under another usage, so it can never be reused for a STORAGE request.
        let parked = pool.acquire(1024, READBACK, None).expect("acquire");
        pool.recycle(parked, 1024, READBACK);

        // 1024 held + 1024 asked = 2048, exactly the budget.
        let a = pool
            .acquire(1024, STORAGE, None)
            .expect("an exact fit is allowed");
        assert_eq!(
            pool.available(),
            1,
            "reaching the budget exactly must not clear idle buffers"
        );
        pool.recycle(a, 1024, STORAGE);

        // 2048 held + 2048 asked overshoots; clearing both idle buffers makes it an exact fit.
        let b = pool
            .acquire(2048, STORAGE, None)
            .expect("fits once idle buffers are cleared");
        assert_eq!(pool.available(), 0);
        pool.recycle(b, 2048, STORAGE);
    }

    /// Buffers released inside a scope are parked until it ends, and only for the pool that
    /// opened it.
    #[test]
    fn a_scope_parks_its_own_pools_buffers_until_it_ends() {
        let Some(ctx) = test_context() else {
            return;
        };
        let pool = GpuBufferPool::new(ctx.clone(), None);
        let bystander = GpuBufferPool::new(ctx, None);

        let scope = pool.execution_scope();
        let b = pool.acquire(1024, STORAGE, None).expect("acquire");
        pool.recycle(b, 1024, STORAGE);
        assert_eq!(
            pool.available(),
            0,
            "released inside the scope, so parked rather than idle"
        );

        // The scope belongs to `pool`; another pool on the same thread recycles straight to idle.
        let o = bystander.acquire(1024, STORAGE, None).expect("acquire");
        bystander.recycle(o, 1024, STORAGE);
        assert_eq!(
            bystander.available(),
            1,
            "a scope must not capture another pool's buffers"
        );

        // The parked list mixes usages; a MAP_READ buffer must not be handed out for STORAGE.
        let readback = pool.acquire(1024, READBACK, None).expect("acquire");
        pool.recycle(readback, 1024, READBACK);
        let storage = pool.acquire(1024, STORAGE, None).expect("acquire");
        assert_eq!(
            storage.usage(),
            STORAGE,
            "a parked buffer of another usage was reused"
        );
        pool.recycle(storage, 1024, STORAGE);

        drop(scope);
        assert_eq!(
            pool.available(),
            2,
            "ending the scope returns its parked buffers to idle"
        );
    }

    #[derive(Debug, Clone)]
    enum IdleOp {
        Push(bool, u64),
        Take(bool, u64),
        Evict(u64),
        Clear,
    }

    fn idle_op() -> impl proptest::strategy::Strategy<Value = IdleOp> {
        use proptest::prelude::*;
        prop_oneof![
            4 => (any::<bool>(), 0u64..1000).prop_map(|(u, s)| IdleOp::Push(u, s)),
            3 => (any::<bool>(), 0u64..1000).prop_map(|(u, s)| IdleOp::Take(u, s)),
            2 => (0u64..3000).prop_map(IdleOp::Evict),
            1 => Just(IdleOp::Clear),
        ]
    }

    proptest::proptest! {
        /// Any sequence of idle-list operations, without a GPU, against a plain list model:
        /// the byte total always equals the sizes held, a take is the smallest sufficient
        /// buffer of its usage, and eviction frees smallest first down to the ceiling.
        #[test]
        fn idle_bytes_always_equal_the_buffers_held(
            ops in proptest::collection::vec(idle_op(), 0..60),
        ) {
            use proptest::prop_assert_eq;
            let usage = |u: bool| if u { READBACK } else { STORAGE_RW };
            let mut idle = IdleBuffers::<()>::default();
            let mut model: Vec<(wgpu::BufferUsages, u64)> = Vec::new();
            for op in ops {
                match op {
                    IdleOp::Push(u, size) => {
                        idle.push(BufferEntry { buffer: (), size, usage: usage(u) });
                        model.push((usage(u), size));
                    }
                    IdleOp::Take(u, size) => {
                        let want = model
                            .iter()
                            .filter(|&&(m, s)| m == usage(u) && s >= size)
                            .map(|&(_, s)| s)
                            .min();
                        let got = idle.take_best_fit(size, usage(u)).map(|e| e.size);
                        prop_assert_eq!(got, want);
                        if let Some(s) = got {
                            let at = model.iter().position(|&e| e == (usage(u), s)).unwrap();
                            model.swap_remove(at);
                        }
                    }
                    IdleOp::Evict(cap) => {
                        let before: u64 = model.iter().map(|e| e.1).sum();
                        let freed = idle.evict_to(cap);
                        model.sort_by_key(|e| e.1);
                        let mut dropped = 0;
                        while model.iter().map(|e| e.1).sum::<u64>() > cap {
                            dropped += model.remove(0).1;
                        }
                        prop_assert_eq!(freed, dropped);
                        prop_assert_eq!(before - freed, idle.bytes);
                    }
                    IdleOp::Clear => {
                        let held: u64 = model.drain(..).map(|e| e.1).sum();
                        prop_assert_eq!(idle.clear(), held);
                    }
                }
                prop_assert_eq!(idle.bytes, model.iter().map(|e| e.1).sum::<u64>());
                prop_assert_eq!(idle.len(), model.len());
            }
        }
    }
}
