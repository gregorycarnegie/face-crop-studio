#![no_std]
#![feature(abi_ptx, stdarch_nvptx, asm_experimental_arch)]

fn index() -> u32 {
    use core::arch::nvptx::{_block_dim_x, _block_idx_x, _thread_idx_x};
    // SAFETY: called only inside a CUDA kernel.
    unsafe { (_block_idx_x() * _block_dim_x() + _thread_idx_x()) as u32 }
}

macro_rules! cuda_kernel {
    (fn $name:ident($($arg:ident: $ty:ty),* $(,)?) $body:block) => {
        #[unsafe(no_mangle)]
        pub unsafe extern "ptx-kernel" fn $name($($arg: $ty),*) $body
    };
}

macro_rules! device_fn {
    ($item:item) => { $item };
}
fn exp(x: f32) -> f32 {
    let out: f32;
    unsafe { core::arch::asm!("ex2.approx.f32 {0}, {1};", out(reg32) out, in(reg32) (x * core::f32::consts::LOG2_E)); }
    out
}
fn sqrt(x: f32) -> f32 {
    let out: f32;
    unsafe { core::arch::asm!("sqrt.rn.f32 {0}, {1};", out(reg32) out, in(reg32) x); }
    out
}
fn fma(x: f32, y: f32, z: f32) -> f32 {
    let out: f32;
    unsafe { core::arch::asm!("fma.rn.f32 {0}, {1}, {2}, {3};", out(reg32) out, in(reg32) x, in(reg32) y, in(reg32) z); }
    out
}
unsafe fn atomic_increment(p: *mut u32) -> u32 {
    unsafe { (&*p.cast::<core::sync::atomic::AtomicU32>()).fetch_add(1, core::sync::atomic::Ordering::Relaxed) }
}

include!("../../kernels.rs");
