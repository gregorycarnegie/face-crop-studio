#[cuda_device::device]
fn index() -> u32 {
    cuda_device::thread::index_1d().get() as u32
}

macro_rules! cuda_kernel {
    (fn $name:ident($($arg:ident: $ty:ty),* $(,)?) $body:block) => {
        #[cuda_device::kernel]
        pub unsafe fn $name($($arg: $ty),*) $body
    };
}

macro_rules! device_fn {
    ($item:item) => { #[cuda_device::device] $item };
}
#[cuda_device::device]
fn exp(x: f32) -> f32 { cuda_device::float::ex2_approx_f32(x * core::f32::consts::LOG2_E) }
#[cuda_device::device]
fn sqrt(x: f32) -> f32 { cuda_device::float::sqrt_rn_f32(x) }
#[cuda_device::device]
fn fma(x: f32, y: f32, z: f32) -> f32 { cuda_device::float::fma_rn_f32(x, y, z) }
#[cuda_device::device]
unsafe fn atomic_increment(p: *mut u32) {
    use cuda_device::atomic::{DeviceAtomicU32, AtomicOrdering};
    unsafe { (&*p.cast::<DeviceAtomicU32>()).fetch_add(1, AtomicOrdering::Relaxed); }
}

include!("../../kernels.rs");

fn main() {}
