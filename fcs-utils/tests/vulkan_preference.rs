//! A binary of its own because what is under test is how a context comes up. Inside the
//! library's test binary dozens of GPU tests hold devices on the same adapter at once, and there
//! a Vulkan device sometimes failed to come up -- whereupon the fallback to D3D12, which is
//! correct, read as the two checks disagreeing.

use fcs_utils::gpu::{GpuAvailability, GpuContext, GpuContextOptions, vulkan_renderer_available};

/// Two routes to one fact: `init_preferring_vulkan` requests a Vulkan adapter, and the renderer
/// check enumerates them. A context that came up on hardware Vulkan proves the check must say
/// yes. Not the converse: a Vulkan device can still fail to come up, and falling back is then
/// correct. Off Windows the renderer check never offers Vulkan.
#[test]
fn a_hardware_vulkan_context_means_the_renderer_check_says_yes() {
    let options = GpuContextOptions {
        respect_env: false,
        ..GpuContextOptions::default()
    };
    let on_hardware_vulkan = match GpuContext::init_preferring_vulkan(&options) {
        GpuAvailability::Available(ctx) => {
            let info = ctx.adapter_info();
            info.backend == wgpu::Backend::Vulkan && info.device_type != wgpu::DeviceType::Cpu
        }
        _ => false,
    };
    if !cfg!(target_os = "windows") {
        assert!(!vulkan_renderer_available());
    } else if on_hardware_vulkan {
        assert!(vulkan_renderer_available());
    }
}

/// Off Windows there is no DXGI to ask, and Vulkan is never filtered out in the first place.
#[test]
#[cfg(not(target_os = "windows"))]
fn off_windows_vulkan_is_never_called_safe() {
    assert!(!fcs_utils::gpu::vulkan_is_safe());
}
