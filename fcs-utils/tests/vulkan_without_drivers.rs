//! A binary of its own because it changes the environment the Vulkan loader reads, which no
//! other test may see.

/// A machine whose only Vulkan "devices" fail to load -- a VM, a Remote Desktop session -- must
/// keep its renderer on D3D12, or eframe has nothing to fall back to.
#[test]
fn no_vulkan_driver_means_no_vulkan_renderer() {
    // SAFETY: the only test in this binary, so no other thread reads the environment. The
    // loader honours this from 1.3.234 and reads it whenever an instance is created.
    unsafe { std::env::set_var("VK_LOADER_DRIVERS_DISABLE", "*") };
    assert!(!fcs_utils::gpu::vulkan_renderer_available());
}
