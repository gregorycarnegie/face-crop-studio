#[cfg(any(target_os = "windows", target_os = "macos"))]
use log::debug;

/// Query the available VRAM (in bytes) on the system.
///
/// This provides a "budget" that the OS considers safe for the application to use.
/// - **Windows**: Queries DXGI for the local video memory budget.
/// - **macOS**: Queries Metal for the recommended max working set size (unified memory).
/// - **Linux/Other**: Returns `None` (implementing reliable VRAM queries via Vulkan/sysfs is complex).
pub fn get_available_vram() -> Option<u64> {
    #[cfg(target_os = "windows")]
    return get_vram_windows();

    #[cfg(target_os = "macos")]
    return get_vram_macos();

    #[cfg(not(any(target_os = "windows", target_os = "macos")))]
    None
}

#[cfg(target_os = "windows")]
fn get_vram_windows() -> Option<u64> {
    use windows::{Win32::Graphics::Dxgi::*, core::Interface};

    // Safety: FFI calls to DXGI. We act conservatively and catch errors as Option::None.
    unsafe {
        let factory_result: windows::core::Result<IDXGIFactory4> =
            CreateDXGIFactory2(DXGI_CREATE_FACTORY_FLAGS(0));
        let factory = match factory_result {
            Ok(f) => f,
            Err(e) => {
                debug!("DXGI factory creation failed: {e}");
                return None;
            }
        };

        let adapter_result: windows::core::Result<IDXGIAdapter3> =
            factory.EnumAdapters1(0).and_then(|a| a.cast());
        let adapter = match adapter_result {
            Ok(a) => a,
            Err(e) => {
                debug!("DXGI adapter enumeration/cast failed: {e}");
                return None;
            }
        };

        let mut info = DXGI_QUERY_VIDEO_MEMORY_INFO::default();
        if let Err(e) = adapter.QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, &mut info)
        {
            debug!("QueryVideoMemoryInfo failed: {e}");
            return None;
        }

        // Available = Budget - CurrentUsage.
        // We use saturating_sub just to be safe against race conditions where usage > budget temporarily.
        let available = info.Budget.saturating_sub(info.CurrentUsage);
        debug!(
            "DXGI VRAM Budget: {} MB, Usage: {} MB, Available: {} MB",
            info.Budget >> 20, // bytes to MiB
            info.CurrentUsage >> 20,
            available >> 20
        );
        Some(available)
    }
}

#[cfg(target_os = "macos")]
fn get_vram_macos() -> Option<u64> {
    // metal-rs crate is required
    let device = metal::Device::system_default()?;

    // recommendedMaxWorkingSetSize is available on macOS 10.12+ / iOS 10.0+
    // It returns the approximate limit of memory the app can use before the OS starts aggressive paging/killing.
    let budget = device.recommended_max_working_set_size();
    let used = device.current_allocated_size();

    let available = budget.saturating_sub(used);
    debug!(
        "Metal Memory Budget: {} MB, Usage: {} MB, Available: {} MB",
        budget >> 20, // bytes to MiB
        used >> 20,
        available >> 20
    );
    Some(available)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vram_query_does_not_panic() {
        // This test just ensures the FFI calls don't crash.
        // It's acceptable for it to return None on CI/headless environments.
        let vram = get_available_vram();
        if let Some(bytes) = vram {
            println!("Detected VRAM: {} MB", bytes >> 20); // bytes to MiB
            assert!(bytes > 0, "VRAM should be positive");
        } else {
            println!("VRAM query returned None (expected on some platforms/configs)");
        }
    }
}
