//! Process environment the bundled C libraries can see.
//!
//! Native code (libheif, Qt) reads variables with the C runtime's `getenv`. On Windows that is
//! the CRT's own copy of the environment, which [`std::env::set_var`] (`SetEnvironmentVariableW`)
//! never updates, so a variable set from Rust is invisible to it. [`set_default`] goes through
//! the CRT's `_wputenv_s`, which updates both copies.

use std::ffi::OsStr;

/// Set `name` to `value` for Rust and the C runtime alike, unless it is already set.
///
/// # Safety
///
/// Call it before any other thread starts. Like [`std::env::set_var`], it races with any
/// concurrent read of the environment.
pub unsafe fn set_default(name: &str, value: &OsStr) {
    if std::env::var_os(name).is_some() {
        return;
    }
    #[cfg(windows)]
    {
        use std::os::windows::ffi::OsStrExt;
        unsafe extern "C" {
            fn _wputenv_s(name: *const u16, value: *const u16) -> i32;
        }
        let wide = |s: &OsStr| s.encode_wide().chain([0]).collect::<Vec<u16>>();
        let (name, value) = (wide(name.as_ref()), wide(value));
        // SAFETY: both are NUL-terminated; the caller guarantees no other thread is running.
        if unsafe { _wputenv_s(name.as_ptr(), value.as_ptr()) } != 0 {
            log::warn!("Failed to set {}", String::from_utf16_lossy(&name));
        }
    }
    // glibc's and libSystem's getenv read the environment set_var writes.
    #[cfg(not(windows))]
    // SAFETY: the caller guarantees no other thread is running.
    unsafe {
        std::env::set_var(name, value)
    };
}

/// Keep libheif from loading plugins out of the root of the current drive.
///
/// vcpkg builds libheif with an empty plugin directory, and on Windows `heif_init` then calls
/// `LoadLibrary` on every `*.dll` in `\` (`C:\*.dll`), each time a HEIC is decoded: a DLL
/// planted there would run in the process. This points `LIBHEIF_PLUGIN_PATH` at a
/// `libheif-plugins` folder beside the executable instead, unless it is already set. The
/// vcpkg build has its HEVC decoder built in, so it needs no plugins. Elsewhere it does
/// nothing: distro and Homebrew builds have a real plugin directory, and Linux packages often
/// put the HEVC decoder itself in a plugin.
///
/// # Safety
///
/// As [`set_default`]: call it before any other thread starts.
pub unsafe fn confine_libheif_plugins() {
    #[cfg(windows)]
    if let Some(dir) = std::env::current_exe()
        .ok()
        .and_then(|exe| exe.parent().map(|dir| dir.join("libheif-plugins")))
    {
        // SAFETY: forwarded from the caller.
        unsafe { set_default("LIBHEIF_PLUGIN_PATH", dir.as_os_str()) };
    }
}

#[cfg(all(test, windows))]
mod tests {
    use super::*;

    unsafe extern "C" {
        fn getenv(name: *const std::ffi::c_char) -> *const std::ffi::c_char;
    }

    fn crt_getenv(name: &std::ffi::CStr) -> Option<String> {
        // SAFETY: a NUL-terminated name; the result is copied before anything else runs.
        let value = unsafe { getenv(name.as_ptr()) };
        // SAFETY: getenv returns null or a NUL-terminated string.
        (!value.is_null()).then(|| {
            unsafe { std::ffi::CStr::from_ptr(value) }
                .to_string_lossy()
                .into()
        })
    }

    /// The bug this module exists for: C code must see the variable, and an existing value
    /// must win. A name no other test touches; the CRT locks its own environment.
    #[test]
    fn the_c_runtime_sees_the_default_and_an_existing_value_wins() {
        let name = "FCS_PROCESS_ENV_TEST";
        // SAFETY: nothing else in the test binary reads or writes this name.
        unsafe { set_default(name, OsStr::new("first")) };
        assert_eq!(
            crt_getenv(c"FCS_PROCESS_ENV_TEST").as_deref(),
            Some("first")
        );
        assert_eq!(std::env::var(name).as_deref(), Ok("first"));

        // SAFETY: as above.
        unsafe { set_default(name, OsStr::new("second")) };
        assert_eq!(
            crt_getenv(c"FCS_PROCESS_ENV_TEST").as_deref(),
            Some("first")
        );
    }
}
