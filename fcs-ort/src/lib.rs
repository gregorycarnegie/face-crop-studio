//! ONNX Runtime discovery and loading for Face Crop Studio.
//!
//! This crate exists to take over from the `ort` crate one piece at a time. It
//! currently owns exactly one job — finding a usable ONNX Runtime and proving
//! it is usable — while `fcs-core` still uses `ort` for sessions and tensors.
//! See [`sys`] for how to grow the C API surface safely.
//!
//! # Why the validation is this careful
//!
//! `ort` has no fallible initialisation: it calls `.expect()` inside a `#[cold]`
//! non-unwinding function, the failure poisons a global mutex, and the process
//! **aborts** somewhere `catch_unwind` cannot reach. So anything `ort` would
//! reject has to be rejected here first, before any `ort` API is touched.
//!
//! That is not theoretical. Several unrelated desktop applications install an
//! `onnxruntime.dll` on PATH; a machine carrying a 1.17 copy resolves the bare
//! library name to it, and a probe that checks only that the file loads and
//! exports `OrtGetApiBase` passes it straight through to an abort at startup.
//! [`locate`] therefore reproduces `ort`'s own path resolution *and* its
//! version rule, and adds the ABI check `ort` does not do: asking the library
//! for the exact API version wanted.

pub mod session;
pub mod sys;

pub use session::{Environment, Error, OutputTensor, Session, SessionOptions};

use std::{
    ffi::CStr,
    path::{Path, PathBuf},
};

/// API version this build requires. For ONNX Runtime this equals the minor
/// version: `ORT_API_VERSION` of `N` means `1.N.x`.
///
/// While `fcs-core` still depends on `ort`, this MUST match the `api-NN`
/// feature selected for `ort` in the workspace manifest. If this is lower,
/// `fcs-core` accepts a library that `ort` then aborts on.
pub const REQUIRED_API_VERSION: u32 = 24;

/// Number of `OrtApi` fields declared in [`sys`].
const DECLARED_API_FIELDS: usize = 101;

// Every `OrtApi` entry is a function pointer, so the declared struct must be
// exactly one pointer per field. This catches a field declared with a
// non-pointer type or accidentally duplicated — which would shift every later
// offset and call the wrong function, with no other compile-time symptom.
const _: () = assert!(
    size_of::<sys::OrtApi>() == DECLARED_API_FIELDS * size_of::<*const ()>(),
    "OrtApi prefix is not one pointer per declared field"
);

/// A validated ONNX Runtime, kept loaded.
///
/// Holding the library open means the copy that was validated is the copy that
/// stays mapped, rather than being unloaded and re-resolved later against a
/// possibly different file.
pub struct Runtime {
    /// Unread, but load-bearing: dropping it unloads the module and invalidates
    /// `api`. Held for the lifetime of the `Runtime` for exactly that reason.
    _library: libloading::Library,
    api: *const sys::OrtApi,
    path: PathBuf,
    version: String,
}

// SAFETY: `api` points into the loaded library's static data. It is immutable,
// outlives every use because `library` keeps the module mapped, and nothing in
// `Runtime` is mutated after construction.
unsafe impl Send for Runtime {}
unsafe impl Sync for Runtime {}

impl std::fmt::Debug for Runtime {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Runtime")
            .field("path", &self.path)
            .field("version", &self.version)
            .finish()
    }
}

impl Runtime {
    /// Where this runtime was loaded from.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Version as reported by the runtime itself, e.g. `"1.24.4"`.
    pub fn version(&self) -> &str {
        &self.version
    }

    /// The validated API table.
    ///
    /// Not public: the layout in [`sys`] is a prefix, so calling through it
    /// correctly depends on invariants only this crate maintains.
    pub(crate) fn api(&self) -> *const sys::OrtApi {
        self.api
    }
}

/// Why a candidate was not usable. Rejection is routine — callers fall back to
/// another backend — so this exists for logging rather than as an error type.
#[derive(Debug)]
enum Rejected {
    NotLoadable(libloading::Error),
    NoEntryPoint,
    NullApiBase,
    /// Older than [`REQUIRED_API_VERSION`]. `ort` aborts on these.
    TooOld(String),
    /// Loads and reports a new enough version, but will not serve the API
    /// version asked for. A corrupt or mismatched build looks like this.
    ApiUnavailable(String),
}

impl std::fmt::Display for Rejected {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NotLoadable(e) => write!(f, "not loadable: {e}"),
            Self::NoEntryPoint => f.write_str("does not export OrtGetApiBase"),
            Self::NullApiBase => f.write_str("OrtGetApiBase returned null"),
            Self::TooOld(v) => write!(f, "version {v} is older than 1.{REQUIRED_API_VERSION}"),
            Self::ApiUnavailable(v) => write!(
                f,
                "version {v} does not provide API version {REQUIRED_API_VERSION}"
            ),
        }
    }
}

/// Find and validate an ONNX Runtime, or `None` when there is no usable one.
///
/// Candidates are tried in the order `ort` resolves them, so the library
/// validated here is the one `ort` goes on to load.
pub fn locate() -> Option<Runtime> {
    for candidate in candidates() {
        match validate(&candidate) {
            Ok(runtime) => {
                log::debug!(
                    "ONNX Runtime {} at {}",
                    runtime.version,
                    runtime.path.display()
                );
                return Some(runtime);
            }
            Err(reason) => log::debug!("ignoring {}: {reason}", candidate.display()),
        }
    }
    None
}

/// Platform file name for the runtime.
pub fn library_name() -> &'static str {
    if cfg!(target_os = "windows") {
        "onnxruntime.dll"
    } else if cfg!(target_os = "macos") {
        "libonnxruntime.dylib"
    } else {
        "libonnxruntime.so"
    }
}

fn candidates() -> Vec<PathBuf> {
    let mut out = Vec::new();
    // `ort` reads ORT_DYLIB_PATH first and loads whatever it names, so it has
    // to be validated first or a different file gets checked than used.
    if let Ok(path) = std::env::var("ORT_DYLIB_PATH")
        && !path.is_empty()
    {
        out.push(PathBuf::from(path));
    }
    out.push(PathBuf::from(library_name()));
    out
}

/// Resolve like `ort` does: a bare name is looked for beside the executable
/// first, then left to the platform loader's search path.
fn resolve(path: &Path) -> PathBuf {
    if path.is_absolute() {
        return path.to_path_buf();
    }
    match std::env::current_exe()
        .ok()
        .and_then(|exe| exe.parent().map(|dir| dir.join(path)))
    {
        Some(beside_exe) if beside_exe.exists() => beside_exe,
        _ => path.to_path_buf(),
    }
}

fn validate(candidate: &Path) -> Result<Runtime, Rejected> {
    let path = resolve(candidate);

    // SAFETY: loading an arbitrary library runs its initialisers, which is
    // inherent to dynamic loading and is what `ort` does too. Everything after
    // the load only reads through pointers the library itself returned, and
    // the library stays mapped for as long as the returned `Runtime` lives.
    unsafe {
        let library = libloading::Library::new(&path).map_err(Rejected::NotLoadable)?;

        let entry: libloading::Symbol<unsafe extern "system" fn() -> *const sys::OrtApiBase> =
            library
                .get(b"OrtGetApiBase")
                .map_err(|_| Rejected::NoEntryPoint)?;
        let base = entry();
        if base.is_null() {
            return Err(Rejected::NullApiBase);
        }

        let version = CStr::from_ptr(((*base).get_version_string)())
            .to_string_lossy()
            .into_owned();

        // Mirror ort's rule exactly: it compares only the minor component and
        // rejects anything lower, and its rejection aborts the process.
        let minor = version
            .split('.')
            .nth(1)
            .and_then(|m| m.parse::<u32>().ok())
            .unwrap_or(0);
        if minor < REQUIRED_API_VERSION {
            return Err(Rejected::TooOld(version));
        }

        // The authoritative check the version string only approximates: ask for
        // the exact API version. A build that cannot serve it returns null
        // rather than a table with shifted offsets.
        let api = ((*base).get_api)(REQUIRED_API_VERSION);
        if api.is_null() {
            return Err(Rejected::ApiUnavailable(version));
        }

        Ok(Runtime {
            _library: library,
            api,
            path,
            version,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn library_name_matches_the_platform() {
        let name = library_name();
        assert!(name.contains("onnxruntime"));
        if cfg!(target_os = "windows") {
            assert!(name.ends_with(".dll"));
        } else if cfg!(target_os = "macos") {
            assert!(name.ends_with(".dylib"));
        } else {
            assert!(name.ends_with(".so"));
        }
    }

    /// A path that cannot be loaded must be reported, not panic. The entire
    /// point of this crate is that a bad library never reaches `ort`.
    #[test]
    fn a_missing_library_is_rejected_without_panicking() {
        let err = validate(Path::new("definitely-not-a-real-onnxruntime.dll"))
            .expect_err("a nonexistent library must not validate");
        assert!(matches!(err, Rejected::NotLoadable(_)), "got {err:?}");
    }

    /// A real library that is not ONNX Runtime loads fine but exports no entry
    /// point. This is the case separating "loadable" from "usable", and getting
    /// it wrong is what let an incompatible DLL through before.
    #[test]
    fn a_library_without_the_entry_point_is_rejected() {
        let system_lib = if cfg!(target_os = "windows") {
            "kernel32.dll"
        } else if cfg!(target_os = "macos") {
            "libSystem.B.dylib"
        } else {
            "libc.so.6"
        };
        match validate(Path::new(system_lib)) {
            Err(Rejected::NoEntryPoint) => {}
            // Acceptable: the platform may not have it under that name here.
            Err(Rejected::NotLoadable(_)) => {}
            other => panic!("expected rejection, got {other:?}"),
        }
    }

    #[test]
    fn an_absolute_path_resolves_to_itself() {
        let absolute = if cfg!(target_os = "windows") {
            PathBuf::from("C:\\some\\where\\onnxruntime.dll")
        } else {
            PathBuf::from("/some/where/libonnxruntime.so")
        };
        assert_eq!(resolve(&absolute), absolute);
    }
}
