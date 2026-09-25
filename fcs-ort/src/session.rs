//! Safe wrappers over the ONNX Runtime session API.
//!
//! Scope is deliberately narrow: load a model from a path, run it on f32
//! tensors, read f32 tensors back. That is everything Face Crop Studio asks of
//! ONNX Runtime, and anything wider would be surface to maintain for no caller.

use std::{
    ffi::{CStr, CString},
    path::Path,
    ptr,
    sync::Arc,
};

use crate::{
    Runtime,
    sys::{self, OrtApi},
};

/// An ONNX Runtime error, carrying the runtime's own message.
#[derive(Debug)]
pub struct Error {
    code: sys::OrtErrorCode,
    message: String,
}

impl Error {
    /// The runtime's error code.
    pub fn code(&self) -> sys::OrtErrorCode {
        self.code
    }
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "onnxruntime error {:?}: {}", self.code, self.message)
    }
}

impl std::error::Error for Error {}

type Result<T> = std::result::Result<T, Error>;

/// Convert a returned `OrtStatusPtr` into a `Result`, taking ownership of it.
///
/// # Safety
///
/// `status` must be null or a status pointer freshly returned by `api`, not yet
/// released. Ownership transfers here — the caller must not release it again.
unsafe fn check(api: *const OrtApi, status: sys::OrtStatusPtr) -> Result<()> {
    if status.is_null() {
        return Ok(());
    }
    unsafe {
        let code = ((*api).GetErrorCode)(status);
        let message = CStr::from_ptr(((*api).GetErrorMessage)(status))
            .to_string_lossy()
            .into_owned();
        ((*api).ReleaseStatus)(status);
        Err(Error { code, message })
    }
}

/// A loaded runtime plus the process-wide `OrtEnv` that sessions hang off.
///
/// ONNX Runtime requires an environment to outlive every session created from
/// it, so sessions hold an `Arc` of this rather than a bare pointer.
pub struct Environment {
    runtime: Runtime,
    env: *mut sys::OrtEnv,
    memory_info: *mut sys::OrtMemoryInfo,
}

// SAFETY: `OrtEnv` and `OrtMemoryInfo` are documented as safe to share across
// threads; ONNX Runtime's own threadpools use the environment from many
// threads. Neither pointer is mutated after construction, and both stay valid
// until `Drop`, which cannot run while an `Arc` is outstanding.
unsafe impl Send for Environment {}
unsafe impl Sync for Environment {}

impl std::fmt::Debug for Environment {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Environment")
            .field("runtime", &self.runtime)
            .finish()
    }
}

impl Environment {
    /// The process-wide environment, created on first use.
    ///
    /// ONNX Runtime expects one environment per process, and the result is
    /// cached either way: a machine with no runtime should not pay for a failed
    /// library search on every model load.
    pub fn shared() -> Option<Arc<Self>> {
        static SHARED: std::sync::OnceLock<Option<Arc<Environment>>> = std::sync::OnceLock::new();
        SHARED.get_or_init(Self::load).clone()
    }

    /// Locate a usable ONNX Runtime and build an environment from it.
    ///
    /// Returns `None` when no compatible runtime is installed — that is a
    /// routine outcome, not an error, and callers fall back to another backend.
    /// Prefer [`Environment::shared`] outside tests.
    pub fn load() -> Option<Arc<Self>> {
        let runtime = crate::locate()?;
        match Self::from_runtime(runtime) {
            Ok(env) => Some(Arc::new(env)),
            Err(err) => {
                log::warn!("ONNX Runtime found but the environment failed: {err}");
                None
            }
        }
    }

    fn from_runtime(runtime: Runtime) -> Result<Self> {
        let api = runtime.api();
        // Static lifetime: ONNX Runtime copies the log id, but a temporary
        // CString here would still be a dangling read during the call itself.
        let log_id = c"face-crop-studio";

        // SAFETY: `api` was validated by `locate`, which asked the library for
        // exactly this API version and rejected it if unavailable. Both out
        // pointers are initialised on success and checked before use.
        unsafe {
            let mut env = ptr::null_mut();
            check(
                api,
                ((*api).CreateEnv)(sys::OrtLoggingLevel::Warning, log_id.as_ptr(), &mut env),
            )?;

            let mut memory_info = ptr::null_mut();
            let status = ((*api).CreateCpuMemoryInfo)(
                sys::OrtAllocatorType::Device,
                sys::OrtMemType::Default,
                &mut memory_info,
            );
            if let Err(err) = check(api, status) {
                ((*api).ReleaseEnv)(env);
                return Err(err);
            }

            Ok(Self {
                runtime,
                env,
                memory_info,
            })
        }
    }

    /// The underlying runtime, for its path and version.
    pub fn runtime(&self) -> &Runtime {
        &self.runtime
    }

    fn api(&self) -> *const OrtApi {
        self.runtime.api()
    }
}

impl Drop for Environment {
    fn drop(&mut self) {
        // SAFETY: both pointers came from this environment's own constructor and
        // are released exactly once. Sessions hold an `Arc`, so none can be alive.
        unsafe {
            let api = self.api();
            ((*api).ReleaseMemoryInfo)(self.memory_info);
            ((*api).ReleaseEnv)(self.env);
        }
    }
}

/// Options applied when creating a session.
#[derive(Debug, Clone, Copy)]
pub struct SessionOptions {
    /// Threads ONNX Runtime may use *within* one inference.
    ///
    /// The pool belongs to the session, and Face Crop Studio shares one session across all
    /// rayon workers, so this is a total rather than a per-run multiplier. Raising it adds
    /// threads to that one pool; it does not give each concurrent inference its own.
    ///
    /// Re-measured on a 7950X at 640x640, order alternated (experiment 69). One inference at
    /// a time: **7.27 ms at 1 thread against 4.17 ms at 4**, winning every pair. A 1239-image
    /// folder export on the same machine: 10.32 s against 10.18 s, which is no difference --
    /// rayon has already filled the cores there, so the extra threads have nothing to add.
    ///
    /// So this buys preview latency on a machine with no GPU and costs nothing in batch, which
    /// is why [`default_intra_threads`] raises it only where there are spare cores.
    ///
    /// An earlier note here recorded 8.69/1.94/3.74 ms for 1/4/16 threads and concluded that
    /// the runtime "oversubscribes itself past about 4". The 16-thread regression did not
    /// reproduce (5.64 ms against 5.10 at 8), and neither did the size of the gain. Both
    /// measurements agree on the direction.
    pub intra_threads: i32,
    /// Graph optimisation level. `All` matches what the `ort` crate defaults to.
    pub optimization: sys::GraphOptimizationLevel,
}

/// Intra-op threads to use by default: enough to help a lone inference, never enough to
/// crowd a small machine.
///
/// Divided by four rather than taken straight, because the gain was measured on a 16-core
/// box with cores to spare and the batch path already runs one image per rayon worker. A
/// machine with four logical processors keeps today's single thread, which is the
/// configuration this change was *not* able to test.
fn default_intra_threads() -> i32 {
    intra_threads_for(std::thread::available_parallelism().map_or(1, std::num::NonZeroUsize::get))
}

fn intra_threads_for(logical: usize) -> i32 {
    (logical / 4).clamp(1, 4) as i32
}

impl Default for SessionOptions {
    fn default() -> Self {
        Self {
            intra_threads: default_intra_threads(),
            optimization: sys::GraphOptimizationLevel::All,
        }
    }
}

/// A loaded model, ready to run.
pub struct Session {
    environment: Arc<Environment>,
    session: *mut sys::OrtSession,
    input_names: Vec<CString>,
    output_names: Vec<CString>,
}

// SAFETY: ONNX Runtime documents `OrtSession` as safe for concurrent `Run`
// calls from multiple threads — that is the whole reason it owns threadpools
// rather than requiring external locking. Nothing in this wrapper mutates
// `Session` after construction, so `&self` is honest. `concurrent_runs_match_
// sequential` in fcs-core exercises this rather than taking the docs on trust.
unsafe impl Send for Session {}
unsafe impl Sync for Session {}

impl std::fmt::Debug for Session {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Session")
            .field("inputs", &self.input_names)
            .field("outputs", &self.output_names.len())
            .finish()
    }
}

/// One output tensor, copied out of the runtime's memory.
#[derive(Debug, Clone)]
pub struct OutputTensor {
    /// Dimensions, outermost first.
    pub shape: Vec<usize>,
    /// Elements in row-major order.
    pub data: Vec<f32>,
}

/// Encode a path the way `ORTCHAR_T` expects: UTF-16 on Windows, bytes
/// elsewhere. Getting this wrong silently fails to open perfectly good files.
#[cfg(target_os = "windows")]
fn encode_path(path: &Path) -> Vec<sys::OsChar> {
    use std::os::windows::ffi::OsStrExt;
    path.as_os_str().encode_wide().chain(Some(0)).collect()
}

#[cfg(not(target_os = "windows"))]
fn encode_path(path: &Path) -> Vec<sys::OsChar> {
    use std::os::unix::ffi::OsStrExt;
    path.as_os_str()
        .as_bytes()
        .iter()
        .map(|b| *b as sys::OsChar)
        .chain(Some(0))
        .collect()
}

impl Session {
    /// Load a model from disk.
    pub fn new(
        environment: &Arc<Environment>,
        model_path: &Path,
        options: SessionOptions,
    ) -> Result<Self> {
        let api = environment.api();
        let path = encode_path(model_path);

        // SAFETY: every handle below is created here, checked for error before
        // use, and released on every exit path. `path` outlives the
        // `CreateSession` call that borrows it.
        unsafe {
            let mut opts = ptr::null_mut();
            check(api, ((*api).CreateSessionOptions)(&mut opts))?;
            // From here on, `opts` must be released before returning.
            let configured = (|| {
                check(
                    api,
                    ((*api).SetIntraOpNumThreads)(opts, options.intra_threads),
                )?;
                check(
                    api,
                    ((*api).SetSessionGraphOptimizationLevel)(opts, options.optimization),
                )
            })();
            if let Err(err) = configured {
                ((*api).ReleaseSessionOptions)(opts);
                return Err(err);
            }

            let mut session = ptr::null_mut();
            let status = ((*api).CreateSession)(environment.env, path.as_ptr(), opts, &mut session);
            ((*api).ReleaseSessionOptions)(opts);
            check(api, status)?;

            let names = (|| {
                let inputs = Self::names(api, session, true)?;
                let outputs = Self::names(api, session, false)?;
                Ok((inputs, outputs))
            })();
            let (input_names, output_names) = match names {
                Ok(pair) => pair,
                Err(err) => {
                    ((*api).ReleaseSession)(session);
                    return Err(err);
                }
            };

            Ok(Self {
                environment: Arc::clone(environment),
                session,
                input_names,
                output_names,
            })
        }
    }

    /// Read the graph's input or output names.
    ///
    /// # Safety
    ///
    /// `session` must be a live session belonging to `api`.
    unsafe fn names(
        api: *const OrtApi,
        session: *mut sys::OrtSession,
        inputs: bool,
    ) -> Result<Vec<CString>> {
        unsafe {
            let mut allocator = ptr::null_mut();
            check(api, ((*api).GetAllocatorWithDefaultOptions)(&mut allocator))?;

            let mut count = 0usize;
            let status = if inputs {
                ((*api).SessionGetInputCount)(session, &mut count)
            } else {
                ((*api).SessionGetOutputCount)(session, &mut count)
            };
            check(api, status)?;

            let mut names = Vec::with_capacity(count);
            for index in 0..count {
                let mut raw = ptr::null_mut();
                let status = if inputs {
                    ((*api).SessionGetInputName)(session, index, allocator, &mut raw)
                } else {
                    ((*api).SessionGetOutputName)(session, index, allocator, &mut raw)
                };
                check(api, status)?;
                // The runtime allocated this string; copy it and hand the
                // allocation straight back rather than leaking one per name.
                names.push(CStr::from_ptr(raw).to_owned());
                let _ = ((*api).AllocatorFree)(allocator, raw.cast());
            }
            Ok(names)
        }
    }

    /// Graph input names, in order.
    pub fn input_names(&self) -> &[CString] {
        &self.input_names
    }

    /// Graph output names, in order.
    pub fn output_names(&self) -> &[CString] {
        &self.output_names
    }

    /// Run the model on a single f32 input, returning every output.
    ///
    /// Takes `&self`: ONNX Runtime permits concurrent runs on one session, so
    /// callers need no lock and no session pool.
    pub fn run(&self, input: &[f32], shape: &[usize]) -> Result<Vec<OutputTensor>> {
        if self.input_names.len() != 1 {
            return Err(Error {
                code: sys::OrtErrorCode::InvalidArgument,
                message: format!(
                    "run() takes a single input, but the graph declares {}",
                    self.input_names.len()
                ),
            });
        }
        let expected: usize = shape.iter().product();
        if expected != input.len() {
            return Err(Error {
                code: sys::OrtErrorCode::InvalidArgument,
                message: format!(
                    "shape {shape:?} needs {expected} elements, got {}",
                    input.len()
                ),
            });
        }

        let api = self.environment.api();
        let dims: Vec<i64> = shape.iter().map(|d| *d as i64).collect();
        let input_ptrs: Vec<*const std::ffi::c_char> =
            self.input_names.iter().map(|n| n.as_ptr()).collect();
        let output_ptrs: Vec<*const std::ffi::c_char> =
            self.output_names.iter().map(|n| n.as_ptr()).collect();
        let mut outputs: Vec<*mut sys::OrtValue> = vec![ptr::null_mut(); self.output_names.len()];

        // SAFETY: `CreateTensorWithDataAsOrtValue` borrows `input` rather than
        // copying it, so `input` must outlive both the value and the `Run` call
        // — it does, being a parameter. The cast to `*mut` is required by the
        // signature; the runtime does not write through it for an input.
        unsafe {
            let mut value = ptr::null_mut();
            check(
                api,
                ((*api).CreateTensorWithDataAsOrtValue)(
                    self.environment.memory_info,
                    input.as_ptr().cast_mut().cast(),
                    std::mem::size_of_val(input),
                    dims.as_ptr(),
                    dims.len(),
                    sys::TensorElementDataType::Float,
                    &mut value,
                ),
            )?;

            let value_ptrs: [*const sys::OrtValue; 1] = [value];
            let status = ((*api).Run)(
                self.session,
                ptr::null(),
                input_ptrs.as_ptr(),
                value_ptrs.as_ptr(),
                input_ptrs.len(),
                output_ptrs.as_ptr(),
                output_ptrs.len(),
                outputs.as_mut_ptr(),
            );
            ((*api).ReleaseValue)(value);
            check(api, status)?;

            let collected = outputs
                .iter()
                .filter(|out| !out.is_null())
                .map(|&out| Self::read_tensor(api, out))
                .collect::<Result<Vec<_>>>();
            // Every output is released whatever happened, so an error partway through leaks
            // nothing. Release* accepts null.
            for out in outputs {
                ((*api).ReleaseValue)(out);
            }
            collected
        }
    }

    /// Copy one output tensor out of runtime-owned memory.
    ///
    /// # Safety
    ///
    /// `value` must be a live f32 tensor produced by `api`.
    unsafe fn read_tensor(api: *const OrtApi, value: *mut sys::OrtValue) -> Result<OutputTensor> {
        unsafe {
            let mut info = ptr::null_mut();
            check(api, ((*api).GetTensorTypeAndShape)(value, &mut info))?;

            let shape = (|| {
                let mut rank = 0usize;
                check(api, ((*api).GetDimensionsCount)(info, &mut rank))?;
                let mut dims = vec![0i64; rank];
                check(api, ((*api).GetDimensions)(info, dims.as_mut_ptr(), rank))?;
                Ok(dims)
            })();
            ((*api).ReleaseTensorTypeAndShapeInfo)(info);
            let dims = shape?;

            // A negative dimension would mean a symbolic/unresolved shape, which
            // cannot happen for a completed run — but casting it to usize would
            // produce an enormous length and a wild read, so reject it.
            let mut shape = Vec::with_capacity(dims.len());
            for dim in dims {
                if dim < 0 {
                    return Err(Error {
                        code: sys::OrtErrorCode::Fail,
                        message: format!("output tensor has unresolved dimension {dim}"),
                    });
                }
                shape.push(dim as usize);
            }

            let len: usize = shape.iter().product();
            let mut data = ptr::null_mut();
            check(api, ((*api).GetTensorMutableData)(value, &mut data))?;
            if data.is_null() {
                // A zero-element tensor legitimately has no buffer.
                return Ok(OutputTensor {
                    shape,
                    data: Vec::new(),
                });
            }

            Ok(OutputTensor {
                shape,
                data: std::slice::from_raw_parts(data.cast::<f32>(), len).to_vec(),
            })
        }
    }
}

impl Drop for Session {
    fn drop(&mut self) {
        // SAFETY: `session` came from `CreateSession` here and is released once.
        // The environment outlives this via the `Arc`.
        unsafe {
            ((*self.environment.api()).ReleaseSession)(self.session);
        }
    }
}

#[cfg(test)]
mod option_tests {
    use super::*;

    #[test]
    fn default_intra_threads_stays_within_range() {
        let n = default_intra_threads();
        assert!(
            (1..=4).contains(&n),
            "intra_threads {n} outside the intended 1..=4"
        );
    }

    #[test]
    fn intra_threads_are_a_quarter_of_the_logical_cores_within_one_to_four() {
        // 13 separates `/` (3) from `*` (4, clamped) and `%` (1).
        assert_eq!([3, 13, 64].map(intra_threads_for), [1, 3, 4]);
        // The default is that rule applied to this machine, not a fixed count.
        let logical = std::thread::available_parallelism().map_or(1, std::num::NonZeroUsize::get);
        assert_eq!(default_intra_threads(), intra_threads_for(logical));
    }

    #[test]
    fn default_session_options_use_full_optimization() {
        let options = SessionOptions::default();
        assert_eq!(options.intra_threads, default_intra_threads());
        assert!(matches!(
            options.optimization,
            sys::GraphOptimizationLevel::All
        ));
    }
}

/// `Drop` is the only place either type frees its native handles, and no functional test can see
/// a leak: replacing either `drop` with `()` survived. So this runs the real runtime through a
/// copy of its API table in which the three release entries count their calls and then forward to
/// the real ones. Everything else in the table is the genuine function.
#[cfg(test)]
mod drop_tests {
    use super::*;
    use std::sync::{
        OnceLock,
        atomic::{AtomicUsize, Ordering::SeqCst},
    };

    static ENV_RELEASES: AtomicUsize = AtomicUsize::new(0);
    static MEMORY_INFO_RELEASES: AtomicUsize = AtomicUsize::new(0);
    static SESSION_RELEASES: AtomicUsize = AtomicUsize::new(0);
    /// The unpatched table the shims forward to, as an address: raw pointers are not `Sync`.
    static REAL_API: OnceLock<usize> = OnceLock::new();

    fn real() -> *const OrtApi {
        *REAL_API.get().expect("set before any shim can run") as *const OrtApi
    }

    unsafe extern "system" fn release_env(env: *mut sys::OrtEnv) {
        ENV_RELEASES.fetch_add(1, SeqCst);
        unsafe { ((*real()).ReleaseEnv)(env) }
    }

    unsafe extern "system" fn release_memory_info(info: *mut sys::OrtMemoryInfo) {
        MEMORY_INFO_RELEASES.fetch_add(1, SeqCst);
        unsafe { ((*real()).ReleaseMemoryInfo)(info) }
    }

    unsafe extern "system" fn release_session(session: *mut sys::OrtSession) {
        SESSION_RELEASES.fetch_add(1, SeqCst);
        unsafe { ((*real()).ReleaseSession)(session) }
    }

    #[test]
    fn dropping_a_session_and_its_environment_releases_each_handle_once() {
        let strict = std::env::var("FCS_STRICT_TESTS").is_ok_and(|v| v != "0" && !v.is_empty());
        let model = Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .expect("fcs-ort sits in the workspace root")
            .join("models/eye_refiner.onnx");
        let Some(mut runtime) = crate::locate() else {
            assert!(!strict, "FCS_STRICT_TESTS: no ONNX Runtime");
            eprintln!("skipped: no ONNX Runtime");
            return;
        };
        if !model.exists() {
            assert!(!strict, "FCS_STRICT_TESTS: no model at {model:?}");
            eprintln!("skipped: no model at {model:?}");
            return;
        }

        // Dropping the environment drops its `Runtime`, and with it a library handle. A second
        // handle, never closed, keeps the module mapped for the rest of the process: unloading
        // ONNX Runtime while its global thread pools exist is not something to test by accident.
        // SAFETY: the same library `locate` already loaded and validated.
        std::mem::forget(unsafe { libloading::Library::new(runtime.path()) }.expect("reload"));

        // SAFETY: `OrtApi` is a `repr(C)` table of plain pointers, so a bitwise copy is an
        // equally valid table; leaking it gives it the process lifetime the runtime assumes.
        let mut table = unsafe { ptr::read(runtime.api) };
        REAL_API.get_or_init(|| runtime.api as usize);
        table.ReleaseEnv = release_env;
        table.ReleaseMemoryInfo = release_memory_info;
        table.ReleaseSession = release_session;
        runtime.api = Box::leak(Box::new(table));

        let environment = Arc::new(Environment::from_runtime(runtime).expect("environment"));
        let session =
            Session::new(&environment, &model, SessionOptions::default()).expect("session");

        drop(session);
        assert_eq!(
            SESSION_RELEASES.load(SeqCst),
            1,
            "a dropped session must release its handle"
        );
        assert_eq!(
            ENV_RELEASES.load(SeqCst),
            0,
            "the environment is still held"
        );

        drop(environment);
        assert_eq!(
            ENV_RELEASES.load(SeqCst),
            1,
            "a dropped environment must release the env"
        );
        assert_eq!(MEMORY_INFO_RELEASES.load(SeqCst), 1, "and its memory info");
    }
}
