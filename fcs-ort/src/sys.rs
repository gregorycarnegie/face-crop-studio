//! Raw declarations from `onnxruntime_c_api.h`.
//!
//! # Why a prefix is enough
//!
//! `OrtApi` is a struct of 424 function pointers, and the ONNX Runtime C API
//! guarantees it is **append-only**: new releases add entries at the end and
//! never reorder or remove existing ones. That guarantee is what lets a single
//! binary serve every `ORT_API_VERSION`, and it means a declaration covering
//! only the leading fields is ABI-correct. Everything Face Crop Studio needs
//! lives in the first 101, so 323 fields are omitted entirely.
//!
//! # Rules for adding to `OrtApi`
//!
//! 1. Declare **every** field from the start of the struct up to the one you
//!    want, in header order, including ones you never call. A skipped or
//!    reordered field shifts every later offset and you will call the wrong
//!    function pointer — undefined behaviour, not a compile error. Use
//!    [`Unused`] for the ones you do not need.
//! 2. Beware `#[cfg]`-duplicated fields when cross-checking against generated
//!    bindings. `ort-sys` declares `CreateSession` twice (a `wasm32` variant and
//!    a native one); counting both shifts every subsequent index. Three fields
//!    in this prefix are affected.
//! 3. Take signatures from the **oldest** ONNX Runtime this crate supports, not
//!    the newest.
//!
//! `extern "system"` (not `extern "C"`) matches the header's `ORT_API_CALL`.
//! The two are identical on x86_64 but differ on 32-bit Windows, where `system`
//! is `stdcall`.
#![allow(non_snake_case)]

#[cfg(target_os = "windows")]
use std::ffi::c_ushort;
use std::ffi::{c_char, c_int, c_void};

/// `ORTCHAR_T`: UTF-16 on Windows, bytes elsewhere. Model paths use it, which
/// is why [`crate::session`] encodes paths per platform rather than as UTF-8.
#[cfg(target_os = "windows")]
pub type OsChar = c_ushort;
/// `ORTCHAR_T`: UTF-16 on Windows, bytes elsewhere.
#[cfg(not(target_os = "windows"))]
pub type OsChar = c_char;

/// A field declared for layout only and never called.
///
/// Typed as a data pointer rather than a function pointer so that calling it is
/// not expressible; its only job is to occupy one pointer of space.
pub type Unused = *const c_void;

macro_rules! opaque {
    ($($name:ident),* $(,)?) => {$(
        /// Opaque handle owned by ONNX Runtime.
        #[repr(C)]
        pub struct $name {
            _opaque: [u8; 0],
            /// Prevents auto `Send`/`Sync` and makes the type `!Unpin`-ish, so
            /// handles cannot be moved across threads without a deliberate
            /// `unsafe impl` on the wrapper that owns them.
            _marker: core::marker::PhantomData<(*mut u8, core::marker::PhantomPinned)>,
        }
    )*};
}

opaque!(
    OrtStatus,
    OrtEnv,
    OrtSession,
    OrtSessionOptions,
    OrtRunOptions,
    OrtValue,
    OrtMemoryInfo,
    OrtAllocator,
    OrtTensorTypeAndShapeInfo,
);

/// Null means success. Any other value is an owned status that must be released.
pub type OrtStatusPtr = *mut OrtStatus;

#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OrtLoggingLevel {
    Verbose = 0,
    Info = 1,
    Warning = 2,
    Error = 3,
    Fatal = 4,
}

#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OrtErrorCode {
    Ok = 0,
    Fail = 1,
    InvalidArgument = 2,
    NoSuchFile = 3,
    NoModel = 4,
    EngineError = 5,
    RuntimeException = 6,
    InvalidProtobuf = 7,
    ModelLoaded = 8,
    NotImplemented = 9,
    InvalidGraph = 10,
    EpFail = 11,
}

/// Only the one variant this crate creates or reads is named; the tag is what
/// matters at the ABI boundary.
#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TensorElementDataType {
    Undefined = 0,
    Float = 1,
}

#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OrtAllocatorType {
    Invalid = -1,
    Device = 0,
    Arena = 1,
}

#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OrtMemType {
    // CpuInput (-2) and CpuOutput (-1) exist upstream; only what the crate uses is named.
    Default = 0,
}

#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GraphOptimizationLevel {
    DisableAll = 0,
    Basic = 1,
    Extended = 2,
    Layout = 3,
    All = 99,
}

/// The versioned API table. See the module docs before adding a field.
#[repr(C)]
pub struct OrtApi {
    /// `OrtApi` field 0.
    pub CreateStatus:
        unsafe extern "system" fn(code: OrtErrorCode, msg: *const c_char) -> OrtStatusPtr,
    /// `OrtApi` field 1.
    pub GetErrorCode: unsafe extern "system" fn(status: *const OrtStatus) -> OrtErrorCode,
    /// `OrtApi` field 2.
    pub GetErrorMessage: unsafe extern "system" fn(status: *const OrtStatus) -> *const c_char,
    /// `OrtApi` field 3.
    pub CreateEnv: unsafe extern "system" fn(
        log_severity_level: OrtLoggingLevel,
        logid: *const c_char,
        out: *mut *mut OrtEnv,
    ) -> OrtStatusPtr,
    _reserved_4: Unused,
    _reserved_5: Unused,
    _reserved_6: Unused,
    /// `OrtApi` field 7.
    pub CreateSession: unsafe extern "system" fn(
        env: *const OrtEnv,
        model_path: *const OsChar,
        options: *const OrtSessionOptions,
        out: *mut *mut OrtSession,
    ) -> OrtStatusPtr,
    _reserved_8: Unused,
    /// `OrtApi` field 9.
    pub Run: unsafe extern "system" fn(
        session: *mut OrtSession,
        run_options: *const OrtRunOptions,
        input_names: *const *const c_char,
        inputs: *const *const OrtValue,
        input_len: usize,
        output_names: *const *const c_char,
        output_names_len: usize,
        outputs: *mut *mut OrtValue,
    ) -> OrtStatusPtr,
    /// `OrtApi` field 10.
    pub CreateSessionOptions:
        unsafe extern "system" fn(options: *mut *mut OrtSessionOptions) -> OrtStatusPtr,
    _reserved_11: Unused,
    _reserved_12: Unused,
    _reserved_13: Unused,
    _reserved_14: Unused,
    _reserved_15: Unused,
    _reserved_16: Unused,
    _reserved_17: Unused,
    _reserved_18: Unused,
    _reserved_19: Unused,
    _reserved_20: Unused,
    _reserved_21: Unused,
    _reserved_22: Unused,
    /// `OrtApi` field 23.
    pub SetSessionGraphOptimizationLevel: unsafe extern "system" fn(
        options: *mut OrtSessionOptions,
        graph_optimization_level: GraphOptimizationLevel,
    ) -> OrtStatusPtr,
    /// `OrtApi` field 24.
    pub SetIntraOpNumThreads: unsafe extern "system" fn(
        options: *mut OrtSessionOptions,
        intra_op_num_threads: c_int,
    ) -> OrtStatusPtr,
    /// `OrtApi` field 25.
    pub SetInterOpNumThreads: unsafe extern "system" fn(
        options: *mut OrtSessionOptions,
        inter_op_num_threads: c_int,
    ) -> OrtStatusPtr,
    _reserved_26: Unused,
    _reserved_27: Unused,
    _reserved_28: Unused,
    _reserved_29: Unused,
    /// `OrtApi` field 30.
    pub SessionGetInputCount:
        unsafe extern "system" fn(session: *const OrtSession, out: *mut usize) -> OrtStatusPtr,
    /// `OrtApi` field 31.
    pub SessionGetOutputCount:
        unsafe extern "system" fn(session: *const OrtSession, out: *mut usize) -> OrtStatusPtr,
    _reserved_32: Unused,
    _reserved_33: Unused,
    _reserved_34: Unused,
    _reserved_35: Unused,
    /// `OrtApi` field 36.
    pub SessionGetInputName: unsafe extern "system" fn(
        session: *const OrtSession,
        index: usize,
        allocator: *mut OrtAllocator,
        value: *mut *mut c_char,
    ) -> OrtStatusPtr,
    /// `OrtApi` field 37.
    pub SessionGetOutputName: unsafe extern "system" fn(
        session: *const OrtSession,
        index: usize,
        allocator: *mut OrtAllocator,
        value: *mut *mut c_char,
    ) -> OrtStatusPtr,
    _reserved_38: Unused,
    _reserved_39: Unused,
    _reserved_40: Unused,
    _reserved_41: Unused,
    _reserved_42: Unused,
    _reserved_43: Unused,
    _reserved_44: Unused,
    _reserved_45: Unused,
    _reserved_46: Unused,
    _reserved_47: Unused,
    _reserved_48: Unused,
    /// `OrtApi` field 49.
    pub CreateTensorWithDataAsOrtValue: unsafe extern "system" fn(
        info: *const OrtMemoryInfo,
        p_data: *mut c_void,
        p_data_len: usize,
        shape: *const i64,
        shape_len: usize,
        type_: TensorElementDataType,
        out: *mut *mut OrtValue,
    ) -> OrtStatusPtr,
    _reserved_50: Unused,
    /// `OrtApi` field 51.
    pub GetTensorMutableData:
        unsafe extern "system" fn(value: *mut OrtValue, out: *mut *mut c_void) -> OrtStatusPtr,
    _reserved_52: Unused,
    _reserved_53: Unused,
    _reserved_54: Unused,
    _reserved_55: Unused,
    _reserved_56: Unused,
    _reserved_57: Unused,
    _reserved_58: Unused,
    _reserved_59: Unused,
    _reserved_60: Unused,
    /// `OrtApi` field 61.
    pub GetDimensionsCount: unsafe extern "system" fn(
        info: *const OrtTensorTypeAndShapeInfo,
        out: *mut usize,
    ) -> OrtStatusPtr,
    /// `OrtApi` field 62.
    pub GetDimensions: unsafe extern "system" fn(
        info: *const OrtTensorTypeAndShapeInfo,
        dim_values: *mut i64,
        dim_values_length: usize,
    ) -> OrtStatusPtr,
    _reserved_63: Unused,
    _reserved_64: Unused,
    /// `OrtApi` field 65.
    pub GetTensorTypeAndShape: unsafe extern "system" fn(
        value: *const OrtValue,
        out: *mut *mut OrtTensorTypeAndShapeInfo,
    ) -> OrtStatusPtr,
    _reserved_66: Unused,
    _reserved_67: Unused,
    _reserved_68: Unused,
    /// `OrtApi` field 69.
    pub CreateCpuMemoryInfo: unsafe extern "system" fn(
        type_: OrtAllocatorType,
        mem_type: OrtMemType,
        out: *mut *mut OrtMemoryInfo,
    ) -> OrtStatusPtr,
    _reserved_70: Unused,
    _reserved_71: Unused,
    _reserved_72: Unused,
    _reserved_73: Unused,
    _reserved_74: Unused,
    _reserved_75: Unused,
    /// `OrtApi` field 76.
    pub AllocatorFree:
        unsafe extern "system" fn(ort_allocator: *mut OrtAllocator, p: *mut c_void) -> OrtStatusPtr,
    _reserved_77: Unused,
    /// `OrtApi` field 78.
    pub GetAllocatorWithDefaultOptions:
        unsafe extern "system" fn(out: *mut *mut OrtAllocator) -> OrtStatusPtr,
    _reserved_79: Unused,
    _reserved_80: Unused,
    _reserved_81: Unused,
    _reserved_82: Unused,
    _reserved_83: Unused,
    _reserved_84: Unused,
    _reserved_85: Unused,
    _reserved_86: Unused,
    _reserved_87: Unused,
    _reserved_88: Unused,
    _reserved_89: Unused,
    _reserved_90: Unused,
    _reserved_91: Unused,
    /// `OrtApi` field 92.
    pub ReleaseEnv: unsafe extern "system" fn(input: *mut OrtEnv),
    /// `OrtApi` field 93.
    pub ReleaseStatus: unsafe extern "system" fn(input: *mut OrtStatus),
    /// `OrtApi` field 94.
    pub ReleaseMemoryInfo: unsafe extern "system" fn(input: *mut OrtMemoryInfo),
    /// `OrtApi` field 95.
    pub ReleaseSession: unsafe extern "system" fn(input: *mut OrtSession),
    /// `OrtApi` field 96.
    pub ReleaseValue: unsafe extern "system" fn(input: *mut OrtValue),
    _reserved_97: Unused,
    _reserved_98: Unused,
    /// `OrtApi` field 99.
    pub ReleaseTensorTypeAndShapeInfo:
        unsafe extern "system" fn(input: *mut OrtTensorTypeAndShapeInfo),
    /// `OrtApi` field 100.
    pub ReleaseSessionOptions: unsafe extern "system" fn(input: *mut OrtSessionOptions),
}

/// The entry point every ONNX Runtime build exports, and the only symbol this
/// crate resolves by name.
#[repr(C)]
pub struct OrtApiBase {
    /// Returns the API table for `version`, or null if the runtime is older
    /// than the version requested.
    pub get_api: unsafe extern "system" fn(version: u32) -> *const OrtApi,
    /// Null-terminated library version, e.g. `"1.24.4"`. Borrowed — do not free.
    pub get_version_string: unsafe extern "system" fn() -> *const c_char,
}
