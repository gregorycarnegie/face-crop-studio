#[path = "../../fcs-core/src/postprocess.rs"]
#[allow(dead_code)]
pub mod postprocess;

// `include!` rather than `#[path]` so the harnesses can sit in a child module and reach the
// grid's private helpers without changing the production file.
#[allow(dead_code)]
pub mod nms {
    include!("../../fcs-core/src/nms.rs");

    #[cfg(kani)]
    mod proofs;
}

#[path = "../../fcs-core/src/scrfd/decode.rs"]
#[allow(dead_code)]
pub mod decode;

#[cfg(kani)]
mod decode_proofs;
