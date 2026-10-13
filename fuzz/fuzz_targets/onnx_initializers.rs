#![no_main]

use libfuzzer_sys::fuzz_target;

#[allow(dead_code)]
#[path = "../../fcs-core/src/onnx/mod.rs"]
mod onnx;

fuzz_target!(|data: &[u8]| {
    if let Ok(tensors) = onnx::OnnxInitializerMap::from_bytes(data, &["fuzz"]) {
        let tensor = tensors.tensor("fuzz").unwrap();
        assert_eq!(tensor.dims().iter().product::<usize>(), tensor.data().len());
    }
});
