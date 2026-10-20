//! Explicit NVIDIA hardware check against the CPU detector on the licensed image fixtures.
use anyhow::{Context, Result, ensure};
use fcs_core::ScrfdDetector;
use fcs_utils::config::GpuSettings;

#[test]
#[ignore = "requires a compatible NVIDIA GPU and the detector model"]
fn cuda_detection_matches_cpu_on_the_image_fixtures() -> Result<()> {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap();
    let model = root.join("models/scrfd80k_500m_640.onnx");
    let cuda = ScrfdDetector::load_from_cuda(&model, &GpuSettings::default())?;
    let cpu = ScrfdDetector::load_from_with_context(&model, None).context("CPU model required")?;
    assert_eq!(cuda.engine(), "cuda-rust");
    let mut images = std::fs::read_dir(root.join("fixtures/images"))?
        .map(|entry| entry.map(|entry| entry.path()))
        .collect::<std::io::Result<Vec<_>>>()?;
    images.retain(|path| fcs_utils::is_supported_image_path(path));
    images.sort();
    ensure!(!images.is_empty(), "image fixtures required");
    for path in images {
        let image = fcs_utils::load_image(&path)?;
        let expected = cpu.detect(&image, 0.5, 0.4)?;
        let actual = cuda.detect(&image, 0.5, 0.4)?;
        assert_eq!(actual.len(), expected.len(), "{}", path.display());
        for (actual, expected) in actual.iter().zip(&expected) {
            assert!(
                (actual.score - expected.score).abs() < 1e-4,
                "{}",
                path.display()
            );
            let a = actual.bbox;
            let b = expected.bbox;
            for (a, b) in [
                (a.x, b.x),
                (a.y, b.y),
                (a.width, b.width),
                (a.height, b.height),
            ] {
                assert!((a - b).abs() < 0.1, "{}: {a} != {b}", path.display());
            }
        }
    }
    Ok(())
}
