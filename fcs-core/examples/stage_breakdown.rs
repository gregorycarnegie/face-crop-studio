//! Full per-stage cost of one detection, so optimisation targets something real.
use fcs_core::{
    InputSize, PostprocessConfig, PreprocessConfig, YuNetDetector, preprocess_dynamic_image,
};
use fcs_utils::config::ResizeQuality;
use rayon::prelude::*;
use std::time::Instant;

#[global_allocator]
static ALLOC: mimalloc::MiMalloc = mimalloc::MiMalloc;

const MODEL: &str = "models/face_detection_yunet_2023mar_640.onnx";

fn med(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

fn time<T>(n: usize, mut f: impl FnMut() -> T) -> f64 {
    for _ in 0..2 {
        std::hint::black_box(f());
    }
    med((0..n)
        .map(|_| {
            let t = Instant::now();
            std::hint::black_box(f());
            t.elapsed().as_secs_f64() * 1000.0
        })
        .collect())
}

fn main() -> anyhow::Result<()> {
    let path = std::env::args()
        .nth(1)
        .unwrap_or("fixtures/images/006.jpg".into());
    let bytes = std::fs::read(&path)?;
    let img = image::open(&path)?;
    let cfg = PreprocessConfig {
        input_size: InputSize::new(640, 640),
        resize_quality: ResizeQuality::Quality,
    };
    let det = YuNetDetector::new(MODEL, cfg.clone(), PostprocessConfig::default())?;

    let decode = time(10, || image::load_from_memory(&bytes).unwrap());
    let preproc = time(15, || preprocess_dynamic_image(&img, &cfg).unwrap());
    let full = time(10, || det.detect_image(&img).unwrap());
    let tensor = preprocess_dynamic_image(&img, &cfg)?.tensor;
    let model = fcs_core::YuNetModel::load(MODEL, InputSize::new(640, 640))?;
    let infer = time(10, || model.run(tensor.clone()).unwrap());
    let cpu = fcs_core::cpu::runtime::CpuYuNet::load(
        std::path::Path::new(MODEL),
        InputSize::new(640, 640),
    )?;
    let cpu_ms = time(10, || cpu.run(tensor.clone()).unwrap());

    println!("{path}  {:?}\n", (img.width(), img.height()));
    println!("  jpeg decode        {decode:7.2} ms   (per image in batch; not in detect_image)");
    println!("  preprocess         {preproc:7.2} ms   (resize + BGR CHW)");
    println!("  inference (backend) {infer:6.2} ms");
    println!("  inference (pure-rust cpu) {cpu_ms:6.2} ms");
    println!("  ------------------------------------");
    println!("  detect_image       {full:7.2} ms   (preprocess + inference + postprocess)");
    println!(
        "  postprocess        {:7.2} ms   (by difference)",
        full - preproc - infer
    );
    println!(
        "\n  end-to-end per image (decode + detect) {:7.2} ms",
        decode + full
    );
    println!(
        "  inference share of detect_image: {:.0}%",
        100.0 * infer / full
    );
    println!(
        "  inference share of end-to-end:   {:.0}%",
        100.0 * infer / (decode + full)
    );

    // Per-image latency and batch throughput diverge sharply here: batch runs
    // detections concurrently over one shared detector, so a backend that
    // serialises can lose to a slower one that parallelises.
    let mut files: Vec<_> = std::fs::read_dir("fixtures/images")?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|e| e.eq_ignore_ascii_case("jpg")))
        .collect();
    files.sort();
    files.truncate(20);
    let imgs: Vec<_> = files.iter().filter_map(|p| image::open(p).ok()).collect();
    let run_batch = || {
        imgs.par_iter().for_each(|i| {
            if let Ok(p) = preprocess_dynamic_image(i, &cfg) {
                let _ = model.run(p.tensor);
            }
        })
    };
    for _ in 0..2 {
        run_batch();
    }
    let t = Instant::now();
    run_batch();
    let batch = t.elapsed().as_secs_f64() * 1000.0;
    println!(
        "
  batch: {} images via rayon  {batch:7.1} ms  ({:.2} ms/image)",
        imgs.len(),
        batch / imgs.len() as f64
    );
    println!(
        "  rayon threads {}   backend {}",
        rayon::current_num_threads(),
        model.backend_name()
    );
    Ok(())
}
