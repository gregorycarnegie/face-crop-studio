use fcs_core::{
    YuNetModel,
    postprocess::{PostprocessConfig, apply_postprocess},
    preprocess::{InputSize, PreprocessConfig, PreprocessOutput, preprocess_image},
};
use std::time::Instant;

fn main() -> anyhow::Result<()> {
    let model_path = "models/face_detection_yunet_2023mar_640.onnx";
    let image_path = "fixtures/images/006.jpg";
    let input_size = InputSize::new(640, 640);

    println!("=== YuNet Pipeline Performance Breakdown ===\n");

    // 1. Model Loading
    let start = Instant::now();
    let model = YuNetModel::load(model_path, input_size)?;
    let load_time = start.elapsed();
    println!(
        "1. Model Loading:     {:>8.4}s ({:>5.1}%)",
        load_time.as_secs_f64(),
        0.0
    );

    // 2. Preprocessing
    let preprocess_config = PreprocessConfig {
        input_size,
        ..Default::default()
    };
    let start = Instant::now();
    let preprocessed = preprocess_image(image_path, &preprocess_config)?;
    let preprocess_time = start.elapsed();

    let PreprocessOutput {
        tensor,
        scale_x,
        scale_y,
        fit: _,
        original_size: _,
    } = preprocessed;

    // 3. Model Inference
    let start = Instant::now();
    let output = model.run(tensor)?;
    let inference_time = start.elapsed();

    // 4. Postprocessing
    let postprocess_config = PostprocessConfig::default();
    let start = Instant::now();
    let detections = apply_postprocess(&output, scale_x, scale_y, &postprocess_config)?;
    let postprocess_time = start.elapsed();

    let total = load_time + preprocess_time + inference_time + postprocess_time;
    let total_secs = total.as_secs_f64();

    println!(
        "2. Preprocessing:     {:>8.4}s ({:>5.1}%)",
        preprocess_time.as_secs_f64(),
        preprocess_time.as_secs_f64() / total_secs * 100.0
    );
    println!(
        "3. Model Inference:   {:>8.4}s ({:>5.1}%)",
        inference_time.as_secs_f64(),
        inference_time.as_secs_f64() / total_secs * 100.0
    );
    println!(
        "4. Postprocessing:    {:>8.4}s ({:>5.1}%)",
        postprocess_time.as_secs_f64(),
        postprocess_time.as_secs_f64() / total_secs * 100.0
    );
    println!("{}", "─".repeat(45));
    println!("   TOTAL:             {:>8.4}s (100.0%)\n", total_secs);

    let load_pct = load_time.as_secs_f64() / total_secs * 100.0;
    println!("Model loading is {:.1}% of total pipeline time", load_pct);
    println!("Found {} face(s) in test image", detections.len());

    Ok(())
}
