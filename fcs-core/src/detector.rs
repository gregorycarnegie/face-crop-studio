//! High-level detector abstraction combining preprocessing, model inference, and postprocessing.
//!
//! This module exposes [`YuNetDetector`], the primary interface used by both the CLI and GUI
//! front-ends to run YuNet against images.

use crate::{
    gpu::runtime::GpuYuNet,
    model::YuNetModel,
    postprocess::{Detection, PostprocessConfig, apply_postprocess},
    preprocess::{CpuPreprocessor, InputSize, PreprocessConfig, PreprocessOutput, Preprocessor},
};

use crate::tensor::Tensor;
use anyhow::{Context, Result};
use fcs_utils::{InputFit, load_image, timing_guard};
use image::DynamicImage;
use std::{path::Path, sync::Arc};

/// Result of running YuNet on an image.
///
/// Contains the final list of detections along with metadata to map them
/// back to the original image's coordinate space.
#[derive(Debug)]
pub struct DetectionOutput {
    /// A list of detected faces.
    pub detections: Vec<Detection>,
    /// The horizontal scale factor to convert detection coordinates to the original image space.
    pub scale_x: f32,
    /// The vertical scale factor to convert detection coordinates to the original image space.
    pub scale_y: f32,
    /// The original dimensions of the input image.
    pub original_size: (u32, u32),
}

/// Convenience wrapper that couples the YuNet model with preprocessing and postprocessing settings.
///
/// This is the main entry point for running face detection.
#[derive(Debug)]
pub struct YuNetDetector {
    backend: Arc<DetectorBackend>,
    preprocess: PreprocessConfig,
    postprocess: PostprocessConfig,
    preprocessor: Arc<dyn Preprocessor>,
}

#[derive(Debug)]
enum DetectorBackend {
    Cpu(Box<YuNetModel>),
    Gpu(GpuYuNet),
}

impl DetectorBackend {
    fn run(&self, tensor: Tensor) -> Result<Tensor> {
        match self {
            DetectorBackend::Cpu(model) => model.run(tensor),
            DetectorBackend::Gpu(model) => model.run(tensor),
        }
    }

    /// Run one zeroed tensor through the backend so a bad input size is caught here rather
    /// than on every image.
    ///
    /// The shipped YuNet has a fixed 640x640 input, but `input.width` and `input.height` are
    /// settings and nothing rejected other values. A folder run then loaded the model, read
    /// every file, and failed each one separately -- "Got invalid dimensions for input" from
    /// ONNX Runtime, "stage0 conv" from the WGSL graph -- before ending with "all detections
    /// failed". Probing once costs a single inference at construction (experiment 74).
    ///
    /// Deliberately not a hardcoded 640: a caller may supply a model built for another size,
    /// and the backend is the thing that knows.
    fn probe_input_size(&self, input_size: InputSize) -> Result<()> {
        let elements = 3 * input_size.height as usize * input_size.width as usize;
        let probe = Tensor::from_vec(
            &[1, 3, input_size.height as usize, input_size.width as usize],
            vec![0.0f32; elements],
        )
        .context("failed to build the input-size probe tensor")?;

        self.run(probe).with_context(|| {
            format!(
                "this backend rejected a {}x{} input; the bundled YuNet model and the GPU graph are both fixed at 640x640, so check `input.width` and `input.height` in your settings",
                input_size.width, input_size.height
            )
        })?;
        Ok(())
    }
}

/// Take the letterbox bars back off, turning model coordinates into source pixels.
///
/// The source is fitted into the model input rather than stretched to it, so a detection
/// comes out offset by however wide the bars were. `apply_postprocess` has already
/// multiplied by the scale, and subtracting `origin * scale` afterwards is the same as
/// subtracting `origin` before it -- which keeps the decode, and its sixteen call sites,
/// unaware that letterboxing exists.
///
/// Box sizes are untouched: a bar shifts a face, it does not resize one.
fn remove_letterbox_offset(detections: &mut [Detection], fit: &InputFit) {
    let (offset_x, offset_y) = fit.source_offset();
    if offset_x == 0.0 && offset_y == 0.0 {
        return;
    }
    for detection in detections {
        detection.bbox.x -= offset_x;
        detection.bbox.y -= offset_y;
        for landmark in &mut detection.landmarks {
            landmark.x -= offset_x;
            landmark.y -= offset_y;
        }
    }
}

impl YuNetDetector {
    /// Construct a detector from a model path and configuration.
    ///
    /// # Arguments
    ///
    /// * `model_path` - The path to the ONNX model file.
    /// * `preprocess` - The configuration for image preprocessing.
    /// * `postprocess` - The configuration for detection postprocessing.
    pub fn new<P: AsRef<Path>>(
        model_path: P,
        preprocess: PreprocessConfig,
        postprocess: PostprocessConfig,
    ) -> Result<Self> {
        let cpu = Arc::new(CpuPreprocessor);
        Self::with_preprocessor(model_path, preprocess, postprocess, cpu)
    }

    /// Construct a detector with a custom preprocessor implementation.
    pub fn with_preprocessor<P: AsRef<Path>>(
        model_path: P,
        preprocess: PreprocessConfig,
        postprocess: PostprocessConfig,
        preprocessor: Arc<dyn Preprocessor>,
    ) -> Result<Self> {
        let model = YuNetModel::load(model_path.as_ref(), preprocess.input_size)?;
        let backend = DetectorBackend::Cpu(Box::new(model));
        backend.probe_input_size(preprocess.input_size)?;
        Ok(Self {
            backend: Arc::new(backend),
            preprocess,
            postprocess,
            preprocessor,
        })
    }

    /// Construct a detector that executes inference on the GPU (if available).
    pub fn new_gpu<P: AsRef<Path>>(
        model_path: P,
        preprocess: PreprocessConfig,
        postprocess: PostprocessConfig,
    ) -> Result<Self> {
        let cpu = Arc::new(CpuPreprocessor);
        Self::with_gpu_preprocessor(model_path, preprocess, postprocess, cpu)
    }

    /// GPU-backed detector with a custom preprocessor implementation.
    pub fn with_gpu_preprocessor<P: AsRef<Path>>(
        model_path: P,
        preprocess: PreprocessConfig,
        postprocess: PostprocessConfig,
        preprocessor: Arc<dyn Preprocessor>,
    ) -> Result<Self> {
        // Reuse the preprocessor's device when it has one. Two independent `wgpu::Device`s
        // cannot share tensors, so this is what makes `detect_on_device` possible at all -- and
        // it also avoids initialising a second adapter for the same physical GPU.
        let model = match preprocessor.as_wgpu() {
            Some(gpu) => GpuYuNet::with_context(
                gpu.context().clone(),
                model_path.as_ref(),
                preprocess.input_size,
            )?,
            None => GpuYuNet::new(model_path.as_ref(), preprocess.input_size)?,
        };
        let backend = DetectorBackend::Gpu(model);
        backend.probe_input_size(preprocess.input_size)?;
        Ok(Self {
            backend: Arc::new(backend),
            preprocess,
            postprocess,
            preprocessor,
        })
    }

    /// Run detection on an image file path.
    ///
    /// # Arguments
    ///
    /// * `path` - The path to the image file.
    pub fn detect_path<P: AsRef<Path>>(&self, path: P) -> Result<DetectionOutput> {
        let _guard = timing_guard("fcs_core::detect_path", log::Level::Debug);
        let path_ref = path.as_ref();
        let image = load_image(path_ref)
            .with_context(|| format!("failed to load image from {}", path_ref.display()))?;
        self.detect_image(&image)
    }

    /// Run detection on an in-memory dynamic image.
    ///
    /// # Arguments
    ///
    /// * `image` - The dynamic image to process.
    pub fn detect_image(&self, image: &DynamicImage) -> Result<DetectionOutput> {
        let _guard = timing_guard("fcs_core::detect_image", log::Level::Debug);
        if let Some(output) = self.detect_on_device(image)? {
            return Ok(output);
        }
        let prep = self.preprocessor.preprocess(image, &self.preprocess)?;
        self.run_preprocessed(prep)
    }

    /// End-to-end GPU detection: preprocess writes straight into the tensor inference reads.
    ///
    /// Returns `Ok(None)` whenever the pairing does not apply -- CPU inference, a CPU
    /// preprocessor, a preprocessor on a different device, or an image too large for a single
    /// texture -- leaving the caller to take the ordinary path.
    ///
    /// When it does apply, nothing crosses the PCIe bus between the two stages. The tensor is
    /// allocated on the model's device, the preprocess dispatch writes it, and inference reads
    /// it from the same queue, which orders the two submissions without host synchronisation.
    /// The path this replaces downloaded 4.9 MB through a blocking map and uploaded the same
    /// 4.9 MB straight back -- to a second device, since the two stages used to build their own.
    ///
    /// That saving is only a saving while the source is small. Fusing sends the image up at
    /// full resolution, so a 10 MP photo uploads 40 MB to avoid a 9.8 MB round trip, and the
    /// CPU pays a full-resolution `to_rgba8` first. Measured on an RTX 4090 the preprocess
    /// shader itself runs in 0.04 ms while the path around it takes 6.5 ms, against 0.6 ms for
    /// resizing on the CPU and uploading the 640x640 tensor -- so past roughly 1.2 MP the
    /// fusion loses, and it loses by more the larger the image gets. See
    /// `examples/preprocess_cost.rs`, which prints both sides and the crossover.
    fn detect_on_device(&self, image: &DynamicImage) -> Result<Option<DetectionOutput>> {
        let DetectorBackend::Gpu(model) = &*self.backend else {
            return Ok(None);
        };
        let Some(gpu_preprocessor) = self.preprocessor.as_wgpu() else {
            return Ok(None);
        };
        if !Arc::ptr_eq(gpu_preprocessor.context(), model.context()) {
            return Ok(None);
        }

        let _guard = timing_guard("fcs_core::detect_on_device", log::Level::Debug);
        let input = {
            let _guard = timing_guard("fcs_core::allocate_input", log::Level::Trace);
            model.allocate_input(self.preprocess.input_size)?
        };
        let Some(scales) =
            gpu_preprocessor.preprocess_into_tensor(image, &self.preprocess, &input)?
        else {
            return Ok(None);
        };

        let raw = {
            let _guard = timing_guard("fcs_core::onnx_inference", log::Level::Debug);
            model.run_on_device_filtered(&input, Some(self.postprocess.score_threshold))?
        };
        let detections = {
            let _guard = timing_guard("fcs_core::postprocess", log::Level::Debug);
            let mut detections =
                apply_postprocess(&raw, scales.scale_x, scales.scale_y, &self.postprocess)?;
            remove_letterbox_offset(&mut detections, &scales.fit);
            detections
        };

        Ok(Some(DetectionOutput {
            detections,
            scale_x: scales.scale_x,
            scale_y: scales.scale_y,
            original_size: scales.original_size,
        }))
    }

    /// Returns the estimated GPU memory usage in bytes, or None if running on CPU.
    pub fn gpu_memory_usage(&self) -> Option<u64> {
        match &*self.backend {
            DetectorBackend::Cpu(_) => None,
            DetectorBackend::Gpu(gpu) => Some(gpu.memory_usage()),
        }
    }

    /// Access the underlying postprocess configuration.
    /// Which inference backend this detector actually ended up on.
    ///
    /// Backends are chosen by what is available at load time, so "which one am
    /// I running?" is not answerable from configuration alone — and it is the
    /// first thing worth knowing when detection is unexpectedly slow.
    pub fn inference_backend(&self) -> &'static str {
        match &*self.backend {
            DetectorBackend::Gpu(_) => "wgsl-gpu",
            DetectorBackend::Cpu(model) => model.backend_name(),
        }
    }

    pub fn postprocess_config(&self) -> &PostprocessConfig {
        &self.postprocess
    }

    /// This detector with different postprocessing, sharing the loaded model, its compiled
    /// pipelines and the preprocessor.
    ///
    /// Score threshold, NMS and top-k all apply after inference, so changing one needs
    /// nothing rebuilt. The GUI used to rebuild the whole detector for it (experiment 66).
    pub fn with_postprocess(&self, postprocess: PostprocessConfig) -> Self {
        Self {
            backend: Arc::clone(&self.backend),
            preprocess: self.preprocess.clone(),
            postprocess,
            preprocessor: Arc::clone(&self.preprocessor),
        }
    }

    /// Access the preprocessing configuration.
    pub fn preprocess_config(&self) -> &PreprocessConfig {
        &self.preprocess
    }

    /// Run the model on a preprocessed tensor and return the final detections.
    fn run_preprocessed(&self, prep: PreprocessOutput) -> Result<DetectionOutput> {
        let _guard = timing_guard("fcs_core::run_preprocessed", log::Level::Trace);

        let PreprocessOutput {
            tensor,
            scale_x,
            scale_y,
            fit,
            original_size,
        } = prep;

        let raw = {
            let _guard = timing_guard("fcs_core::onnx_inference", log::Level::Debug);
            self.backend.run(tensor)?
        };

        let detections = {
            let _guard = timing_guard("fcs_core::postprocess", log::Level::Debug);
            let mut detections = apply_postprocess(&raw, scale_x, scale_y, &self.postprocess)?;
            remove_letterbox_offset(&mut detections, &fit);
            detections
        };

        Ok(DetectionOutput {
            detections,
            scale_x,
            scale_y,
            original_size,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::postprocess::{BoundingBox, Landmark};
    use fcs_utils::fit_input;

    fn detection(x: f32, y: f32) -> Detection {
        Detection {
            bbox: BoundingBox {
                x,
                y,
                width: 40.0,
                height: 50.0,
            },
            landmarks: [Landmark { x, y }; 5],
            score: 0.9,
        }
    }

    /// The one thing here that can go wrong silently: a detection that is still found, and
    /// still the right size, but placed by however wide the bars were. Every crop would be
    /// off and nothing would fail.
    #[test]
    fn removing_the_offset_lands_a_corner_detection_on_the_source_corner() {
        // 16:9 into a square: bars top and bottom, so only y moves.
        let fit = fit_input((1920, 1080), (640, 640)).expect("fit");
        assert_eq!(fit.origin, (0, 140));

        // A detection sitting exactly on the top-left of the drawn region, already scaled to
        // source pixels by `apply_postprocess`.
        let mut detections = vec![detection(0.0, fit.origin.1 as f32 * fit.scale)];
        remove_letterbox_offset(&mut detections, &fit);

        assert!((detections[0].bbox.x - 0.0).abs() < 0.01);
        assert!(
            detections[0].bbox.y.abs() < 0.01,
            "top of the drawn region should be y=0 in the source, got {}",
            detections[0].bbox.y
        );
        assert!((detections[0].landmarks[0].y).abs() < 0.01);
        // A bar shifts a face; it does not resize one.
        assert_eq!(detections[0].bbox.width, 40.0);
        assert_eq!(detections[0].bbox.height, 50.0);
    }

    #[test]
    fn a_source_that_needs_no_bars_is_left_alone() {
        let fit = fit_input((640, 640), (640, 640)).expect("fit");
        let mut detections = vec![detection(11.0, 22.0)];
        remove_letterbox_offset(&mut detections, &fit);
        assert_eq!(detections[0].bbox.x, 11.0);
        assert_eq!(detections[0].bbox.y, 22.0);
    }
}
