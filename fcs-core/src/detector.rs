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
use fcs_utils::{load_image, timing_guard};
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
    backend: DetectorBackend,
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
            backend,
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
            backend,
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
        let DetectorBackend::Gpu(model) = &self.backend else {
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
            model.run_on_device(&input)?
        };
        let detections = {
            let _guard = timing_guard("fcs_core::postprocess", log::Level::Debug);
            apply_postprocess(&raw, scales.scale_x, scales.scale_y, &self.postprocess)?
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
        match &self.backend {
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
        match &self.backend {
            DetectorBackend::Gpu(_) => "wgsl-gpu",
            DetectorBackend::Cpu(model) => model.backend_name(),
        }
    }

    pub fn postprocess_config(&self) -> &PostprocessConfig {
        &self.postprocess
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
            original_size,
        } = prep;

        let raw = {
            let _guard = timing_guard("fcs_core::onnx_inference", log::Level::Debug);
            self.backend.run(tensor)?
        };

        let detections = {
            let _guard = timing_guard("fcs_core::postprocess", log::Level::Debug);
            apply_postprocess(&raw, scale_x, scale_y, &self.postprocess)?
        };

        Ok(DetectionOutput {
            detections,
            scale_x,
            scale_y,
            original_size,
        })
    }
}
