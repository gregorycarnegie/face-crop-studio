//! Basic image enhancement utilities for Phase 5.
//!
//! Provides a small, deterministic enhancement pipeline used by the CLI
//! when `--enhance` is requested. The implementations are intentionally
//! lightweight and pure-Rust using the `image` crate.

mod detail;
mod gpu;
mod pipeline;
mod red_eye;
mod settings;
mod skin;
mod tone;

#[cfg(test)]
mod benches;
#[cfg(test)]
mod tests;

pub use gpu::WgpuEnhancer;
pub use pipeline::apply_enhancements;
pub use settings::EnhancementSettings;

#[cfg(test)]
use {
    detail::{
        apply_background_blur, apply_background_blur_with_preblur, apply_unsharp_mask,
        apply_unsharp_with_preblur, background_blur_from_rgba,
    },
    image::DynamicImage,
    red_eye::apply_red_eye_removal,
    skin::{apply_skin_smoothing, skin_kernel},
    tone::{
        apply_brightness, apply_contrast, apply_exposure, apply_histogram_equalization,
        apply_saturation, build_equalization_lut,
    },
};

const EPSILON: f32 = 1e-6;
