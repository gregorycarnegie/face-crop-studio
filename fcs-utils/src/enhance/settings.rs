//! Enhancement pipeline settings and presets.

/// Settings for the enhancement pipeline.
#[derive(Debug, Clone)]
pub struct EnhancementSettings {
    /// Apply histogram-equalization based auto color correction.
    pub auto_color: bool,
    /// Exposure adjustment expressed in stops (-2.0..=2.0).
    pub exposure_stops: f32,
    /// Additional brightness offset applied after exposure.
    pub brightness: i32,
    /// Contrast multiplier (0.5..=2.0, 1.0 = unchanged).
    pub contrast: f32,
    /// Saturation multiplier (0.0..=2.5, 1.0 = unchanged).
    pub saturation: f32,
    /// Strength of unsharp mask (0.0..=2.0).
    pub unsharp_amount: f32,
    /// Blur radius used for the unsharp mask in pixels.
    pub unsharp_radius: f32,
    /// Additional sharpening control layered on top of the base amount.
    pub sharpness: f32,
    /// Skin smoothing strength (0.0 = off, 1.0 = maximum).
    pub skin_smooth_amount: f32,
    /// Spatial sigma for bilateral filter (controls spatial extent).
    pub skin_smooth_sigma_space: f32,
    /// Color sigma for bilateral filter (controls color similarity threshold).
    pub skin_smooth_sigma_color: f32,
    /// Enable automated red-eye removal.
    pub red_eye_removal: bool,
    /// Red-eye detection threshold (higher = more selective).
    pub red_eye_threshold: f32,
    /// Enable background blur (portrait mode effect).
    pub background_blur: bool,
    /// Background blur strength (radius in pixels).
    pub background_blur_radius: f32,
    /// Background blur mask size (0.0-1.0, larger = more area kept sharp).
    pub background_blur_mask_size: f32,
}

impl Default for EnhancementSettings {
    fn default() -> Self {
        Self {
            auto_color: false,
            exposure_stops: 0.0,
            brightness: 0,
            contrast: 1.0,
            saturation: 1.0,
            unsharp_amount: 0.6,
            unsharp_radius: 1.0,
            sharpness: 0.0,
            skin_smooth_amount: 0.0,
            skin_smooth_sigma_space: 3.0,
            skin_smooth_sigma_color: 25.0,
            red_eye_removal: false,
            red_eye_threshold: 1.5,
            background_blur: false,
            background_blur_radius: 15.0,
            background_blur_mask_size: 0.6,
        }
    }
}

impl EnhancementSettings {
    /// Gentle preset: light tonal lift with subtle sharpening.
    pub fn natural() -> Self {
        Self {
            auto_color: true,
            exposure_stops: 0.1,
            contrast: 1.1,
            saturation: 1.05,
            sharpness: 0.2,
            ..Self::default()
        }
    }

    /// Punchier preset: warmer exposure, higher contrast and saturation.
    pub fn vivid() -> Self {
        Self {
            exposure_stops: 0.3,
            brightness: 10,
            contrast: 1.25,
            saturation: 1.3,
            unsharp_amount: 0.9,
            unsharp_radius: 1.2,
            sharpness: 0.5,
            ..Self::default()
        }
    }

    /// Headshot preset: balanced tone with stronger detail enhancement.
    pub fn professional() -> Self {
        Self {
            auto_color: true,
            exposure_stops: 0.2,
            contrast: 1.15,
            saturation: 1.05,
            unsharp_amount: 1.2,
            sharpness: 0.8,
            ..Self::default()
        }
    }

    /// Resolve a preset name (case-sensitive lowercase). Returns `None` for unknown names.
    pub fn preset_by_name(name: &str) -> Option<Self> {
        match name {
            "natural" => Some(Self::natural()),
            "vivid" => Some(Self::vivid()),
            "professional" => Some(Self::professional()),
            _ => None,
        }
    }
}
