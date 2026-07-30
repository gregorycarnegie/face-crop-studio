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

#[cfg(test)]
mod tests {
    use super::*;

    /// The preset values are the product decision this module exists to hold;
    /// nothing else in the codebase asserts them, so any one of them could be
    /// changed silently. Each preset pins both the fields it sets and a
    /// representative field it inherits from `Default`.

    #[test]
    fn default_settings_are_neutral() {
        let d = EnhancementSettings::default();
        assert!(!d.auto_color);
        assert_eq!(d.exposure_stops, 0.0);
        assert_eq!(d.brightness, 0);
        assert_eq!(d.contrast, 1.0);
        assert_eq!(d.saturation, 1.0);
        assert_eq!(d.unsharp_amount, 0.6);
        assert_eq!(d.unsharp_radius, 1.0);
        assert_eq!(d.sharpness, 0.0);
        assert_eq!(d.skin_smooth_amount, 0.0);
        assert_eq!(d.skin_smooth_sigma_space, 3.0);
        assert_eq!(d.skin_smooth_sigma_color, 25.0);
        assert!(!d.red_eye_removal);
        assert_eq!(d.red_eye_threshold, 1.5);
        assert!(!d.background_blur);
        assert_eq!(d.background_blur_radius, 15.0);
        assert_eq!(d.background_blur_mask_size, 0.6);
    }

    #[test]
    fn natural_preset_is_a_gentle_lift() {
        let s = EnhancementSettings::natural();
        assert!(s.auto_color);
        assert_eq!(s.exposure_stops, 0.1);
        assert_eq!(s.contrast, 1.1);
        assert_eq!(s.saturation, 1.05);
        assert_eq!(s.sharpness, 0.2);
        // Inherited, not set by the preset.
        assert_eq!(s.brightness, 0);
        assert_eq!(s.unsharp_amount, 0.6);
        assert_eq!(s.unsharp_radius, 1.0);
        assert!(!s.red_eye_removal);
    }

    #[test]
    fn vivid_preset_pushes_tone_and_detail() {
        let s = EnhancementSettings::vivid();
        assert_eq!(s.exposure_stops, 0.3);
        assert_eq!(s.brightness, 10);
        assert_eq!(s.contrast, 1.25);
        assert_eq!(s.saturation, 1.3);
        assert_eq!(s.unsharp_amount, 0.9);
        assert_eq!(s.unsharp_radius, 1.2);
        assert_eq!(s.sharpness, 0.5);
        // Vivid is the one preset that leaves auto-colour off.
        assert!(!s.auto_color);
        assert_eq!(s.skin_smooth_amount, 0.0);
    }

    #[test]
    fn professional_preset_favours_detail_over_saturation() {
        let s = EnhancementSettings::professional();
        assert!(s.auto_color);
        assert_eq!(s.exposure_stops, 0.2);
        assert_eq!(s.contrast, 1.15);
        assert_eq!(s.saturation, 1.05);
        assert_eq!(s.unsharp_amount, 1.2);
        assert_eq!(s.sharpness, 0.8);
        // Inherited: the radius is not widened, unlike vivid.
        assert_eq!(s.unsharp_radius, 1.0);
        assert_eq!(s.brightness, 0);
    }

    #[test]
    fn preset_by_name_resolves_each_preset() {
        // Compare a field that is unique to each preset so the arms cannot be
        // swapped without failing.
        assert_eq!(
            EnhancementSettings::preset_by_name("natural").map(|s| s.exposure_stops),
            Some(0.1)
        );
        assert_eq!(
            EnhancementSettings::preset_by_name("vivid").map(|s| s.exposure_stops),
            Some(0.3)
        );
        assert_eq!(
            EnhancementSettings::preset_by_name("professional").map(|s| s.exposure_stops),
            Some(0.2)
        );
    }

    #[test]
    fn preset_by_name_rejects_unknown_and_miscased_names() {
        assert!(EnhancementSettings::preset_by_name("").is_none());
        assert!(EnhancementSettings::preset_by_name("custom").is_none());
        // Documented as case-sensitive lowercase.
        assert!(EnhancementSettings::preset_by_name("Natural").is_none());
        assert!(EnhancementSettings::preset_by_name("VIVID").is_none());
    }
}
