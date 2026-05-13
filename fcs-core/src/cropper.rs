//! Crop calculation utilities for face-centric image crops.
//!
//! Implements a scale-based approach where the source crop region is computed
//! from a desired face height percentage of the output image. The crop region
//! preserves the output aspect ratio and records any padding required when the region extends past the image boundaries.

use crate::postprocess::BoundingBox;

use fcs_utils::color::RgbaColor;

/// Alias exported alongside [`CropSettings`] so downstream crates don't need to depend on `fcs_utils::color`.
pub type FillColor = RgbaColor;

/// How to position the face within the crop region.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PositioningMode {
    /// Center the face in the crop region.
    Center,
    /// Place the face roughly at the upper third (rule of thirds).
    RuleOfThirds,
    /// Use custom offsets (fractions) to nudge the face relative to crop center.
    Custom,
}

/// Settings controlling how crops are computed and the output size.
#[derive(Debug, Clone)]
pub struct CropSettings {
    /// Desired output width in pixels.
    pub output_width: u32,
    /// Desired output height in pixels.
    pub output_height: u32,
    /// Percentage of the output's height that the face should occupy (e.g. 70.0).
    /// Must be > 0.0 and <= 100.0.
    pub face_height_pct: f32,
    /// Positioning mode to use when placing the face inside the crop.
    pub positioning_mode: PositioningMode,
    /// Horizontal offset used only for `Custom` mode. Range expected -1.0..=1.0
    /// where negative moves the face left and positive moves it right.
    pub horizontal_offset: f32,
    /// Vertical offset used only for `Custom` mode. Range expected -1.0..=1.0
    /// where negative moves the face upward and positive moves it downward.
    pub vertical_offset: f32,
    /// Background fill color for areas that fall outside the source image.
    pub fill_color: FillColor,
    /// Rotate the crop so the eye line is horizontal.
    pub eye_line_align: bool,
}

/// Integer crop region in source image coordinates.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CropRegion {
    /// Top-left x coordinate (may extend outside the source image).
    pub x: i32,
    /// Top-left y coordinate (may extend outside the source image).
    pub y: i32,
    /// Width of the crop region before any padding is applied.
    pub width: u32,
    /// Height of the crop region before any padding is applied.
    pub height: u32,
    /// Number of pixels that need to be padded on the left.
    pub pad_left: u32,
    /// Number of pixels that need to be padded on the top.
    pub pad_top: u32,
    /// Number of pixels that need to be padded on the right.
    pub pad_right: u32,
    /// Number of pixels that need to be padded on the bottom.
    pub pad_bottom: u32,
}

impl CropRegion {
    /// Returns true if any padding is required to realize this crop.
    pub fn requires_padding(&self) -> bool {
        self.pad_left > 0 || self.pad_top > 0 || self.pad_right > 0 || self.pad_bottom > 0
    }

    /// Width of the sub-rectangle that actually intersects the source image.
    pub fn in_bounds_width(&self) -> u32 {
        self.width
            .saturating_sub(self.pad_left.saturating_add(self.pad_right))
    }

    /// Height of the sub-rectangle that actually intersects the source image.
    pub fn in_bounds_height(&self) -> u32 {
        self.height
            .saturating_sub(self.pad_top.saturating_add(self.pad_bottom))
    }

    /// Returns the in-bounds rectangle (x, y, width, height) in source coordinates, if any.
    pub fn in_bounds_rect(&self, img_w: u32, img_h: u32) -> Option<(u32, u32, u32, u32)> {
        let start_x = self.x.max(0) as u32;
        let start_y = self.y.max(0) as u32;
        let max_w = img_w.saturating_sub(start_x);
        let max_h = img_h.saturating_sub(start_y);
        let width = self.in_bounds_width().min(max_w);
        let height = self.in_bounds_height().min(max_h);
        if width == 0 || height == 0 {
            None
        } else {
            Some((start_x, start_y, width, height))
        }
    }
}

impl Default for CropSettings {
    fn default() -> Self {
        Self {
            output_width: 400,
            output_height: 400,
            face_height_pct: 70.0,
            positioning_mode: PositioningMode::Center,
            horizontal_offset: 0.0,
            vertical_offset: 0.0,
            fill_color: FillColor::default(),
            eye_line_align: false,
        }
    }
}

/// Calculate a crop region in source image coordinates.
///
/// The algorithm proceeds as follows:
/// 1. Clamp `face_height_pct` into the `[1, 100]` range to keep the math stable.
/// 2. Derive the source crop height so that, after resizing, the detected face
///    occupies the requested percentage of the output height.
/// 3. Derive the source crop width that preserves the requested output aspect ratio.
/// 4. Shift the crop center according to [`PositioningMode`], applying clamped custom offsets.
/// 5. Record padding for portions of the rectangle that fall outside the source image.
///
/// # Arguments
/// - `img_w`, `img_h`: source image dimensions in pixels.
/// - `face_bbox`: detected face bounding box in source coordinates.
/// - `settings`: configuration that controls output size and target face coverage.
///
/// # Examples
///
/// ```rust
/// # use fcs_core::cropper::{calculate_crop_region, CropSettings, PositioningMode};
/// # use fcs_core::postprocess::BoundingBox;
/// let settings = CropSettings {
///     output_width: 400,
///     output_height: 400,
///     face_height_pct: 50.0,
///     positioning_mode: PositioningMode::Center,
///     ..Default::default()
/// };
/// let crop = calculate_crop_region(
///     1000,
///     1000,
///     BoundingBox {
///         x: 400.0,
///         y: 400.0,
///         width: 200.0,
///         height: 200.0,
///     },
///     &settings,
/// );
/// assert_eq!(crop.x, 300);
/// assert_eq!(crop.y, 300);
/// assert_eq!(crop.width, 400);
/// assert_eq!(crop.height, 400);
/// ```
///
/// Positioning modes allow the caller to shift the face within the crop:
///
/// ```rust
/// # use fcs_core::cropper::{calculate_crop_region, CropSettings, PositioningMode};
/// # use fcs_core::postprocess::BoundingBox;
/// let bbox = BoundingBox {
///     x: 400.0,
///     y: 300.0,
///     width: 200.0,
///     height: 200.0,
/// };
/// let center = calculate_crop_region(
///     1200,
///     800,
///     bbox,
///     &CropSettings {
///         output_width: 600,
///         output_height: 400,
///         face_height_pct: 50.0,
///         positioning_mode: PositioningMode::Center,
///         ..Default::default()
///     },
/// );
/// let thirds = calculate_crop_region(
///     1200,
///     800,
///     bbox,
///     &CropSettings {
///         output_width: 600,
///         output_height: 400,
///         face_height_pct: 50.0,
///         positioning_mode: PositioningMode::RuleOfThirds,
///         ..Default::default()
///     },
/// );
/// let custom = calculate_crop_region(
///     1200,
///     800,
///     bbox,
///     &CropSettings {
///         output_width: 600,
///         output_height: 400,
///         face_height_pct: 50.0,
///         positioning_mode: PositioningMode::Custom,
///         horizontal_offset: 0.5,
///         vertical_offset: -0.5,
///         ..Default::default()
///     },
/// );
/// assert!(thirds.y < center.y, "rule of thirds nudges the face upward");
/// assert!(custom.x > center.x, "positive horizontal offset moves right");
/// ```
pub fn calculate_crop_region(
    img_w: u32,
    img_h: u32,
    face_bbox: BoundingBox,
    settings: &CropSettings,
) -> CropRegion {
    // Safety clamps and early guards
    let face_h = face_bbox.height.max(1.0);
    let face_cx = face_bbox.width.mul_add(0.5, face_bbox.x);
    let face_cy = face_bbox.height.mul_add(0.5, face_bbox.y);

    let face_height_pct = settings.face_height_pct.clamp(1.0, 100.0);

    // Source region height (in source pixels) so that after resizing the face
    // will occupy `face_height_pct` percent of the output height.
    // Derived from: (face_h / src_region_h) = face_height_pct/100
    let src_h = face_h * (100.0 / face_height_pct);

    // Maintain output aspect ratio for width.
    let out_w = settings.output_width as f32;
    let out_h = settings.output_height as f32;
    let aspect = if out_h > 0.0 { out_w / out_h } else { 1.0 };
    let src_w = src_h * aspect;

    // Determine crop center based on positioning mode
    let mut cx = face_cx;
    let mut cy = face_cy;

    match settings.positioning_mode {
        PositioningMode::Center => {
            // face center remains as-is
        }
        PositioningMode::RuleOfThirds => {
            // Place face center at 1/3 down from the top of the crop region.
            // So crop_top = face_cy - src_h * (1.0/3.0)
            cy = face_cy - src_h / 6.0;
        }
        PositioningMode::Custom => {
            // Offsets are fractions in [-1,1] relative to half the crop dimension.
            let ho = settings.horizontal_offset.clamp(-1.0, 1.0);
            let vo = settings.vertical_offset.clamp(-1.0, 1.0);
            cx = src_w.mul_add(ho * 0.5, face_cx);
            cy = src_h.mul_add(vo * 0.5, face_cy);
        }
    }

    // Compute top-left corner from center
    let left = (-src_w).mul_add(0.5, cx);
    let top = (-src_h).mul_add(0.5, cy);

    // Clamp coordinates to i32 range to prevent undefined behaviour from
    // out-of-range float-to-int casts (NaN, infinity, extreme values).
    let x = left.round().clamp(i32::MIN as f32, i32::MAX as f32) as i32;
    let y = top.round().clamp(i32::MIN as f32, i32::MAX as f32) as i32;
    let width = src_w.round().clamp(1.0, u32::MAX as f32) as u32;
    let height = src_h.round().clamp(1.0, u32::MAX as f32) as u32;

    let x_i = x as i64;
    let y_i = y as i64;
    let width_i = width as i64;
    let height_i = height as i64;
    let img_w_i = img_w as i64;
    let img_h_i = img_h as i64;

    let pad_left = (-x_i).max(0) as u32;
    let pad_top = (-y_i).max(0) as u32;
    let pad_right = (x_i + width_i - img_w_i).max(0) as u32;
    let pad_bottom = (y_i + height_i - img_h_i).max(0) as u32;

    CropRegion {
        x,
        y,
        width,
        height,
        pad_left,
        pad_top,
        pad_right,
        pad_bottom,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn center_crop_basic() {
        let img_w = 1000;
        let img_h = 1000;
        let face = BoundingBox {
            x: 400.0,
            y: 400.0,
            width: 200.0,
            height: 200.0,
        };

        let settings = CropSettings {
            output_width: 400,
            output_height: 400,
            face_height_pct: 50.0,
            positioning_mode: PositioningMode::Center,
            horizontal_offset: 0.0,
            vertical_offset: 0.0,
            fill_color: FillColor::default(),
            eye_line_align: false,
        };

        let crop = calculate_crop_region(img_w, img_h, face, &settings);
        // face_h = 200, src_h = 200 * (100/50) = 400, aspect=1 -> src_w=400
        assert_eq!(crop.width, 400);
        assert_eq!(crop.height, 400);
        assert_eq!(crop.pad_left, 0);
        assert_eq!(crop.pad_top, 0);
        // center at (500,500) => top-left = (300,300)
        assert_eq!(crop.x, 300);
        assert_eq!(crop.y, 300);
    }

    #[test]
    fn clamp_near_top_edge() {
        let img_w = 800;
        let img_h = 600;
        let face = BoundingBox {
            x: 350.0,
            y: 10.0,
            width: 100.0,
            height: 100.0,
        };

        let settings = CropSettings {
            output_width: 400,
            output_height: 600,
            face_height_pct: 30.0,
            positioning_mode: PositioningMode::Center,
            horizontal_offset: 0.0,
            vertical_offset: 0.0,
            fill_color: FillColor::default(),
            eye_line_align: false,
        };

        let crop = calculate_crop_region(img_w, img_h, face, &settings);
        let rect = crop.in_bounds_rect(img_w, img_h).expect("rect");
        assert!(rect.0 + rect.2 <= img_w);
        assert!(rect.1 + rect.3 <= img_h);
    }

    #[test]
    fn rule_of_thirds_moves_face_upwards() {
        let img_w = 1000;
        let img_h = 1000;
        let face = BoundingBox {
            x: 400.0,
            y: 400.0,
            width: 200.0,
            height: 200.0,
        };

        let settings = CropSettings {
            output_width: 400,
            output_height: 600,
            face_height_pct: 50.0,
            positioning_mode: PositioningMode::RuleOfThirds,
            ..Default::default()
        };

        let crop = calculate_crop_region(img_w, img_h, face, &settings);
        // For rule of thirds the crop should be shifted so the face is above center.
        // Compare to center positioning to ensure y is smaller (moved up).
        let center_settings = CropSettings {
            positioning_mode: PositioningMode::Center,
            ..settings.clone()
        };
        let center_crop = calculate_crop_region(img_w, img_h, face, &center_settings);
        assert!(
            crop.y < center_crop.y,
            "rule of thirds should move crop upwards"
        );
    }

    #[test]
    fn custom_offset_moves_crop() {
        let img_w = 1000;
        let img_h = 1000;
        let face = BoundingBox {
            x: 400.0,
            y: 400.0,
            width: 200.0,
            height: 200.0,
        };

        let settings = CropSettings {
            output_width: 400,
            output_height: 400,
            face_height_pct: 50.0,
            positioning_mode: PositioningMode::Custom,
            horizontal_offset: 0.5,
            vertical_offset: -0.5,
            fill_color: FillColor::default(),
            eye_line_align: false,
        };

        let crop = calculate_crop_region(img_w, img_h, face, &settings);
        // Compared to center, x should be larger and y should be smaller
        let center_settings = CropSettings {
            positioning_mode: PositioningMode::Center,
            ..settings.clone()
        };
        let center_crop = calculate_crop_region(img_w, img_h, face, &center_settings);
        assert!(
            crop.x > center_crop.x,
            "custom horizontal offset should move crop right"
        );
        assert!(
            crop.y < center_crop.y,
            "custom vertical offset should move crop up"
        );
    }

    #[test]
    fn clamps_when_face_near_left_and_top_edges() {
        let img_w = 600;
        let img_h = 400;
        let face = BoundingBox {
            x: 5.0,
            y: 8.0,
            width: 90.0,
            height: 110.0,
        };

        let settings = CropSettings {
            output_width: 400,
            output_height: 400,
            face_height_pct: 80.0,
            positioning_mode: PositioningMode::Center,
            horizontal_offset: 0.0,
            vertical_offset: 0.0,
            fill_color: FillColor::default(),
            eye_line_align: false,
        };

        let crop = calculate_crop_region(img_w, img_h, face, &settings);
        assert!(crop.pad_left > 0, "should record left padding");
        assert!(crop.pad_top > 0, "should record top padding");
        let rect = crop.in_bounds_rect(img_w, img_h).expect("in-bounds rect");
        assert_eq!(rect.0, 0, "in-bounds x should be clamped to 0");
        assert_eq!(rect.1, 0, "in-bounds y should be clamped to 0");
    }

    #[test]
    fn clamps_when_face_near_bottom_right_edges() {
        let img_w = 640;
        let img_h = 480;
        let face = BoundingBox {
            x: 520.0,
            y: 360.0,
            width: 100.0,
            height: 100.0,
        };

        let settings = CropSettings {
            output_width: 320,
            output_height: 480,
            face_height_pct: 60.0,
            positioning_mode: PositioningMode::Center,
            horizontal_offset: 0.0,
            vertical_offset: 0.0,
            fill_color: FillColor::default(),
            eye_line_align: false,
        };

        let crop = calculate_crop_region(img_w, img_h, face, &settings);
        assert!(crop.pad_bottom > 0, "should record bottom padding");
        let rect = crop.in_bounds_rect(img_w, img_h).expect("in-bounds rect");
        assert!(rect.0 + rect.2 <= img_w);
        assert!(rect.1 + rect.3 <= img_h);
    }

    #[test]
    fn records_padding_information() {
        let img_w = 200;
        let img_h = 200;
        let face = BoundingBox {
            x: 0.0,
            y: 150.0,
            width: 60.0,
            height: 60.0,
        };

        let settings = CropSettings {
            output_width: 160,
            output_height: 160,
            face_height_pct: 60.0,
            positioning_mode: PositioningMode::Center,
            horizontal_offset: 0.0,
            vertical_offset: 0.0,
            fill_color: FillColor::default(),
            eye_line_align: false,
        };

        let crop = calculate_crop_region(img_w, img_h, face, &settings);
        assert!(crop.requires_padding());
        assert!(crop.pad_top > 0 || crop.pad_bottom > 0);
        let (cx, cy, cw, ch) = crop.in_bounds_rect(img_w, img_h).expect("rect");
        assert!(cx + cw <= img_w);
        assert!(cy + ch <= img_h);
    }

    #[test]
    fn respects_non_square_output_aspect_ratio() {
        let img_w = 1200;
        let img_h = 800;
        let face = BoundingBox {
            x: 500.0,
            y: 300.0,
            width: 120.0,
            height: 120.0,
        };

        let settings = CropSettings {
            output_width: 600,
            output_height: 300,
            face_height_pct: 50.0,
            positioning_mode: PositioningMode::Center,
            horizontal_offset: 0.0,
            vertical_offset: 0.0,
            fill_color: FillColor::default(),
            eye_line_align: false,
        };

        let crop = calculate_crop_region(img_w, img_h, face, &settings);
        assert_eq!(crop.height, 240);
        assert_eq!(crop.width, 480);
        assert!(
            (crop.width as f32 / crop.height as f32 - 2.0).abs() < f32::EPSILON,
            "crop width/height should match requested aspect ratio"
        );
    }

    #[test]
    fn face_height_percent_is_clamped() {
        let img_w = 800;
        let img_h = 800;
        let face = BoundingBox {
            x: 300.0,
            y: 300.0,
            width: 200.0,
            height: 200.0,
        };

        let settings = CropSettings {
            output_width: 400,
            output_height: 400,
            face_height_pct: 150.0,
            positioning_mode: PositioningMode::Center,
            horizontal_offset: 0.0,
            vertical_offset: 0.0,
            fill_color: FillColor::default(),
            eye_line_align: false,
        };

        let crop = calculate_crop_region(img_w, img_h, face, &settings);
        // With clamp to 100%, the source region equals bounding box size.
        assert_eq!(crop.width, 200);
        assert_eq!(crop.height, 200);
    }
}
