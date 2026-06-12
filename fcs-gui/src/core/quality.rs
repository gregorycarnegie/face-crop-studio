//! Quality helpers — thin wrappers around fcs-utils quality functions.

use crate::types::DetectionWithQuality;
use fcs_core::{CropSettings, Detection, crop_face_from_image};
use fcs_utils::{config::AppSettings, quality::Quality};
use image::DynamicImage;
use std::collections::HashSet;

pub fn apply_quality_rules_to_preview(
    detections: &[DetectionWithQuality],
    selected: &mut HashSet<usize>,
    auto_skip_no_high_quality: bool,
    auto_select_best: bool,
) {
    if detections.is_empty() {
        return;
    }

    if auto_select_best {
        let best = detections
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.quality_score.partial_cmp(&b.quality_score).unwrap())
            .map(|(i, _)| i);
        if let Some(idx) = best {
            selected.clear();
            selected.insert(idx);
        }
    }

    if auto_skip_no_high_quality {
        let has_high = detections
            .iter()
            .any(|d| matches!(d.quality, Quality::High));
        if !has_high {
            selected.clear();
        }
    }
}

pub fn refresh_thumbnail(
    ctx: &egui::Context,
    det: &mut DetectionWithQuality,
    source: &DynamicImage,
    settings: &AppSettings,
    texture_seq: &mut u64,
) {
    let crop_settings: CropSettings = (&settings.crop).into();
    let detection = Detection {
        bbox: det.active_bbox(),
        landmarks: det.detection.landmarks,
        score: det.detection.score,
    };
    let raw = crop_face_from_image(source, &detection, &crop_settings);
    // 96×96 thumbnails skip enhancement (bilateral filter / red-eye / sharpening) —
    // the effects are imperceptible at this size and dominate the per-detection cost.
    let thumb = raw.resize(96, 96, image::imageops::FilterType::Triangle);
    let img = super::detection::color_image_from_dynamic(&thumb);
    let name = format!("thumb_{}", texture_seq);
    *texture_seq = texture_seq.wrapping_add(1);
    det.thumbnail = Some(ctx.load_texture(name, img, egui::TextureOptions::LINEAR));
}
