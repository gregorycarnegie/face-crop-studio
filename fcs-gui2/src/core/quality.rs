//! Quality helpers — thin wrappers around fcs-utils quality functions.

use crate::types::DetectionWithQuality;
use fcs_utils::quality::Quality;
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
    texture_seq: &mut u64,
) {
    let bbox = det.active_bbox();
    let x = bbox.x.max(0.0) as u32;
    let y = bbox.y.max(0.0) as u32;
    let w = bbox.width.max(1.0) as u32;
    let h = bbox.height.max(1.0) as u32;
    let x = x.min(source.width().saturating_sub(1));
    let y = y.min(source.height().saturating_sub(1));
    let w = w.min(source.width().saturating_sub(x));
    let h = h.min(source.height().saturating_sub(y));
    let face = source.crop_imm(x, y, w, h);
    let thumb = face.resize(96, 96, image::imageops::FilterType::Triangle);
    let rgba = thumb.to_rgba8();
    let size = [rgba.width() as usize, rgba.height() as usize];
    let img = egui::ColorImage::from_rgba_unmultiplied(size, rgba.as_raw());
    let name = format!("thumb_{}", texture_seq);
    *texture_seq = texture_seq.wrapping_add(1);
    det.thumbnail = Some(ctx.load_texture(name, img, egui::TextureOptions::LINEAR));
}
