//! Live result preview: the selected face as it will be exported, re-rendered as settings change.
//!
//! Rendering runs on a background thread through [`super::export::finish_face`], the function
//! export itself calls, so the preview cannot drift from the saved file. One render is in flight
//! at a time; when it lands, the next frame compares the inputs again and starts another if the
//! user has moved on. A slider drag therefore shows the latest value as fast as renders finish,
//! never a queue of stale ones.

use std::{
    collections::HashSet,
    panic::{AssertUnwindSafe, catch_unwind},
    sync::Arc,
};

use egui::{ColorImage, TextureHandle, TextureOptions};
use fcs_core::Detection;
use fcs_utils::{
    config::{CropSettings, EnhanceSettings},
    quality::Quality,
};
use image::DynamicImage;

use crate::types::{App2, JobMessage};

/// Everything a render depends on. A change to any of it makes the preview stale.
#[derive(Clone, PartialEq)]
pub struct PreviewInputs {
    /// Identity of the loaded image. Same pointer, same pixels: the `Arc` is replaced, never
    /// mutated, when a new image loads.
    image: usize,
    face: usize,
    detection: Detection,
    crop: CropSettings,
    enhance: EnhanceSettings,
}

#[derive(Default)]
pub struct ResultPreview {
    /// Show the finished crop on the canvas instead of the source image.
    pub enabled: bool,
    pub texture: Option<TextureHandle>,
    /// Which face `texture` shows, and the quality it was scored at.
    pub shown: Option<(usize, Quality)>,
    /// A render is running; its inputs are `requested`.
    pub in_flight: bool,
    requested: Option<PreviewInputs>,
    generation: u64,
}

impl ResultPreview {
    /// Whether a finished render is the newest one, clearing `in_flight` if so. An older one is
    /// dropped: showing it would flash a setting the user has already moved past.
    fn land(&mut self, generation: u64) -> bool {
        let newest = generation == self.generation;
        if newest {
            self.in_flight = false;
        }
        newest
    }
}

/// The face the preview follows: the lowest-numbered selected face, else the first.
fn preview_face(selected: &HashSet<usize>, faces: usize) -> Option<usize> {
    selected
        .iter()
        .copied()
        .filter(|&i| i < faces)
        .min()
        .or_else(|| (faces > 0).then_some(0))
}

fn current_inputs(app: &App2) -> Option<(PreviewInputs, Arc<DynamicImage>)> {
    let source = app.preview.source_image.clone()?;
    let face = preview_face(&app.selected_faces, app.preview.detections.len())?;
    let inputs = PreviewInputs {
        image: Arc::as_ptr(&source) as usize,
        face,
        detection: app.preview.detections[face].edited_detection(),
        crop: app.settings.crop.clone(),
        enhance: app.settings.enhance.clone(),
    };
    Some((inputs, source))
}

/// Called every frame while the preview is on: start a render if what is on screen is stale.
pub fn refresh(app: &mut App2, ctx: &egui::Context) {
    let Some((inputs, source)) = current_inputs(app) else {
        app.result_preview.texture = None;
        app.result_preview.shown = None;
        app.result_preview.requested = None;
        return;
    };
    let state = &mut app.result_preview;
    if state.in_flight || state.requested.as_ref() == Some(&inputs) {
        return;
    }
    state.generation += 1;
    state.in_flight = true;
    state.requested = Some(inputs.clone());

    let generation = state.generation;
    let enhancement = app.gpu.enhancement.clone();
    let tx = app.job_tx.clone();
    let ctx = ctx.clone();
    std::thread::spawn(move || {
        // A panic here must still clear `in_flight`, or the preview would stop updating.
        let rendered = catch_unwind(AssertUnwindSafe(|| {
            let finished = super::export::finish_face(
                &source,
                &inputs.detection,
                &inputs.crop,
                &inputs.enhance,
                &enhancement,
            );
            (
                super::detection::color_image_from_dynamic(&finished.image),
                finished.quality,
            )
        }))
        .ok();
        let _ = tx.send(JobMessage::ResultPreview {
            generation,
            face: inputs.face,
            rendered,
        });
        ctx.request_repaint();
    });
}

/// A render finished. Only the newest one is shown; `refresh` starts the next if needed.
pub fn receive(
    app: &mut App2,
    ctx: &egui::Context,
    generation: u64,
    face: usize,
    rendered: Option<(ColorImage, Quality)>,
) {
    let state = &mut app.result_preview;
    if !state.land(generation) {
        return;
    }
    match rendered {
        Some((image, quality)) => {
            match &mut state.texture {
                Some(texture) => texture.set(image, TextureOptions::LINEAR),
                None => {
                    state.texture =
                        Some(ctx.load_texture("result_preview", image, TextureOptions::LINEAR))
                }
            }
            state.shown = Some((face, quality));
        }
        None => log::warn!("result preview render panicked; showing the previous one"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn follows_the_lowest_selected_face_else_the_first() {
        assert_eq!(preview_face(&HashSet::new(), 0), None);
        assert_eq!(preview_face(&HashSet::new(), 3), Some(0));
        assert_eq!(preview_face(&HashSet::from([2, 1]), 3), Some(1));
        // A selection left over from a deleted face falls back rather than indexing past the end.
        assert_eq!(preview_face(&HashSet::from([5]), 3), Some(0));
    }

    #[test]
    fn only_the_newest_render_lands() {
        let mut state = ResultPreview {
            in_flight: true,
            generation: 2,
            ..Default::default()
        };
        assert!(!state.land(1));
        assert!(
            state.in_flight,
            "a stale render must not end the one still running"
        );
        assert!(state.land(2));
        assert!(!state.in_flight);
    }
}
