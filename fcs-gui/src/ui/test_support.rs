//! Shared `egui_kittest` scaffolding for the widget tests in this module tree.
//!
//! Every widget in `ui/` is an immediate-mode function that only exists inside a
//! live frame, so the only way to assert on click routing or returned state is to
//! run a real egui app. These two helpers are the whole harness; each widget's
//! own tests supply the state struct and the click positions.

use egui::{Pos2, Ui};
use egui_kittest::Harness;

/// Builds a harness around `app` with a fixed viewport.
///
/// The size is pinned because several widgets derive their geometry from
/// `ui.available_width()`; a viewport that varied with the runner would make the
/// computed click positions non-deterministic.
pub fn harness<'a, S>(state: S, app: impl FnMut(&mut Ui, &mut S) + 'a) -> Harness<'a, S> {
    Harness::builder()
        .with_size(egui::vec2(240.0, 120.0))
        .build_ui_state(app, state)
}

/// Presses and releases at `pos`.
///
/// egui needs the pointer over the target on the frame the press lands, so
/// hover, press, and release each get their own frame.
pub fn click_at<S>(h: &mut Harness<'_, S>, pos: Pos2) {
    h.hover_at(pos);
    h.run();
    h.drag_at(pos);
    h.run();
    h.drop_at(pos);
    h.run();
}
