pub mod app;
pub mod core;
pub mod interaction;
pub mod rendering;
pub mod theme;
pub mod types;
pub mod ui;

pub use types::*;

/// When `main` started, for the startup timings logged during the first frame.
///
/// Experiment 81: the only number that matters here is how long the user looks at an
/// empty window, and that is measured from process entry to the first painted frame --
/// not from anything `App::new` can see on its own.
pub static LAUNCH: std::sync::OnceLock<std::time::Instant> = std::sync::OnceLock::new();

/// Milliseconds since [`LAUNCH`], or 0.0 if it was never set (tests, benches).
pub fn since_launch_ms() -> f64 {
    LAUNCH
        .get()
        .map(|t| t.elapsed().as_secs_f64() * 1e3)
        .unwrap_or(0.0)
}
