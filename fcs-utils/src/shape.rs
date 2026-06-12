//! Shape utilities for custom crop geometry and masking.
//!
//! Provides the `CropShape` enum shared across the workspace together with helpers
//! for generating polygon outlines and applying alpha masks to RGBA images.

mod mask;
mod outline;
#[cfg(test)]
mod tests;
mod types;

pub use mask::{apply_shape_mask, apply_shape_mask_dynamic};
pub use outline::outline_points_for_rect;
pub use types::{CropShape, PolygonCornerStyle};
