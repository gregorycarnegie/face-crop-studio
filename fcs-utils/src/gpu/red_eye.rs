use bytemuck::{Pod, Zeroable};

/// Circular correction region in image pixels.
///
/// Red-eye removal itself runs on the CPU (see `enhance::red_eye`); the layout is a
/// leftover of the GPU pass it replaced.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct RedEye {
    /// Horizontal eye-center coordinate in pixels from the image's left edge.
    pub x: f32,
    /// Vertical eye-center coordinate in pixels from the image's top edge.
    pub y: f32,
    /// Radius of the correction region in pixels.
    pub radius: f32,
    /// Unused padding; initialize to 0.0.
    pub _pad: f32,
}
