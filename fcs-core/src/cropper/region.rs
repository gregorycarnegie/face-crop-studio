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
    /// The region `(x, y, width, height)` in an `img_w` x `img_h` source, with the padding
    /// each side needs where it runs past the image.
    pub fn new(x: i32, y: i32, width: u32, height: u32, img_w: u32, img_h: u32) -> Self {
        let (x_i, y_i) = (x as i64, y as i64);
        Self {
            x,
            y,
            width,
            height,
            pad_left: (-x_i).max(0) as u32,
            pad_top: (-y_i).max(0) as u32,
            pad_right: (x_i + width as i64 - img_w as i64).max(0) as u32,
            pad_bottom: (y_i + height as i64 - img_h as i64).max(0) as u32,
        }
    }

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
