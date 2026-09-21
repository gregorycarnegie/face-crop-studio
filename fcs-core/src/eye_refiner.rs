//! A small convnet that replaces a detector's two eye points with better ones.
//!
//! `face_cropper` uses landmarks 0 and 1 for exactly one thing: the angle of the eye line,
//! which levels the crop. A detector's own eye points are adequate for locating a face and mediocre at that
//! angle -- measured against 1,382 hand-clicked test faces it lands 4.05 degrees out at the
//! median, with only 57.6% of faces within 5 degrees. This model, trained on 1,988 clicked
//! eye pairs with rotation augmentation and an explicit angle term in its loss, gives 1.18
//! degrees and 93.3% within 5. See `tools/dataset/CURVE_RESULTS.md` for how that was
//! measured and what it does not cover.
//!
//! It refines rather than detects. It reads a box somebody else found and rewrites two
//! points inside it, so it is independent of which detector produced that box -- the reason
//! it lives here rather than inside the detector, and the reason replacing the detector does
//! not disturb it.
//!
//! **It runs only under ONNX Runtime.** The built-in CPU graph and the WGSL kernels cover the
//! ops the detector needs, which do not include
//! this model's `GlobalAveragePool` and `Gemm`. Rather than hand-write a second topology
//! twice, [`EyeRefiner::load`] returns `None` when no runtime or no model file is present and
//! callers keep the detector's own landmarks. Releases bundle the runtime on all three
//! platforms, so that fallback is for source builds, not for users.

use std::path::Path;

use image::{DynamicImage, GenericImageView};
use log::{debug, info, warn};

use crate::postprocess::{BoundingBox, Detection, Landmark};

/// The network's input side, fixed at export time.
const SIZE: usize = 112;

/// How much the box is expanded before cropping.
///
/// Load-bearing, and not a free parameter: `tools/dataset/train_eye_refiner.py` builds every
/// evaluation crop as `max(w, h) * 1.25` about the box centre. Feeding a differently-framed
/// crop presents the model with a face at a scale it never trained on, which degrades the
/// output quietly rather than failing.
const BOX_SCALE: f32 = 1.25;

/// Workspace-relative location of the exported model.
const DEFAULT_MODEL: &str = "models/eye_refiner.onnx";

/// A loaded eye-point refiner.
///
/// Construct once with [`Self::load`] and share it; `fcs_ort::Session` allows concurrent
/// runs, so batch processing needs neither a lock nor a pool.
#[derive(Debug)]
pub struct EyeRefiner {
    session: fcs_ort::Session,
}

impl EyeRefiner {
    /// Load the refiner from the default location, or return `None` if it cannot run.
    ///
    /// `None` is the ordinary outcome on a source build with no ONNX Runtime, not an error:
    /// the caller keeps whatever landmarks the detector produced.
    pub fn load() -> Option<Self> {
        Self::load_from(fcs_utils::resolve_data_path(DEFAULT_MODEL))
    }

    /// Load the refiner from an explicit path. See [`Self::load`] for the `None` cases.
    pub fn load_from<P: AsRef<Path>>(path: P) -> Option<Self> {
        let path = path.as_ref();
        if !path.exists() {
            debug!(
                "eye refiner not loaded: no model at {}; keeping detector landmarks",
                path.display()
            );
            return None;
        }
        let environment = fcs_ort::Environment::shared().or_else(|| {
            debug!(
                "eye refiner not loaded: no compatible ONNX Runtime; keeping detector landmarks"
            );
            None
        })?;
        match fcs_ort::Session::new(&environment, path, fcs_ort::SessionOptions::default()) {
            Ok(session) => {
                info!("eye refiner loaded from {}", path.display());
                Some(Self { session })
            }
            Err(err) => {
                // A model that will not open is worth a louder line than an absent one: the
                // file is there, so this is a corrupt or incompatible export rather than a
                // build without the optional runtime.
                warn!(
                    "eye refiner at {} failed to open a session ({err}); keeping detector landmarks",
                    path.display()
                );
                None
            }
        }
    }

    /// Rewrite landmarks 0 and 1 of every detection in place.
    ///
    /// Landmarks 2 to 4 (nose tip and mouth corners) are left exactly as the detector
    /// produced them -- this model predicts eyes only, and writing anything else would be
    /// inventing data. A detection whose inference fails keeps all five of its own points.
    ///
    /// Returns how many detections were refined, for the caller to log.
    pub fn refine(&self, image: &DynamicImage, detections: &mut [Detection]) -> usize {
        let mut refined = 0;
        // ponytail: one forward pass per face. The graph takes a dynamic batch dimension, so
        // faces could be stacked into a single run -- worth doing if a workload ever puts many
        // faces in one frame, since the per-run overhead then dominates. Typical photographs
        // hold one to three.
        for detection in detections.iter_mut() {
            if let Some([left, right]) = self.predict(image, &detection.bbox) {
                detection.landmarks[0] = Some(left);
                detection.landmarks[1] = Some(right);
                refined += 1;
            }
        }
        refined
    }

    /// Predict the two eye points for one box, in source-image coordinates.
    fn predict(&self, image: &DynamicImage, bbox: &BoundingBox) -> Option<[Landmark; 2]> {
        if !(bbox.width > 0.0 && bbox.height > 0.0) {
            return None;
        }
        let crop = CropGeometry::for_box(bbox);
        let input = crop.sample(image);

        let outputs = match self.session.run(&input, &[1, 3, SIZE, SIZE]) {
            Ok(outputs) => outputs,
            Err(err) => {
                warn!("eye refiner inference failed ({err}); keeping detector landmarks");
                return None;
            }
        };
        let eyes = outputs.first()?;
        if eyes.data.len() < 4 {
            warn!(
                "eye refiner returned {} values, expected 4; keeping detector landmarks",
                eyes.data.len()
            );
            return None;
        }

        // The model is trained against points divided by the crop side, so its outputs are in
        // units of the crop, not of the image.
        let size = SIZE as f32;
        Some([
            crop.image_point(eyes.data[0] * size, eyes.data[1] * size),
            crop.image_point(eyes.data[2] * size, eyes.data[3] * size),
        ])
    }
}

/// The square crop the network sees, and the mapping back out of it.
///
/// This reproduces the affine `train_eye_refiner.py` builds with
/// `cv2.getRotationMatrix2D(centre, 0, size / side)` followed by a translation putting the
/// box centre at the crop centre. Written out rather than composed from a matrix type
/// because at zero rotation it is two scales and two offsets, and the inverse is what the
/// sampler and the output mapping both need.
#[derive(Debug, Clone, Copy)]
struct CropGeometry {
    centre_x: f32,
    centre_y: f32,
    /// Side of the source-image square that maps onto the crop.
    side: f32,
}

impl CropGeometry {
    fn for_box(bbox: &BoundingBox) -> Self {
        Self {
            centre_x: bbox.x + bbox.width / 2.0,
            centre_y: bbox.y + bbox.height / 2.0,
            side: bbox.width.max(bbox.height) * BOX_SCALE,
        }
    }

    /// Source-image coordinate of a point given in crop pixels.
    #[inline]
    fn image_point(self, crop_x: f32, crop_y: f32) -> Landmark {
        let half = SIZE as f32 / 2.0;
        let scale = self.side / SIZE as f32;
        Landmark::new(
            (crop_x - half) * scale + self.centre_x,
            (crop_y - half) * scale + self.centre_y,
        )
    }

    /// Build the normalised CHW tensor the network expects.
    ///
    /// Bilinear, with zero outside the image, matching `cv2.warpAffine`'s `INTER_LINEAR` and
    /// default constant border -- a face at the edge of a photograph must be padded the same
    /// way it was during training.
    fn sample(&self, image: &DynamicImage) -> Vec<f32> {
        let plane = SIZE * SIZE;
        let mut tensor = vec![0.0f32; 3 * plane];
        let (width, height) = image.dimensions();

        for row in 0..SIZE {
            for column in 0..SIZE {
                let point = self.image_point(column as f32, row as f32);
                let pixel = sample_bilinear(image, width, height, point.x, point.y);
                let index = row * SIZE + column;
                for (channel, value) in pixel.iter().enumerate() {
                    // The training tensor is (RGB - 127.5) / 128.
                    tensor[channel * plane + index] = (value - 127.5) / 128.0;
                }
            }
        }
        tensor
    }
}

/// One bilinear RGB sample, zero outside the image.
#[inline]
fn sample_bilinear(image: &DynamicImage, width: u32, height: u32, x: f32, y: f32) -> [f32; 3] {
    let left = x.floor();
    let top = y.floor();
    let fx = x - left;
    let fy = y - top;

    let mut out = [0.0f32; 3];
    for (dy, wy) in [(0i64, 1.0 - fy), (1, fy)] {
        if wy == 0.0 {
            continue;
        }
        for (dx, wx) in [(0i64, 1.0 - fx), (1, fx)] {
            if wx == 0.0 {
                continue;
            }
            let px = left as i64 + dx;
            let py = top as i64 + dy;
            if px < 0 || py < 0 || px >= width as i64 || py >= height as i64 {
                continue; // constant border of zero, as cv2 pads by default
            }
            let pixel = image.get_pixel(px as u32, py as u32);
            let weight = wx * wy;
            for channel in 0..3 {
                out[channel] += weight * pixel[channel] as f32;
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{Rgba, RgbaImage};

    fn box_at(x: f32, y: f32, width: f32, height: f32) -> BoundingBox {
        BoundingBox {
            x,
            y,
            width,
            height,
        }
    }

    #[test]
    fn crop_centre_maps_to_the_box_centre() {
        // The output mapping is the one piece of arithmetic with no runtime check behind it:
        // a sign or an offset error here moves every predicted eye without failing anything.
        let bbox = box_at(100.0, 200.0, 80.0, 120.0);
        let crop = CropGeometry::for_box(&bbox);
        let half = SIZE as f32 / 2.0;
        let centre = crop.image_point(half, half);
        assert!((centre.x - 140.0).abs() < 1e-3, "got {}", centre.x);
        assert!((centre.y - 260.0).abs() < 1e-3, "got {}", centre.y);
    }

    #[test]
    fn the_crop_spans_the_expanded_box() {
        // side = max(w, h) * 1.25, so the crop edges sit that far from the centre. This pins
        // BOX_SCALE against the trainer: a crop framed differently from training degrades the
        // prediction silently.
        let bbox = box_at(0.0, 0.0, 100.0, 60.0);
        let crop = CropGeometry::for_box(&bbox);
        let expected = 100.0 * BOX_SCALE;
        let left = crop.image_point(0.0, 0.0);
        let right = crop.image_point(SIZE as f32, 0.0);
        assert!((right.x - left.x - expected).abs() < 1e-3);
    }

    #[test]
    fn sampling_pads_with_zero_outside_the_image() {
        // A face at the edge of a photograph is the case this covers: cv2.warpAffine pads with
        // a constant zero border, and training saw that padding.
        let mut image = RgbaImage::new(4, 4);
        for pixel in image.pixels_mut() {
            *pixel = Rgba([255, 255, 255, 255]);
        }
        let image = DynamicImage::ImageRgba8(image);
        let inside = sample_bilinear(&image, 4, 4, 2.0, 2.0);
        let outside = sample_bilinear(&image, 4, 4, -5.0, 2.0);
        assert_eq!(inside, [255.0, 255.0, 255.0]);
        assert_eq!(outside, [0.0, 0.0, 0.0]);
    }

    #[test]
    fn sampling_interpolates_between_neighbours() {
        let mut image = RgbaImage::new(2, 1);
        image.put_pixel(0, 0, Rgba([0, 0, 0, 255]));
        image.put_pixel(1, 0, Rgba([100, 100, 100, 255]));
        let image = DynamicImage::ImageRgba8(image);
        let middle = sample_bilinear(&image, 2, 1, 0.5, 0.0);
        assert!((middle[0] - 50.0).abs() < 1e-3, "got {}", middle[0]);
    }

    #[test]
    fn a_zero_sized_box_is_skipped_rather_than_sampled() {
        let bbox = box_at(10.0, 10.0, 0.0, 0.0);
        assert!(!(bbox.width > 0.0 && bbox.height > 0.0));
    }

    #[test]
    fn tensor_is_normalised_and_chw() {
        let mut image = RgbaImage::new(64, 64);
        for pixel in image.pixels_mut() {
            *pixel = Rgba([127, 128, 129, 255]);
        }
        let image = DynamicImage::ImageRgba8(image);
        let crop = CropGeometry::for_box(&box_at(8.0, 8.0, 48.0, 48.0));
        let tensor = crop.sample(&image);

        assert_eq!(tensor.len(), 3 * SIZE * SIZE);
        let plane = SIZE * SIZE;
        let centre = (SIZE / 2) * SIZE + SIZE / 2;
        // Each channel carries its own value, which is what CHW means here.
        assert!((tensor[centre] - (127.0 - 127.5) / 128.0).abs() < 1e-5);
        assert!((tensor[plane + centre] - (128.0 - 127.5) / 128.0).abs() < 1e-5);
        assert!((tensor[2 * plane + centre] - (129.0 - 127.5) / 128.0).abs() < 1e-5);
    }
}
