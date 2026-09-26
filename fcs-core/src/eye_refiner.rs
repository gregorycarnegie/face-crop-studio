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
//! The fixed graph runs in Rust on the CPU, using the detector's convolution kernels.
//! No ONNX Runtime library is needed. If the model is absent or incompatible, callers
//! keep the detector's own landmarks.

mod native;

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
/// Construct once with [`Self::load`] and share it; weights are immutable and each run
/// owns its activations, so batch processing needs no lock.
#[derive(Debug)]
pub struct EyeRefiner {
    weights: native::Weights,
}

impl EyeRefiner {
    /// Load the refiner from the default location, or return `None` if it cannot run.
    ///
    /// Returns `None` if the model is missing or incompatible; the caller keeps the
    /// detector's own landmarks.
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
        match native::Weights::load(path) {
            Ok(weights) => {
                info!(
                    "eye refiner on the built-in CPU graph, from {}",
                    path.display()
                );
                Some(Self { weights })
            }
            Err(err) => {
                // A model that will not open is worth a louder line than an absent one: the
                // file is there, so this is a corrupt or incompatible export.
                warn!(
                    "eye refiner at {} failed to load ({err}); keeping detector landmarks",
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
        // ponytail: one forward pass per face. Add batched inference if photographs with
        // many faces make per-run overhead significant; typical photographs hold one to three.
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

        let eyes = match self.weights.run(input) {
            Ok(eyes) => eyes,
            Err(err) => {
                warn!("eye refiner inference failed ({err}); keeping detector landmarks");
                return None;
            }
        };
        let [left, right] = eye_points(&eyes)?;
        Some([
            crop.image_point(left.0, left.1),
            crop.image_point(right.0, right.1),
        ])
    }
}

/// The two eye points in the model's output, in crop pixels, or `None` when it returned too few
/// values to hold them.
///
/// Its own function so the length check can be tested: inside `predict` it sat behind a real
/// inference, and the model never returns the wrong length on request.
fn eye_points(values: &[f32]) -> Option<[(f32, f32); 2]> {
    if values.len() < 4 {
        warn!(
            "eye refiner returned {} values, expected 4; keeping detector landmarks",
            values.len()
        );
        return None;
    }
    // The model is trained against points divided by the crop side, so its outputs are in
    // units of the crop, not of the image.
    let size = SIZE as f32;
    Some([
        (values[0] * size, values[1] * size),
        (values[2] * size, values[3] * size),
    ])
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

        // Off-centre, on both axes. At the centre `crop_x - half` is zero, so `* scale` and
        // `/ scale` agree and the scale mutants in `image_point` survive -- they did, on the y
        // axis, which no test read away from the centre.
        let side = 120.0 * BOX_SCALE;
        let scale = side / SIZE as f32;
        let quarter = SIZE as f32 / 4.0;
        let off = crop.image_point(quarter, quarter);
        let expected_x = (quarter - half) * scale + 140.0;
        let expected_y = (quarter - half) * scale + 260.0;
        assert!(
            (off.x - expected_x).abs() < 1e-3,
            "x {} vs {expected_x}",
            off.x
        );
        assert!(
            (off.y - expected_y).abs() < 1e-3,
            "y {} vs {expected_y}",
            off.y
        );
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

    /// A 5x3 image whose every pixel is distinct, for sampling tests.
    ///
    /// Non-square so a width/height swap cannot survive, and no value is a multiple of another
    /// so a wrong weight cannot land on the right answer by arithmetic accident.
    fn gradient_image() -> DynamicImage {
        let mut image = RgbaImage::new(5, 3);
        for (x, y, pixel) in image.enumerate_pixels_mut() {
            // 13 and 29 are coprime, so (x, y) -> value is injective across this grid.
            let v = 13 * x as u16 + 29 * y as u16 + 7;
            *pixel = Rgba([v as u8, (v + 3) as u8, (v + 11) as u8, 255]);
        }
        DynamicImage::ImageRgba8(image)
    }

    fn pixel_at(image: &DynamicImage, x: u32, y: u32) -> [f32; 3] {
        let p = image.get_pixel(x, y);
        [p[0] as f32, p[1] as f32, p[2] as f32]
    }

    /// Bilinear weights, against an expected value computed independently here.
    ///
    /// The sample point is deliberately awkward. Both fractions are non-zero, unequal, and not
    /// complements of each other, so none of the four weights coincide and no pair of them can
    /// swap without changing the result. The previous version of this test sampled a *uniform
    /// white* image at *integer* coordinates, which made every weight and every offset in
    /// `sample_bilinear` invisible: 11 of its mutants survived.
    #[test]
    fn sampling_interpolates_with_the_four_corner_weights() {
        let image = gradient_image();
        // Both coordinates have a non-zero integer part on purpose: at y = 0.7 the floor is 0,
        // so `y - top` and `y + top` are the same number and the offset mutant survives. That
        // is what the first draft of this test did.
        let (x, y) = (1.3f32, 1.7f32);
        let (fx, fy) = (0.3f32, 0.7f32);

        let got = sample_bilinear(&image, 5, 3, x, y);
        // The four pixels around (1.3, 1.7): floors are 1 and 1, so the corners are (1,1),
        // (2,1), (1,2) and (2,2).
        let corners = [
            ((1, 1), (1.0 - fx) * (1.0 - fy)),
            ((2, 1), fx * (1.0 - fy)),
            ((1, 2), (1.0 - fx) * fy),
            ((2, 2), fx * fy),
        ];
        for (channel, value) in got.iter().enumerate() {
            let expected: f32 = corners
                .iter()
                .map(|((cx, cy), weight)| weight * pixel_at(&image, *cx, *cy)[channel])
                .sum();
            assert!(
                (value - expected).abs() < 1e-3,
                "channel {channel}: got {value} expected {expected}"
            );
        }
    }

    /// Zero outside the image, on every edge separately.
    ///
    /// Each edge is its own case because the bounds test is four `||`-joined comparisons: with
    /// one sample point they agree, and `||` collapsing to `&&` or a `<` widening to `<=`
    /// survives. The image is 5x3, so the x and y limits are different numbers.
    #[test]
    fn sampling_pads_with_zero_outside_every_edge() {
        let image = gradient_image();
        for (x, y, edge) in [
            (-1.7f32, 1.4f32, "left"),
            (5.4, 1.4, "right"),
            (2.3, -1.6, "top"),
            (2.3, 3.5, "bottom"),
        ] {
            assert_eq!(
                sample_bilinear(&image, 5, 3, x, y),
                [0.0, 0.0, 0.0],
                "{edge} edge should read as zero padding"
            );
        }
        // Just inside the far corner still reads real pixels, so the bounds are not simply
        // rejecting everything.
        let inside = sample_bilinear(&image, 5, 3, 3.6, 1.8);
        assert!(
            inside.iter().all(|v| *v > 0.0),
            "a point inside the image must sample it: {inside:?}"
        );
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

    pub(super) fn strict_tests() -> bool {
        std::env::var("FCS_STRICT_TESTS").is_ok_and(|v| v != "0" && !v.is_empty())
    }

    pub(super) fn refiner_model() -> std::path::PathBuf {
        std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .expect("fcs-core sits in the workspace root")
            .join(DEFAULT_MODEL)
    }

    /// `refine` must report how many faces it changed, and change them.
    ///
    /// Every other test here works on the geometry helpers, so `refine` and `predict` had no
    /// coverage at all: `refine -> 0`, `refine -> 1` and `refined += 1` becoming `-=` all
    /// survived. Two detections, so a hard-coded 1 is distinguishable from a real count.
    #[test]
    fn refine_reports_and_applies_both_predictions() {
        let model = refiner_model();
        let Some(refiner) = EyeRefiner::load_from(&model) else {
            assert!(
                !strict_tests(),
                "FCS_STRICT_TESTS: the refiner did not load from {model:?}"
            );
            eprintln!("skipped: no compatible refiner model");
            return;
        };

        let image =
            DynamicImage::ImageRgba8(RgbaImage::from_pixel(400, 400, Rgba([90, 120, 150, 255])));
        // Two boxes of different sizes at different places, so one prediction cannot stand in
        // for the other.
        let mut detections = vec![
            detection_at(box_at(40.0, 50.0, 90.0, 110.0)),
            detection_at(box_at(220.0, 180.0, 130.0, 70.0)),
        ];
        let refined = refiner.refine(&image, &mut detections);
        assert_eq!(refined, 2, "both detections should have been refined");

        for (index, detection) in detections.iter().enumerate() {
            let (left, right) = (detection.landmarks[0], detection.landmarks[1]);
            assert!(
                left.is_some() && right.is_some(),
                "detection {index} kept absent eyes"
            );
            // Mapped back into crop pixels, which is where the model's own output lives: it
            // predicts fractions of the crop side, so `image_point(v * SIZE, ..)` must land
            // inside the crop and the two eyes must be a real distance apart.
            //
            // The first version of this test only checked "somewhere inside the crop", which
            // is too loose to see the output mapping at all: replacing `v * SIZE` with
            // `v + SIZE` or `v / SIZE` still lands inside, and all eight of those mutants
            // survived. In crop pixels the two failures are obvious -- `+` pushes the point
            // past SIZE, and `/` collapses both eyes onto the same spot.
            let crop = CropGeometry::for_box(&detection.bbox);
            let half = SIZE as f32 / 2.0;
            let scale = crop.side / SIZE as f32;
            let to_crop = |p: Landmark| {
                (
                    (p.x - crop.centre_x) / scale + half,
                    (p.y - crop.centre_y) / scale + half,
                )
            };
            let (l, r) = (to_crop(left.unwrap()), to_crop(right.unwrap()));
            // A central band, not just "inside": the crop is 1.25x the box, so the face fills
            // the middle and its eyes cannot sit against an edge. Measured on this fixture the
            // four coordinates are at 34%, 38%, 61% and 35% of the side, so 10..90% is a wide
            // margin -- wide enough to survive a retrained model, tight enough that `v + SIZE`
            // (which lands at 100%) and `v / SIZE` (which lands at 0%) both fall outside.
            //
            // Separation alone is not enough, and that is worth recording: dividing *one*
            // eye's x by SIZE moves it to the crop's left edge and makes the two eyes further
            // apart, so a separation check passes while the mapping is broken.
            let band = (SIZE as f32 * 0.10)..=(SIZE as f32 * 0.90);
            for (axis, value) in [("x", l.0), ("y", l.1), ("x", r.0), ("y", r.1)] {
                assert!(
                    band.contains(&value),
                    "detection {index}: eye {axis} is at {value} crop pixels, outside the                      central band {band:?}"
                );
            }
            assert!(
                (l.0 - r.0).abs() > SIZE as f32 * 0.10,
                "detection {index}: the eyes are {} crop pixels apart, which is not a pair",
                (l.0 - r.0).abs()
            );
        }

        // The untrained three stay absent: the refiner replaces two points, not five.
        for detection in &detections {
            for (index, landmark) in detection.landmarks.iter().enumerate().skip(2) {
                assert!(landmark.is_none(), "landmark {index} should stay absent");
            }
        }
    }

    /// A box with no area is skipped rather than sampled, and reported as not refined.
    #[test]
    fn a_degenerate_box_is_not_refined() {
        let model = refiner_model();
        let Some(refiner) = EyeRefiner::load_from(&model) else {
            assert!(
                !strict_tests(),
                "FCS_STRICT_TESTS: the refiner did not load"
            );
            return;
        };
        let image =
            DynamicImage::ImageRgba8(RgbaImage::from_pixel(64, 64, Rgba([10, 20, 30, 255])));
        for bbox in [
            box_at(10.0, 10.0, 0.0, 30.0),
            box_at(10.0, 10.0, 30.0, 0.0),
            box_at(10.0, 10.0, 0.0, 0.0),
        ] {
            let mut detections = vec![detection_at(bbox)];
            assert_eq!(
                refiner.refine(&image, &mut detections),
                0,
                "a zero-area box must not be refined"
            );
            assert!(detections[0].landmarks[0].is_none());
        }
    }

    fn detection_at(bbox: BoundingBox) -> Detection {
        Detection {
            bbox,
            landmarks: [None; 5],
            score: 0.9,
        }
    }

    /// Sampling a point whose neighbours include column 0 and row 0.
    ///
    /// The companion test deliberately sits away from the origin so `x - left` cannot be
    /// confused with `x + left`. That leaves the lower bounds untested: `px < 0` widening to
    /// `px <= 0` throws away the *first* column, which only matters when a sample actually
    /// touches it. Both mutants survived until this existed.
    #[test]
    fn sampling_includes_the_first_row_and_column() {
        let image = gradient_image();
        let (fx, fy) = (0.4f32, 0.3f32);
        let got = sample_bilinear(&image, 5, 3, fx, fy);
        let corners = [
            ((0, 0), (1.0 - fx) * (1.0 - fy)),
            ((1, 0), fx * (1.0 - fy)),
            ((0, 1), (1.0 - fx) * fy),
            ((1, 1), fx * fy),
        ];
        for (channel, value) in got.iter().enumerate() {
            let expected: f32 = corners
                .iter()
                .map(|((cx, cy), weight)| weight * pixel_at(&image, *cx, *cy)[channel])
                .sum();
            assert!(
                (value - expected).abs() < 1e-3,
                "channel {channel}: got {value} expected {expected}"
            );
        }
    }

    /// Three values are one short of two points and must be refused rather than read past the
    /// end; four are exactly enough; a fifth is ignored. Between them they tell `<` from `<=`,
    /// `==` and `>`.
    #[test]
    fn eye_points_needs_four_values_and_scales_them_to_the_crop() {
        assert_eq!(eye_points(&[0.25, 0.5, 0.75]), None);
        let size = SIZE as f32;
        let expected = [(0.25 * size, 0.5 * size), (0.75 * size, 0.125 * size)];
        assert_eq!(eye_points(&[0.25, 0.5, 0.75, 0.125]), Some(expected));
        assert_eq!(eye_points(&[0.25, 0.5, 0.75, 0.125, 9.0]), Some(expected));
    }

    /// `EyeRefiner::load` resolves the model relative to the working directory, and under
    /// `cargo test` that is the crate root, which has no `models/`. So it returns `None` here
    /// whatever it does, and this only pins the negative half.
    ///
    /// The positive half needs `set_current_dir`, which is process-global and would make every
    /// other test in this binary order-dependent. So it lives in
    /// `tests/default_model_location.rs`, a binary holding that one test and nothing else.
    #[test]
    fn load_resolves_relative_to_the_working_directory() {
        // Documents the behaviour rather than asserting a path: from the crate root there is
        // no model, so `None` is correct.
        assert!(
            !std::path::Path::new(DEFAULT_MODEL).exists(),
            "this test's premise is that {DEFAULT_MODEL} is not resolvable from the crate root"
        );
        assert!(EyeRefiner::load().is_none());
    }
}
