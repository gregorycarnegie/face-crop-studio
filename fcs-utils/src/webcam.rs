//! Webcam capture utilities for real-time face detection.

use anyhow::{Context, Result, anyhow};
use image::{DynamicImage, ImageBuffer, Rgb};
use log::{debug, info, warn};
use nokhwa::{
    Camera,
    pixel_format::RgbFormat,
    query,
    utils::{
        CameraFormat, CameraIndex, FrameFormat, RequestedFormat, RequestedFormatType, Resolution,
    },
};

/// Represents a webcam device with capture capabilities.
pub struct WebcamCapture {
    camera: Camera,
    device_index: u32,
    resolution: (u32, u32),
}

impl WebcamCapture {
    /// Creates a new webcam capture instance with the default camera.
    ///
    /// # Arguments
    ///
    /// * `width` - Requested frame width (may be adjusted by driver)
    /// * `height` - Requested frame height (may be adjusted by driver)
    /// * `fps` - Requested frames per second
    pub fn new(width: u32, height: u32, fps: u32) -> Result<Self> {
        Self::with_device_index(0, width, height, fps)
    }

    /// Creates a new webcam capture instance with a specific camera device.
    ///
    /// # Arguments
    ///
    /// * `device_index` - Camera device index (0 for default camera)
    /// * `width` - Requested frame width (may be adjusted by driver)
    /// * `height` - Requested frame height (may be adjusted by driver)
    /// * `fps` - Requested frames per second
    pub fn with_device_index(device_index: u32, width: u32, height: u32, fps: u32) -> Result<Self> {
        let index = CameraIndex::Index(device_index);

        debug!(
            "Opening webcam device {} with requested resolution {}x{} @ {} fps",
            device_index, width, height, fps
        );

        // Ask for the format nearest what the caller wanted, not the camera's largest.
        //
        // This used to open with `AbsoluteHighestResolution` and then call `set_resolution`,
        // which does not take: a C920 asked for 640x480 delivered 1920x1080 regardless, so
        // `--webcam-width`/`--webcam-height` did nothing and every frame carried 6.5x the
        // pixels the caller asked for -- 4.3 ms of MJPEG decode and 3.6 ms of detection per
        // frame instead of a fraction of that (experiment 95).
        //
        // MJPEG because it is what USB cameras stream at their higher resolutions and what
        // nokhwa decodes through libjpeg-turbo. `Closest` scores the whole format, so a
        // camera without MJPEG still gets its nearest match rather than an error, and the
        // old behaviour remains as the fallback if the request cannot be satisfied at all.
        let wanted = CameraFormat::new(Resolution::new(width, height), FrameFormat::MJPEG, fps);
        let mut camera = match Camera::new(
            index.clone(),
            RequestedFormat::new::<RgbFormat>(RequestedFormatType::Closest(wanted)),
        ) {
            Ok(camera) => camera,
            Err(e) => {
                warn!(
                    "Could not open device {device_index} at {width}x{height} @ {fps} ({e});                      falling back to the camera's highest resolution."
                );
                Camera::new(
                    index,
                    RequestedFormat::new::<RgbFormat>(
                        RequestedFormatType::AbsoluteHighestResolution,
                    ),
                )
                .with_context(|| format!("Failed to open webcam device {device_index}"))?
            }
        };

        camera
            .open_stream()
            .context("Failed to open webcam stream")?;

        let actual_resolution = camera.resolution();
        info!(
            "Webcam device {} opened successfully: {}x{} @ {} fps",
            device_index,
            actual_resolution.width(),
            actual_resolution.height(),
            camera.frame_rate()
        );

        Ok(Self {
            camera,
            device_index,
            resolution: (actual_resolution.width(), actual_resolution.height()),
        })
    }

    /// Captures a single frame from the webcam.
    ///
    /// Returns a `DynamicImage` containing the captured frame.
    pub fn capture_frame(&mut self) -> Result<DynamicImage> {
        let frame = {
            let _g = crate::telemetry::timing_guard("fcs_utils::webcam_grab", log::Level::Trace);
            self.camera
                .frame()
                .context("Failed to capture webcam frame")?
        };

        let decoded = {
            let _g = crate::telemetry::timing_guard("fcs_utils::webcam_decode", log::Level::Trace);
            frame
                .decode_image::<RgbFormat>()
                .context("Failed to decode webcam frame")?
        };

        let (width, height) = self.resolution;

        // Convert nokhwa buffer to image::RgbImage
        let _g = crate::telemetry::timing_guard("fcs_utils::webcam_wrap", log::Level::Trace);
        let rgb_image: ImageBuffer<Rgb<u8>, Vec<u8>> =
            ImageBuffer::from_raw(width, height, decoded.to_vec())
                .ok_or_else(|| anyhow!("Failed to create image buffer from webcam data"))?;

        Ok(DynamicImage::ImageRgb8(rgb_image))
    }

    /// Returns the actual resolution being used by the webcam.
    pub fn resolution(&self) -> (u32, u32) {
        self.resolution
    }

    /// Returns the device index of this webcam.
    pub fn device_index(&self) -> u32 {
        self.device_index
    }

    /// Returns the current frame rate.
    pub fn frame_rate(&self) -> u32 {
        self.camera.frame_rate()
    }

    /// Stops the webcam stream and releases the device.
    pub fn stop(mut self) -> Result<()> {
        self.camera
            .stop_stream()
            .context("Failed to stop webcam stream")?;
        info!("Webcam device {} stopped", self.device_index);
        Ok(())
    }
}

impl Drop for WebcamCapture {
    fn drop(&mut self) {
        if let Err(e) = self.camera.stop_stream() {
            warn!("Failed to stop webcam stream in drop: {}", e);
        }
    }
}

/// Lists all available webcam devices on the system.
///
/// Returns a vector of tuples containing (device_index, device_name).
pub fn list_webcam_devices() -> Result<Vec<(u32, String)>> {
    let devices =
        query(nokhwa::utils::ApiBackend::Auto).context("Failed to query webcam devices")?;

    let result = devices
        .iter()
        .enumerate()
        .map(|(idx, info)| (idx as u32, info.human_name().to_string()))
        .collect();

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore] // Requires actual webcam hardware
    fn test_list_devices() {
        let devices = list_webcam_devices().expect("failed to list devices");
        println!("Available webcam devices:");
        for (idx, name) in devices {
            println!("  [{}] {}", idx, name);
        }
    }

    #[test]
    #[ignore] // Requires actual webcam hardware
    fn test_capture_single_frame() {
        let mut webcam = WebcamCapture::new(640, 480, 30).expect("failed to open webcam");
        let frame = webcam.capture_frame().expect("failed to capture frame");

        assert!(frame.width() > 0);
        assert!(frame.height() > 0);
        println!("Captured frame: {}x{}", frame.width(), frame.height());
    }
}
