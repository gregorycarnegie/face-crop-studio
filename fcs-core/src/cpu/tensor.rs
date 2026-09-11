//! NCHW f32 tensors for the CPU inference graph.
//!
//! Deliberately minimal: YuNet's graph only ever needs 4-D f32 activations, so
//! there is no dtype dispatch, no strides and no broadcasting. Anything more
//! general would be machinery with no caller.

use anyhow::Result;

/// A 4-D NCHW tensor of f32, densely packed.
#[derive(Debug, Clone, PartialEq)]
pub struct Tensor {
    batch: usize,
    channels: usize,
    height: usize,
    width: usize,
    data: Vec<f32>,
}

impl Tensor {
    /// Wrap existing data. Fails if the length disagrees with the shape.
    pub fn new(
        batch: usize,
        channels: usize,
        height: usize,
        width: usize,
        data: Vec<f32>,
    ) -> Result<Self> {
        let expected = batch * channels * height * width;
        anyhow::ensure!(
            data.len() == expected,
            "tensor {batch}x{channels}x{height}x{width} needs {expected} elements, got {}",
            data.len()
        );
        Ok(Self {
            batch,
            channels,
            height,
            width,
            data,
        })
    }

    /// A zero-filled tensor.
    pub fn zeros(batch: usize, channels: usize, height: usize, width: usize) -> Self {
        Self {
            batch,
            channels,
            height,
            width,
            data: vec![0.0; batch * channels * height * width],
        }
    }

    /// Return the number of batch items.
    pub fn batch(&self) -> usize {
        self.batch
    }

    /// Return the number of channels per batch item.
    pub fn channels(&self) -> usize {
        self.channels
    }

    /// Return the number of rows in each channel plane.
    pub fn height(&self) -> usize {
        self.height
    }

    /// Return the number of columns in each channel plane.
    pub fn width(&self) -> usize {
        self.width
    }

    /// Elements in one channel plane.
    pub fn plane(&self) -> usize {
        self.height * self.width
    }

    /// Return dimensions in `[batch, channels, height, width]` order.
    pub fn dims(&self) -> [usize; 4] {
        [self.batch, self.channels, self.height, self.width]
    }

    /// Borrow the contiguous NCHW data buffer.
    pub fn data(&self) -> &[f32] {
        &self.data
    }

    /// Mutably borrow the contiguous NCHW data buffer without changing its shape.
    pub fn data_mut(&mut self) -> &mut [f32] {
        &mut self.data
    }

    /// Consume the tensor and return its contiguous NCHW data buffer.
    pub fn into_data(self) -> Vec<f32> {
        self.data
    }

    /// One channel plane of one batch item.
    pub fn plane_slice(&self, batch: usize, channel: usize) -> &[f32] {
        let plane = self.plane();
        let start = (batch * self.channels + channel) * plane;
        &self.data[start..start + plane]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_rejects_a_length_that_disagrees_with_the_shape() {
        let err = Tensor::new(1, 2, 3, 4, vec![0.0; 23]).expect_err("23 != 24");
        assert!(format!("{err}").contains("needs 24"), "{err}");
    }

    #[test]
    fn plane_slice_addresses_the_requested_channel() {
        // Two channels of a 2x2 plane, each filled with its own channel index.
        let data = vec![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0];
        let t = Tensor::new(1, 2, 2, 2, data).expect("valid");
        assert_eq!(t.plane_slice(0, 0), &[0.0, 0.0, 0.0, 0.0]);
        assert_eq!(t.plane_slice(0, 1), &[1.0, 1.0, 1.0, 1.0]);
        assert_eq!(t.plane(), 4);
        assert_eq!(t.dims(), [1, 2, 2, 2]);
    }

    #[test]
    fn plane_slice_offsets_by_batch_as_well_as_channel() {
        // Batch 1 must not alias batch 0: 2 batches x 1 channel x 1x2 plane.
        let t = Tensor::new(2, 1, 1, 2, vec![1.0, 2.0, 3.0, 4.0]).expect("valid");
        assert_eq!(t.plane_slice(0, 0), &[1.0, 2.0]);
        assert_eq!(t.plane_slice(1, 0), &[3.0, 4.0]);
    }
}
