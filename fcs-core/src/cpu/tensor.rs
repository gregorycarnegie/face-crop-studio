//! NCHW f32 tensors for the CPU inference graph.
//!
//! Deliberately minimal: the graph only ever needs 4-D f32 activations, so
//! there is no dtype dispatch, no strides and no broadcasting. Anything more
//! general would be machinery with no caller.

use super::layout::{checked_nchw, plane_range};
use super::size::checked_allocation;
use anyhow::{Context, Result};

/// A 4-D NCHW tensor of f32, densely packed.
#[derive(Debug, Clone, PartialEq)]
pub struct Tensor {
    batch: usize,
    channels: usize,
    height: usize,
    width: usize,
    plane: usize,
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
        let (plane, expected) = checked_nchw(batch, channels, height, width)
            .context("CPU tensor layout overflows usize")?;
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
            plane,
            data,
        })
    }

    /// A zero-filled tensor.
    ///
    /// # Panics
    /// Panics if the layout overflows usize or its byte size exceeds isize::MAX.
    pub fn zeros(batch: usize, channels: usize, height: usize, width: usize) -> Self {
        let (plane, elements) = checked_nchw(batch, channels, height, width)
            .expect("CPU tensor layout overflows usize");
        checked_allocation(elements, 4).expect("CPU tensor allocation exceeds isize::MAX");
        Self {
            batch,
            channels,
            height,
            width,
            plane,
            data: vec![0.0; elements],
        }
    }

    /// Elements in one channel plane.
    pub fn plane(&self) -> usize {
        self.plane
    }

    /// Return dimensions in `[batch, channels, height, width]` order.
    pub fn dims(&self) -> [usize; 4] {
        [self.batch, self.channels, self.height, self.width]
    }

    /// Borrow the contiguous NCHW data buffer.
    pub fn data(&self) -> &[f32] {
        &self.data
    }

    /// Consume the tensor and return its contiguous NCHW data buffer.
    pub fn into_data(self) -> Vec<f32> {
        self.data
    }

    /// One channel plane of one batch item.
    ///
    /// # Panics
    /// Panics if either index is outside the tensor's dimensions.
    pub fn plane_slice(&self, batch: usize, channel: usize) -> &[f32] {
        let (start, end) = plane_range(
            batch,
            channel,
            self.batch,
            self.channels,
            self.plane,
            self.data.len(),
        )
        .expect("CPU tensor plane index out of bounds");
        &self.data[start..end]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_overflow_even_when_an_outer_dimension_is_zero() {
        assert!(Tensor::new(usize::MAX, usize::MAX, 1, 1, vec![0.0]).is_err());
        assert!(Tensor::new(0, 1, usize::MAX, 2, vec![]).is_err());
        assert!(Tensor::zeros(2, 3, 0, 5).plane_slice(1, 2).is_empty());
    }

    #[test]
    #[should_panic(expected = "CPU tensor plane index out of bounds")]
    fn invalid_channel_cannot_alias_the_next_batch() {
        Tensor::zeros(2, 3, 1, 1).plane_slice(0, 3);
    }

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
        // Batch 1 must not alias batch 0, and with 3 channels a batch offset of
        // batch / channels would be 0.
        let t = Tensor::new(2, 3, 1, 2, (0..12).map(|i| i as f32).collect()).expect("valid");
        assert_eq!(t.plane_slice(0, 0), &[0.0, 1.0]);
        assert_eq!(t.plane_slice(1, 2), &[10.0, 11.0]); // (1 * 3 + 2) * 2 = 10
    }

    #[test]
    fn into_data_hands_back_the_buffer() {
        let t = Tensor::new(1, 1, 1, 2, vec![3.0, 7.0]).expect("tensor");
        assert_eq!(t.into_data(), vec![3.0, 7.0]);
    }
}
