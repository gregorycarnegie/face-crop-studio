//! The f32 tensor that flows between preprocessing, inference and decoding.
//!
//! Deliberately minimal. This carried `tract_onnx::prelude::Tensor` until the
//! `tract` backend was removed, and that type brought a whole runtime's worth of
//! generality — arbitrary dtypes, quantisation, views, lazy shapes — for a
//! pipeline that only ever moves densely packed f32 through four shapes.

use anyhow::Result;

/// A densely packed, row-major f32 tensor of arbitrary rank.
#[derive(Debug, Clone, PartialEq)]
pub struct Tensor {
    shape: Vec<usize>,
    data: Vec<f32>,
}

impl Tensor {
    /// Build a tensor, failing if the data length disagrees with the shape.
    pub fn from_shape(shape: &[usize], data: &[f32]) -> Result<Self> {
        let expected: usize = shape.iter().product();
        anyhow::ensure!(
            data.len() == expected,
            "shape {shape:?} needs {expected} elements, got {}",
            data.len()
        );
        Ok(Self {
            shape: shape.to_vec(),
            data: data.to_vec(),
        })
    }

    /// Build a tensor from an owned buffer, avoiding a copy.
    pub fn from_vec(shape: &[usize], data: Vec<f32>) -> Result<Self> {
        let expected: usize = shape.iter().product();
        anyhow::ensure!(
            data.len() == expected,
            "shape {shape:?} needs {expected} elements, got {}",
            data.len()
        );
        Ok(Self {
            shape: shape.to_vec(),
            data,
        })
    }

    /// Dimensions, outermost first.
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Elements in row-major order.
    pub fn as_slice(&self) -> &[f32] {
        &self.data
    }

    /// Total number of elements.
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Return whether the tensor contains no elements.
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Consume the tensor, yielding its buffer.
    pub fn into_vec(self) -> Vec<f32> {
        self.data
    }

    /// Iterate a rank-2 tensor row by row.
    ///
    /// Detection outputs are `[rows, cols]` and every consumer walks them a row
    /// at a time, so this replaces what an ndarray view was doing.
    pub fn rows(&self) -> Result<impl Iterator<Item = &[f32]>> {
        anyhow::ensure!(
            self.shape.len() == 2,
            "rows() needs a rank-2 tensor, got shape {:?}",
            self.shape
        );
        Ok(self.data.chunks_exact(self.shape[1].max(1)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_shape_rejects_a_length_that_disagrees() {
        let err = Tensor::from_shape(&[2, 3], &[0.0; 5]).expect_err("5 != 6");
        assert!(format!("{err}").contains("needs 6"), "{err}");
    }

    #[test]
    fn rows_walks_a_rank_two_tensor() {
        let t = Tensor::from_shape(&[2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).expect("valid");
        let rows: Vec<_> = t.rows().expect("rank 2").collect();
        assert_eq!(rows, vec![&[1.0, 2.0, 3.0][..], &[4.0, 5.0, 6.0][..]]);
    }

    #[test]
    fn rows_refuses_a_rank_that_is_not_two() {
        let t = Tensor::from_shape(&[2, 2, 2], &[0.0; 8]).expect("valid");
        assert!(t.rows().is_err(), "rank 3 must not be walked as rows");
    }

    #[test]
    fn from_vec_preserves_the_buffer() {
        let t = Tensor::from_vec(&[4], vec![1.0, 2.0, 3.0, 4.0]).expect("valid");
        assert_eq!(t.shape(), &[4]);
        assert_eq!(t.as_slice(), &[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(t.len(), 4);
        assert!(!t.is_empty());
        assert!(
            Tensor::from_vec(&[0, 15], Vec::new())
                .expect("empty tensor")
                .is_empty()
        );
    }
}
