use std::ops::Mul;

use crate::{columnvec::ColumnVec, matrix::Matrix};

#[derive(Clone, Copy, Debug, PartialEq)]
/// an Affine transformation which combines a linear Matrix transform with translation by a vector.
pub struct Affine<const HEIGHT: usize, const WIDTH: usize> {
    transform: Matrix<HEIGHT, WIDTH>,
    translate: ColumnVec<HEIGHT>,
}

impl<const HEIGHT: usize, const WIDTH: usize> From<Matrix<HEIGHT, WIDTH>>
    for Affine<HEIGHT, WIDTH>
{
    fn from(value: Matrix<HEIGHT, WIDTH>) -> Self {
        Self {
            transform: value,
            translate: ColumnVec::zero(),
        }
    }
}

impl<const HEIGHT: usize, const WIDTH: usize> From<ColumnVec<HEIGHT>> for Affine<HEIGHT, WIDTH> {
    fn from(value: ColumnVec<HEIGHT>) -> Self {
        Self {
            transform: Matrix::zero(),
            translate: value,
        }
    }
}

impl<const HEIGHT: usize, const WIDTH: usize> Mul<ColumnVec<WIDTH>> for Affine<HEIGHT, WIDTH> {
    type Output = ColumnVec<HEIGHT>;

    /// applies the affine transformation: multiplies by the transform matrix and then adds the translation vector.
    fn mul(self, rhs: ColumnVec<WIDTH>) -> Self::Output {
        self.transform * rhs + self.translate
    }
}

impl<const HEIGHT: usize, const WIDTH: usize, const NEW_WIDTH: usize> Mul<Affine<WIDTH, NEW_WIDTH>>
    for Affine<HEIGHT, WIDTH>
{
    type Output = Affine<HEIGHT, NEW_WIDTH>;

    /// returns the transform that is equivalent to first applying rhs then self.
    fn mul(self, rhs: Affine<WIDTH, NEW_WIDTH>) -> Self::Output {
        Affine::new(
            self.transform * rhs.transform,
            self.transform * rhs.translate + self.translate,
        )
    }
}

impl<const HEIGHT: usize, const WIDTH: usize> Affine<HEIGHT, WIDTH> {
    /// constructs a new [`Affine`] transformation which first applies the Matrix transform then the vector translation.
    pub fn new(transform: Matrix<HEIGHT, WIDTH>, translate: ColumnVec<HEIGHT>) -> Self {
        Self {
            transform,
            translate,
        }
    }

    /// constructs a new [`Affine`] transformation which first applies the vector translation then the Matrix transform
    pub fn new_reverse(translate: ColumnVec<WIDTH>, transform: Matrix<HEIGHT, WIDTH>) -> Self {
        Self {
            transform,
            translate: transform * translate,
        }
    }
}

impl<const WIDTH: usize> Affine<WIDTH, WIDTH> {
    /// returns an affine transformation which undoes the transformation of self.
    /// returns `None` if no such transformation exists.
    pub fn inverse(self) -> Option<Self> {
        Some(Self::new_reverse(
            -self.translate,
            self.transform.inverse()?,
        ))
    }
}
