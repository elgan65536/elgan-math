use std::ops::Mul;

use crate::{columnvec::ColumnVec, matrix::Matrix};

#[derive(Clone, Copy, Debug, PartialEq)]
/// an Affine transformation which combines a Matrix transform with translation by a vector.
pub struct Affine<const A: usize> {
    pub transform: Matrix<A, A>,
    pub translate: ColumnVec<A>,
}

impl<const A: usize> Mul<ColumnVec<A>> for Affine<A> {
    type Output = ColumnVec<A>;

    /// applies the affine transformation
    fn mul(self, rhs: ColumnVec<A>) -> Self::Output {
        self.transform * rhs + self.translate
    }
}

impl<const A: usize> Mul for Affine<A> {
    type Output = Self;

    /// returns the transform that is equivalent to first applying rhs then self.
    fn mul(self, rhs: Self) -> Self::Output {
        Self::new(
            self.transform * rhs.transform,
            self.transform * rhs.translate + self.translate,
        )
    }
}

impl<const A: usize> Affine<A> {
    /// constructs a new [`Affine`] transformation which first applies the Matrix transform then the vector translation.
    pub fn new(transform: Matrix<A, A>, translate: ColumnVec<A>) -> Self {
        Self {
            transform,
            translate,
        }
    }

    /// constructs a new [`Affine`] transformation which first applies the vector translation then the Matrix transform
    pub fn new_reverse(translate: ColumnVec<A>, transform: Matrix<A, A>) -> Self {
        Self {
            transform,
            translate: transform * translate,
        }
    }

    /// returns an affine transformation which undoes the transformation of self.
    pub fn inverse(self) -> Option<Self> {
        Some(Self::new_reverse(
            -self.translate,
            self.transform.inverse()?,
        ))
    }
}
