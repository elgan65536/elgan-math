use std::ops::{Add, Div, Index, IndexMut, Mul, Neg, Sub};

use rand::Rng;

use crate::{complex::Complex, matrix::Matrix};

/// a fixed-size column vector which can be of any size.
#[derive(Debug, PartialEq, Clone, Copy)]
pub struct ColumnVec<const A: usize>(pub [f64; A]);

impl<const A: usize> From<ColumnVec<A>> for [f64; A] {
    fn from(vec: ColumnVec<A>) -> Self {
        vec.0
    }
}

impl<const A: usize> From<[f64; A]> for ColumnVec<A> {
    fn from(arr: [f64; A]) -> Self {
        ColumnVec(arr)
    }
}

impl From<Complex> for ColumnVec<2> {
    fn from(Complex { real, im }: Complex) -> Self {
        Self([real, im])
    }
}

impl From<ColumnVec<2>> for Complex {
    fn from(ColumnVec([a, b]): ColumnVec<2>) -> Self {
        Complex::new(a, b)
    }
}

impl<const A: usize> Default for ColumnVec<A> {
    fn default() -> Self {
        Self([0.; A])
    }
}

impl<const A: usize> Index<usize> for ColumnVec<A> {
    type Output = f64;

    fn index(&self, index: usize) -> &Self::Output {
        &self.0[index]
    }
}

impl<const A: usize> IndexMut<usize> for ColumnVec<A> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.0[index]
    }
}

impl<const A: usize> Add for ColumnVec<A> {
    type Output = ColumnVec<A>;

    fn add(self, rhs: Self) -> Self::Output {
        let mut result = self;
        for i in 0..A {
            result[i] += rhs[i];
        }
        result
    }
}

impl<const A: usize> Sub for ColumnVec<A> {
    type Output = ColumnVec<A>;

    fn sub(self, rhs: Self) -> Self::Output {
        let mut result = self;
        for i in 0..A {
            result[i] -= rhs[i];
        }
        result
    }
}

impl<const A: usize, const B: usize> Mul<ColumnVec<B>> for Matrix<A, B> {
    type Output = ColumnVec<A>;

    fn mul(self, rhs: ColumnVec<B>) -> Self::Output {
        let mut result = ColumnVec([0.; A]);
        for i in 0..A {
            for k in 0..B {
                result[i] += self[i][k] * rhs[k];
            }
        }
        result
    }
}

impl<const A: usize> Mul for ColumnVec<A> {
    type Output = f64;

    /// computes the dot product.
    fn mul(self, rhs: ColumnVec<A>) -> Self::Output {
        let mut result = 0.0;
        for i in 0..A {
            result += self[i] * rhs[i];
        }
        result
    }
}

impl<const A: usize> Mul<f64> for ColumnVec<A> {
    type Output = ColumnVec<A>;

    fn mul(self, rhs: f64) -> Self::Output {
        ColumnVec(self.0.map(|x| x * rhs))
    }
}

impl<const A: usize> Mul<ColumnVec<A>> for f64 {
    type Output = ColumnVec<A>;

    fn mul(self, rhs: ColumnVec<A>) -> Self::Output {
        ColumnVec(rhs.0.map(|x| x * self))
    }
}

impl<const A: usize> Div<f64> for ColumnVec<A> {
    type Output = ColumnVec<A>;

    fn div(self, rhs: f64) -> Self::Output {
        ColumnVec(self.0.map(|x| x / rhs))
    }
}

impl<const A: usize> Neg for ColumnVec<A> {
    type Output = Self;

    fn neg(self) -> Self::Output {
        -1. * self
    }
}

impl<const A: usize> ColumnVec<A> {
    /// returns a vector where all entries are zero.
    pub fn zero() -> Self {
        Self([0.; A])
    }

    /// returns the length of the vector, also known as magnitude or norm.
    pub fn length(self) -> f64 {
        self.0.map(|x| x * x).iter().sum::<f64>().sqrt()
    }

    /// computes the angle between two vectors. This will always be positive.
    pub fn angle_between(self, rhs: Self) -> f64 {
        ((self * rhs) / (self.length() * rhs.length())).acos()
    }

    /// computes a vector with the same direction as the original but with length 1.
    pub fn normalized(self) -> Self {
        self / self.length()
    }

    /// multiplies two vectors by multiplying each component individually.
    pub fn component_mul(self, rhs: Self) -> Self {
        let mut result = self;
        for i in 0..A {
            result[i] *= rhs[i];
        }
        result
    }

    /// computes the dot product. The `*` operator also does this.
    pub fn dot(self, rhs: Self) -> f64 {
        self * rhs
    }

    /// divides two vectors by dividing each component individually.
    pub fn component_div(self, rhs: Self) -> Self {
        let mut result = self;
        for i in 0..A {
            result[i] /= rhs[i];
        }
        result
    }

    /// constructs a vector whose components are all uniformly selected from the specified range.
    pub fn random_box(min: f64, max: f64) -> Self {
        ColumnVec([0.; A].map(|_| rand::thread_rng().gen_range(min..max)))
    }

    /// constructs a vector with random direction and length of 1.
    pub fn random_unit() -> Self {
        ColumnVec(
            [0.; A].map(|_| rand::Rng::sample(&mut rand::thread_rng(), rand_distr::StandardNormal)),
        )
        .normalized()
    }

    /// constructs a vector with random direction and length of less than or equal to 1.
    /// Uniformly selects from all possible points.
    pub fn random_inside_sphere() -> Self {
        Self::random_unit() * rand::random::<f64>().powf(1. / A as f64)
    }

    /// constructs a vector with random direction and length of less than or equal to 1 that forms an acute or right angle with the specified direction.
    /// Uniformly selects from all possible points.
    pub fn random_in_hemisphere(vec: ColumnVec<A>) -> Self {
        let result = Self::random_inside_sphere();
        if result * vec < 0. {
            -result
        } else {
            result
        }
    }

    /// clamps each component of the vector to be between min and max.
    pub fn component_clamp(self, min: f64, max: f64) -> Self {
        ColumnVec(self.0.map(|x| x.clamp(min, max)))
    }

    /// returns true if the two vectors are equivalent within a floating-point rounding error. useful for testing.
    pub fn close_enough(self, rhs: Self) -> bool {
        for i in 0..A {
            if (self[i] - rhs[i]).abs() > f64::EPSILON {
                return false;
            }
        }
        true
    }

    pub fn outer_product<const B: usize>(self, rhs: ColumnVec<B>) -> Matrix<A, B> {
        let mut result = Matrix::zero();
        for i in 0..A {
            for j in 0..B {
                result[i][j] = self[i] * rhs[j];
            }
        }
        result
    }
}

impl ColumnVec<2> {
    /// computes the angle between the given vector and the positive x-axis, in radians, from -pi to pi.
    pub fn angle(self) -> f64 {
        self[1].atan2(self[0])
    }

    /// computes signed magnitude of the cross product.
    pub fn cross(self, rhs: Self) -> f64 {
        self[0] * rhs[1] - self[1] * rhs[0]
    }
}

impl ColumnVec<3> {
    /// computes the cross product.
    pub fn cross(self, rhs: Self) -> Self {
        ColumnVec([
            self[1] * rhs[2] - self[2] * rhs[1],
            self[2] * rhs[0] - self[0] * rhs[2],
            self[0] * rhs[1] - self[1] * rhs[0],
        ])
    }
}
