use derive_more::*;
use std::ops::{Add, Div, Mul, Neg, Sub};

use rand::Rng;

use crate::{complex::Complex, matrix::Matrix, FloatArithmetic};

/// a fixed-size column vector of the specified size.
#[derive(Debug, PartialEq, Clone, Copy, From, Into, Index, IndexMut, IntoIterator, Display)]
#[display(fmt = "{:?}", _0)]
pub struct ColumnVec<const LENGTH: usize>(pub(crate) [f64; LENGTH]);

impl From<Complex> for ColumnVec<2> {
    fn from(complex: Complex) -> Self {
        Self([complex.real(), complex.im()])
    }
}

impl From<ColumnVec<2>> for Complex {
    fn from(ColumnVec([a, b]): ColumnVec<2>) -> Self {
        Complex::new(a, b)
    }
}

impl<const LENGTH: usize> Default for ColumnVec<LENGTH> {
    fn default() -> Self {
        Self([0.; LENGTH])
    }
}

impl<const LENGTH: usize> Add for ColumnVec<LENGTH> {
    type Output = ColumnVec<LENGTH>;

    fn add(mut self, rhs: Self) -> Self::Output {
        for i in 0..LENGTH {
            self[i] += rhs[i];
        }
        self
    }
}

impl<const LENGTH: usize> Sub for ColumnVec<LENGTH> {
    type Output = ColumnVec<LENGTH>;

    fn sub(mut self, rhs: Self) -> Self::Output {
        for i in 0..LENGTH {
            self[i] -= rhs[i];
        }
        self
    }
}

impl<const LENGTH: usize, const WIDTH: usize> Mul<ColumnVec<WIDTH>> for Matrix<LENGTH, WIDTH> {
    type Output = ColumnVec<LENGTH>;

    fn mul(self, rhs: ColumnVec<WIDTH>) -> Self::Output {
        let mut result = ColumnVec([0.; LENGTH]);
        for i in 0..LENGTH {
            for k in 0..WIDTH {
                result[i] += self[i][k] * rhs[k];
            }
        }
        result
    }
}

impl<const LENGTH: usize> Mul for ColumnVec<LENGTH> {
    type Output = f64;

    /// computes the dot product.
    fn mul(self, rhs: ColumnVec<LENGTH>) -> Self::Output {
        self.dot(rhs)
    }
}

impl<const LENGTH: usize> Mul<f64> for ColumnVec<LENGTH> {
    type Output = ColumnVec<LENGTH>;

    fn mul(self, rhs: f64) -> Self::Output {
        self.map(|x| x * rhs)
    }
}

impl<const LENGTH: usize> Mul<ColumnVec<LENGTH>> for f64 {
    type Output = ColumnVec<LENGTH>;

    fn mul(self, rhs: ColumnVec<LENGTH>) -> Self::Output {
        rhs.map(|x| x * self)
    }
}

impl<const LENGTH: usize> Div<f64> for ColumnVec<LENGTH> {
    type Output = ColumnVec<LENGTH>;

    fn div(self, rhs: f64) -> Self::Output {
        self.map(|x| x / rhs)
    }
}

impl<const LENGTH: usize> Neg for ColumnVec<LENGTH> {
    type Output = Self;

    fn neg(self) -> Self::Output {
        self.map(|x| -x)
    }
}

impl<const LENGTH: usize> ColumnVec<LENGTH> {
    /// creates a new vector from the given array.
    ///
    /// # Examples
    ///
    /// ```
    /// use elgan_math::ColumnVec;
    /// let vec = ColumnVec::new([1.0, 2.0, 3.0]);
    /// ```
    pub const fn new(arr: [f64; LENGTH]) -> Self {
        Self(arr)
    }

    /// returns a vector where all entries are zero.
    ///
    /// # Examples
    /// ```
    /// use elgan_math::ColumnVec;
    ///
    /// assert_eq!(ColumnVec::new([0.0, 0.0, 0.0]), ColumnVec::zero())
    /// ```
    pub const fn zero() -> Self {
        Self::all_same(0.)
    }

    /// returns a vector where all of its entries are the specified number.
    ///
    /// # Examples
    /// ```
    /// use elgan_math::ColumnVec;
    ///
    /// assert_eq!(ColumnVec::new([3.5, 3.5, 3.5]), ColumnVec::all_same(3.5))
    /// ```
    pub const fn all_same(num: f64) -> Self {
        Self([num; LENGTH])
    }

    /// computes the sum of all elements in the vector.
    ///
    /// # Examples
    /// ```
    /// use elgan_math::ColumnVec;
    ///
    /// let vec = ColumnVec::new([1.0, 2.0, 3.0]);
    /// assert_eq!(vec.sum_elements(), 6.0)
    /// ```
    pub fn sum_elements(self) -> f64 {
        self.0.iter().sum()
    }

    /// returns the length of the vector, also known as magnitude or norm2.
    ///
    /// # Examples
    /// ```
    /// use elgan_math::ColumnVec;
    ///
    /// let vec = ColumnVec::new([3.0, 4.0]);
    /// assert_eq!(vec.length(), 5.0)
    /// ```
    pub fn length(self) -> f64 {
        self.0.map(|x| x * x).iter().sum::<f64>().sqrt()
    }

    /// computes the `n`-th norm of the vector.
    ///
    /// # Examples
    /// ```
    /// use elgan_math::ColumnVec;
    ///
    /// let vec = ColumnVec::new([-3.0, 4.0]);
    /// assert_eq!(vec.norm(1.0), 7.0);
    /// assert_eq!(vec.norm(2.0), 5.0);
    /// assert_eq!(vec.norm(f64::INFINITY), 4.0)
    /// ```
    pub fn norm(self, n: f64) -> f64 {
        if n == f64::INFINITY {
            *self
                .map(f64::abs)
                .0
                .iter()
                .max_by(|a, b| a.total_cmp(b))
                .unwrap_or(&0.)
        } else {
            self.map(|x| x.powf(n).abs()).sum_elements().powf(n.recip())
        }
    }

    /// computes the angle between two vectors in radians. This will always be positive.
    ///
    /// # Examples
    /// ```
    /// use elgan_math::{ColumnVec, FloatArithmetic};
    ///
    /// let vec1 = ColumnVec::new([1.0, 0.0]);
    /// let vec2 = ColumnVec::new([1.0, 1.0]);
    /// assert!(vec1.angle_between(vec2).close_to(core::f64::consts::FRAC_PI_4))
    /// ```
    pub fn angle_between(self, rhs: Self) -> f64 {
        ((self * rhs) / (self.length() * rhs.length())).acos()
    }

    /// computes a vector with the same direction as the original but with length 1.
    ///
    /// # Examples
    /// ```
    /// use elgan_math::ColumnVec;
    ///
    /// let vec = ColumnVec::new([3.0, 4.0]);
    /// let vec_norm = ColumnVec::new([0.6, 0.8]);
    /// assert_eq!(vec.normalized(), vec_norm)
    /// ```
    pub fn normalized(self) -> Self {
        self / self.length()
    }

    /// multiplies two vectors by multiplying each component individually.
    ///
    /// # Examples
    /// ```
    /// use elgan_math::ColumnVec;
    ///
    /// let vec1 = ColumnVec::new([1.0, 2.0, 3.0]);
    /// let vec2 = ColumnVec::new([3.0, 4.0, 5.0]);
    /// assert_eq!(vec1.component_mul(vec2), ColumnVec::new([3.0, 8.0, 15.0]))
    /// ```
    pub fn component_mul(mut self, rhs: Self) -> Self {
        for i in 0..LENGTH {
            self[i] *= rhs[i];
        }
        self
    }

    /// computes the dot product.
    ///
    /// # Examples
    /// ```
    /// use elgan_math::ColumnVec;
    ///
    /// let vec1 = ColumnVec::new([1.0, 2.0, 3.0]);
    /// let vec2 = ColumnVec::new([3.0, 4.0, 5.0]);
    /// assert_eq!(vec1.dot(vec2), 26.0)
    /// ```
    pub fn dot(self, rhs: Self) -> f64 {
        let mut result = 0.0;
        for i in 0..LENGTH {
            result += self[i] * rhs[i];
        }
        result
    }

    /// divides two vectors by dividing each component individually.
    ///
    /// # Examples
    /// ```
    /// use elgan_math::ColumnVec;
    ///
    /// let vec1 = ColumnVec::new([3.0, 8.0, 15.0]);
    /// let vec2 = ColumnVec::new([3.0, 4.0, 5.0]);
    /// assert_eq!(vec1.component_div(vec2), ColumnVec::new([1.0, 2.0, 3.0]))
    /// ```
    pub fn component_div(mut self, rhs: Self) -> Self {
        for i in 0..LENGTH {
            self[i] /= rhs[i];
        }
        self
    }

    /// constructs a vector whose components are all uniformly selected from the specified range.
    pub fn random_box(min: f64, max: f64) -> Self {
        Self::default().map(|_| rand::thread_rng().gen_range(min..max))
    }

    /// generates a random variable following a normal distribution using Box-Muller Transform; for use in random_unit
    fn random_normal() -> f64 {
        (-2. * rand::random::<f64>().ln()).sqrt()
            * (std::f64::consts::TAU * rand::random::<f64>()).cos()
    }

    /// constructs a vector with random direction and length of 1.
    pub fn random_unit() -> Self {
        Self::default().map(|_| Self::random_normal()).normalized()
    }

    /// constructs a vector with random direction and length of less than or equal to 1.
    /// Uniformly selects from all possible points.
    pub fn random_inside_sphere() -> Self {
        Self::random_unit() * rand::random::<f64>().powf(1. / LENGTH as f64)
    }

    /// constructs a vector with random direction and length of less than or equal to 1
    /// that forms an acute or right angle with the specified direction.
    /// Uniformly selects from all possible points.
    pub fn random_in_hemisphere(vec: ColumnVec<LENGTH>) -> Self {
        let result = Self::random_inside_sphere();
        if result * vec < 0. {
            -result
        } else {
            result
        }
    }

    /// clamps the vector so its length is between min and max.
    pub fn clamp_length(self, min: f64, max: f64) -> Self {
        self.normalized() * self.length().clamp(min, max)
    }

    /// clamps each component of the vector to be between min and max.
    pub fn component_clamp(self, min: f64, max: f64) -> Self {
        self.map(|x| x.clamp(min, max))
    }

    /// returns true if the two vectors are equivalent within a floating-point rounding error.
    /// Use this instead of `==` if you want to accomodate for rounding errors.
    ///
    /// # Examples
    /// ```
    /// use elgan_math::ColumnVec;
    ///
    /// let vec1 = ColumnVec::new([1.0, 1.0]);
    /// let vec2 = ColumnVec::new([1.0000000000000001, 0.9999999999999999]);
    /// assert!(vec1.close_to(vec2))
    /// ```
    pub fn close_to(self, rhs: Self) -> bool {
        for i in 0..LENGTH {
            if !self[i].close_to(rhs[i]) {
                return false;
            }
        }
        true
    }

    /// computes the outer product of two vectors.
    ///
    /// # Examples
    ///
    /// ```
    /// use elgan_math::{ColumnVec, Matrix};
    /// let vec1 = ColumnVec::new([1.0, 2.0]);
    /// let vec2 = ColumnVec::new([3.0, 4.0]);
    /// let matrix = Matrix::new([[3.0, 4.0],[6.0, 8.0]]);
    /// assert_eq!(vec1.outer_product(vec2), matrix);
    /// ```
    pub fn outer_product<const WIDTH: usize>(self, rhs: ColumnVec<WIDTH>) -> Matrix<LENGTH, WIDTH> {
        let mut result = Matrix::zero();
        for i in 0..LENGTH {
            for j in 0..WIDTH {
                result[i][j] = self[i] * rhs[j];
            }
        }
        result
    }

    /// creates a new column vector where each element is the funciton applied to the original element.
    ///
    /// # Examples
    ///
    /// ```
    /// use elgan_math::ColumnVec;
    /// let vec1 = ColumnVec::new([2.0, 3.0, 4.0]);
    /// let vec2 = ColumnVec::new([5.0, 7.5, 10.0]);
    /// assert_eq!(vec1.map(|x| 2.5 * x), vec2);
    /// ```
    pub fn map<F>(mut self, func: F) -> Self
    where
        F: Fn(f64) -> f64,
    {
        for i in 0..LENGTH {
            self[i] = func(self[i]);
        }
        self
    }

    /// converts the vector to a vector of a different size.
    /// If the new vector is smaller, the last elements will be removed.
    /// If the new vector is larger, the new elements will be zero.
    ///
    /// # Examples
    ///
    /// ```
    /// use elgan_math::ColumnVec;
    ///
    /// let vec = ColumnVec::new([1.0, 2.0, 3.0]);
    /// // the generic parameters can be ommitted in this case, but they are included for clarity
    /// assert_eq!(vec.resize::<2>(), ColumnVec::new([1.0, 2.0]));
    /// assert_eq!(vec.resize::<4>(), ColumnVec::new([1.0, 2.0, 3.0, 0.0]))
    /// ```
    pub fn resize<const NEW_LENGTH: usize>(self) -> ColumnVec<NEW_LENGTH> {
        let mut result = ColumnVec::<NEW_LENGTH>::default();
        for i in 0..NEW_LENGTH {
            result[i] = if i >= LENGTH { 0. } else { self[i] }
        }
        result
    }
}

impl ColumnVec<2> {
    /// computes the angle between the given vector and the positive x-axis, in radians, from -pi to pi.
    /// Note that, unlike `angle_between`, this will be negative for vectors with negative y-component.
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

    /// rotates counterclockwise around the specified axis by the specified angle.
    /// the axis should be a normalized vector.
    pub fn rotate_around(self, axis: Self, angle: f64) -> Self {
        self * angle.cos()
            + (axis.cross(self)) * angle.sin()
            + axis * (axis.dot(self)) * (1. - angle.cos())
    }
}

impl ColumnVec<7> {
    /// computes the cross product.
    pub fn cross(self, rhs: Self) -> Self {
        ColumnVec([
            self[1] * rhs[3] - self[3] * rhs[1] + self[2] * rhs[6] - self[6] * rhs[2]
                + self[4] * rhs[5]
                - self[5] * rhs[4],
            self[2] * rhs[4] - self[4] * rhs[2] + self[0] * rhs[3] - self[3] * rhs[0]
                + self[5] * rhs[6]
                - self[6] * rhs[5],
            self[3] * rhs[5] - self[5] * rhs[3] + self[1] * rhs[4] - self[4] * rhs[1]
                + self[0] * rhs[6]
                - self[6] * rhs[0],
            self[4] * rhs[6] - self[6] * rhs[4] + self[2] * rhs[5] - self[5] * rhs[2]
                + self[0] * rhs[1]
                - self[1] * rhs[0],
            self[0] * rhs[5] - self[5] * rhs[0] + self[3] * rhs[6] - self[6] * rhs[3]
                + self[1] * rhs[2]
                - self[2] * rhs[1],
            self[1] * rhs[6] - self[6] * rhs[1] + self[0] * rhs[4] - self[4] * rhs[0]
                + self[2] * rhs[3]
                - self[3] * rhs[2],
            self[0] * rhs[2] - self[2] * rhs[0] + self[1] * rhs[5] - self[5] * rhs[1]
                + self[3] * rhs[4]
                - self[4] * rhs[3],
        ])
    }
}
