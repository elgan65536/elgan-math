use derive_more::*;
use std::ops::{Add, Div, Mul, Neg, Sub};

use crate::{complex::Complex, linalg::ColumnVec, FloatArithmetic};

/// a fixed-size matrix with the specified number of rows and columns.
#[derive(Debug, PartialEq, Clone, Copy, From, Into, Index, IndexMut)]
pub struct Matrix<const HEIGHT: usize, const WIDTH: usize>(pub(crate) [[f64; WIDTH]; HEIGHT]);

impl From<Complex> for Matrix<2, 2> {
    fn from(complex: Complex) -> Self {
        Self([
            [complex.real(), -complex.im()],
            [complex.im(), complex.real()],
        ])
    }
}

impl<const HEIGHT: usize> From<ColumnVec<HEIGHT>> for Matrix<HEIGHT, 1> {
    fn from(vec: ColumnVec<HEIGHT>) -> Self {
        Matrix(<[f64; HEIGHT]>::from(vec).map(|x| [x]))
    }
}

impl<const HEIGHT: usize> From<Matrix<HEIGHT, 1>> for ColumnVec<HEIGHT> {
    fn from(mat: Matrix<HEIGHT, 1>) -> Self {
        ColumnVec::new(mat.0.map(|[x]| x))
    }
}

impl<const HEIGHT: usize, const WIDTH: usize> Default for Matrix<HEIGHT, WIDTH> {
    fn default() -> Self {
        Self([[0.; WIDTH]; HEIGHT])
    }
}

impl<const HEIGHT: usize, const WIDTH: usize> Add for Matrix<HEIGHT, WIDTH> {
    type Output = Matrix<HEIGHT, WIDTH>;

    fn add(mut self, rhs: Self) -> Self::Output {
        for i in 0..HEIGHT {
            for j in 0..WIDTH {
                self[i][j] += rhs[i][j];
            }
        }
        self
    }
}

impl<const HEIGHT: usize, const WIDTH: usize> Sub for Matrix<HEIGHT, WIDTH> {
    type Output = Matrix<HEIGHT, WIDTH>;

    fn sub(mut self, rhs: Self) -> Self::Output {
        for i in 0..HEIGHT {
            for j in 0..WIDTH {
                self[i][j] -= rhs[i][j];
            }
        }
        self
    }
}

impl<const HEIGHT: usize, const WIDTH: usize, const NEW_WIDTH: usize> Mul<Matrix<WIDTH, NEW_WIDTH>>
    for Matrix<HEIGHT, WIDTH>
{
    type Output = Matrix<HEIGHT, NEW_WIDTH>;

    fn mul(self, rhs: Matrix<WIDTH, NEW_WIDTH>) -> Self::Output {
        let mut result = Matrix::default();
        for i in 0..HEIGHT {
            for j in 0..NEW_WIDTH {
                for k in 0..WIDTH {
                    result[i][j] += self[i][k] * rhs[k][j];
                }
            }
        }
        result
    }
}

impl<const HEIGHT: usize, const WIDTH: usize> Mul<f64> for Matrix<HEIGHT, WIDTH> {
    type Output = Matrix<HEIGHT, WIDTH>;

    fn mul(self, rhs: f64) -> Self::Output {
        self.map(|x| x * rhs)
    }
}

impl<const HEIGHT: usize, const WIDTH: usize> Mul<Matrix<HEIGHT, WIDTH>> for f64 {
    type Output = Matrix<HEIGHT, WIDTH>;

    fn mul(self, rhs: Matrix<HEIGHT, WIDTH>) -> Self::Output {
        Matrix(rhs.0.map(|x| x.map(|y| y / self)))
    }
}

impl<const HEIGHT: usize, const WIDTH: usize> Neg for Matrix<HEIGHT, WIDTH> {
    type Output = Self;

    fn neg(self) -> Self::Output {
        self.map(|x| -x)
    }
}

impl<const HEIGHT: usize, const WIDTH: usize> Div<f64> for Matrix<HEIGHT, WIDTH> {
    type Output = Matrix<HEIGHT, WIDTH>;

    fn div(self, rhs: f64) -> Self::Output {
        self.map(|x| x / rhs)
    }
}

impl<const WIDTH: usize> Div<Matrix<WIDTH, WIDTH>> for Matrix<WIDTH, WIDTH> {
    type Output = Option<Matrix<WIDTH, WIDTH>>;

    /// Performs the `/` operation, multiplying by the inverse of the matrix.
    /// Note that, because matrix multiplication is not commutative, `a / b` is equivalent to `a * b.inverse()` and not `b.inverse() * a`.
    /// Returns `None` if `b` is not invertible.
    #[allow(clippy::suspicious_arithmetic_impl)] // needed due to multiplication in a division impl
    fn div(self, rhs: Matrix<WIDTH, WIDTH>) -> Self::Output {
        Some(self * rhs.inverse()?)
    }
}

impl<const WIDTH: usize> Div<Matrix<WIDTH, WIDTH>> for f64 {
    type Output = Option<Matrix<WIDTH, WIDTH>>;

    /// Performs the `/` operation, multiplying by the inverse of the matrix.
    /// Returns `None` if the matrix is not invertible.
    #[allow(clippy::suspicious_arithmetic_impl)] // needed due to multiplication in a division impl
    fn div(self, rhs: Matrix<WIDTH, WIDTH>) -> Self::Output {
        Some(rhs.inverse()? * self)
    }
}

impl Matrix<2, 2> {
    /// constructs a rotation matrix which rotates vectors by the specified angle (in radians) counterclockwise.
    pub fn rotation(angle: f64) -> Self {
        Self([[angle.cos(), -angle.sin()], [angle.sin(), angle.cos()]])
    }

    /// constructs a rotation matrix which rotates the specified vector so it lands on the positive x-axis.
    pub fn rotation_vec(vec: ColumnVec<2>) -> Self {
        let vec = vec.normalized();
        Self([[vec[0], vec[1]], [-vec[1], vec[0]]])
    }

    /// constructs a matrix which reflects vectors using a line of reflections of the specified angle.
    pub fn reflection(angle: f64) -> Self {
        let double_angle = 2. * angle;
        Self([
            [double_angle.cos(), double_angle.sin()],
            [double_angle.sin(), -double_angle.cos()],
        ])
    }

    /// constructs a matrix which reflects vectors using a line of reflections which contains the given vector.
    pub fn reflection_vec(vec: ColumnVec<2>) -> Self {
        let norm = vec.normalized();
        Self([
            [
                norm[0] * norm[0] - norm[1] * norm[1],
                2. * norm[0] * norm[1],
            ],
            [
                2. * norm[0] * norm[1],
                norm[1] * norm[1] - norm[0] * norm[0],
            ],
        ])
    }

    /// constructs a matrix which projects vectors onto a line of the specified angle.
    pub fn projection(angle: f64) -> Self {
        Self([
            [angle.cos() * angle.cos(), angle.cos() * angle.sin()],
            [angle.cos() * angle.sin(), angle.sin() * angle.sin()],
        ])
    }

    /// constructs a matrix which shears by the specified amount in the specified direction (counterclockwise from the positive x-axis in radians).
    pub fn rotated_shear(angle: f64, amount: f64) -> Self {
        Self([
            [
                1. - amount * angle.sin() * angle.cos(),
                amount * angle.cos() * angle.cos(),
            ],
            [
                -amount * angle.sin() * angle.sin(),
                1. + amount * angle.sin() * angle.cos(),
            ],
        ])
    }

    /// constructs a matrix which shears by the specified amount in the direction of the given vector.
    pub fn rotated_shear_vec(vec: ColumnVec<2>, amount: f64) -> Self {
        let norm = vec.normalized();
        Self([
            [
                norm[0] * norm[0] - amount * norm[0] * norm[1] + norm[1] * norm[1],
                amount * norm[0] * norm[0],
            ],
            [
                -amount * norm[1] * norm[1],
                norm[0] * norm[0] + amount * norm[0] * norm[1] + norm[1] * norm[1],
            ],
        ])
    }

    /// scales the x and y components in a basis rotated by the given angle.
    pub fn rotated_scale(angle: f64, x: f64, y: f64) -> Self {
        Self([
            [
                x * angle.cos() * angle.cos() + y * angle.sin() * angle.sin(),
                x * angle.sin() * angle.cos() - y * angle.sin() * angle.cos(),
            ],
            [
                x * angle.sin() * angle.cos() - y * angle.sin() * angle.cos(),
                x * angle.sin() * angle.sin() + y * angle.cos() * angle.cos(),
            ],
        ])
    }

    /// constructs a matrix which scales vectors by the x amount in the direction of the specified vector, and by the y amount in the perpendicular component.
    pub fn rotated_scale_vec(vec: ColumnVec<2>, x: f64, y: f64) -> Self {
        let norm = vec.normalized();
        Self([
            [
                x * norm[0] * norm[0] + y * norm[1] * norm[1],
                x * norm[1] * norm[0] - y * norm[1] * norm[0],
            ],
            [
                x * norm[1] * norm[0] - y * norm[1] * norm[0],
                x * norm[1] * norm[1] + y * norm[0] * norm[0],
            ],
        ])
    }
}

impl Matrix<3, 3> {
    /// constructs a random, uniformly distributed 3D rotation matrix.
    pub fn random_rotation() -> Self {
        let a = ColumnVec::<3>::random_unit();
        let b = ColumnVec::<3>::random_unit();
        let c = a.cross(b).normalized();
        let a = b.cross(c).normalized();
        Self::from_columns([a, b, c])
    }

    /// constructs a matrix that rotates vectors counterclockwise around the specified axis by the specified angle.
    /// the axis should be a normalized vector.
    pub fn rotation_around(axis: ColumnVec<3>, angle: f64) -> Self {
        let (s, c) = angle.sin_cos();
        let ColumnVec([x, y, z]) = axis;
        Self::identity() * c
            + axis.outer_product(axis) * (1. - c)
            + Self([[0., -z, y], [z, 0., -x], [-y, x, 0.]]) * s
    }
}

impl<const WIDTH: usize> Matrix<WIDTH, WIDTH> {
    /// constructs the identity matrix.
    pub fn identity() -> Self {
        let mut result = Self::zero();
        for i in 0..WIDTH {
            result[i][i] = 1.;
        }
        result
    }

    /// constructs a matrix that scales any vector by the specified amount.
    pub fn scaling(n: f64) -> Self {
        let mut result = Self::zero();
        for i in 0..WIDTH {
            result[i][i] = n;
        }
        result
    }

    /// constructs a matrix which projects vectors onto the line of the specified vector.
    pub fn projection_vec(vec: ColumnVec<WIDTH>) -> Self {
        let vec = vec.normalized();
        let mut result = Self::default();
        for i in 0..WIDTH {
            for j in 0..WIDTH {
                result[i][j] = vec[i] * vec[j];
            }
        }
        result
    }

    /// constructs a matrix which reflects vectors across a (n-1)-dimensional plane perpendicular to the specified vector.
    pub fn reflection_normal_vec(vec: ColumnVec<WIDTH>) -> Self {
        let vec = vec.normalized();
        let mut result = Self::default();
        for i in 0..WIDTH {
            for j in 0..WIDTH {
                result[i][j] = -2. * vec[i] * vec[j];
            }
        }
        result + Self::identity()
    }

    /// uses row reduction to reduce the matrix to upper-triangluar form.
    pub fn reduce_upper(self) -> Self {
        let mut result = self;
        for i in 0..WIDTH {
            if result[i][i] == 0. {
                let mut seen_nonzero = false;
                for j in i + 1..WIDTH {
                    if result[j][i] != 0. {
                        result = result.add_scaled_row(1., j, i).unwrap();
                        seen_nonzero = true;
                        break;
                    }
                }
                if !seen_nonzero {
                    continue;
                }
            }
            for j in i + 1..WIDTH {
                if result[j][i] != 0. {
                    result = result
                        .add_scaled_row(-result[j][i] / result[i][i], i, j)
                        .unwrap();
                }
            }
        }
        result
    }

    /// computes the determinant.
    pub fn determinant(self) -> f64 {
        let upper = self.reduce_upper();
        (0..WIDTH).map(|i| upper[i][i]).product::<f64>()
    }

    /// computes the inverse. Returns `None` if the matrix does not have an inverse.
    pub fn inverse(mut self) -> Option<Self> {
        if self.determinant() == 0. {
            None?;
        }
        let mut result = Matrix::<WIDTH, WIDTH>::identity();
        for i in 0..WIDTH {
            if self[i][i] == 0. {
                let mut seen_nonzero = false;
                for j in i + 1..WIDTH {
                    if self[j][i] != 0. {
                        self = self.add_scaled_row(1., j, i)?;
                        result = result.add_scaled_row(1., j, i)?;
                        seen_nonzero = true;
                        break;
                    }
                }
                if !seen_nonzero {
                    None?;
                }
            }
            for j in 0..WIDTH {
                if j != i && self[j][i] != 0. {
                    result = result.add_scaled_row(-self[j][i] / self[i][i], i, j)?;
                    self = self.add_scaled_row(-self[j][i] / self[i][i], i, j)?;
                }
            }
        }
        for i in 0..WIDTH {
            result = result.multiply_row(i, 1. / self[i][i])?;
        }
        Some(result)
    }

    /// computes the n-th power of the matrix. Equivalent to multiplying the matrix by itself n times.
    /// Any zeroth power returns the identity matrix, and negative powers return the corresponding positive power of the inverse matrix.
    /// Returns `None` if the exponent is negative and the matrix does not have an inverse.
    pub fn pow(self, n: i32) -> Option<Self> {
        Some(match n {
            0 => Self::identity(),
            1 => self,
            _ if n < 0 => self.inverse()?.pow(-n)?,
            _ if n % 2 == 1 => {
                let a = self.pow(n / 2)?;
                a * a * self
            }
            _ => {
                let a = self.pow(n / 2)?;
                a * a
            }
        })
    }

    /// returns the sum of the entries along the diagonal (top-left to bottom-right).
    pub fn trace(self) -> f64 {
        (0..WIDTH).map(|i| self[i][i]).sum()
    }
}

impl<const HEIGHT: usize, const WIDTH: usize> Matrix<HEIGHT, WIDTH> {
    /// creates a new matrix from the given array.
    pub const fn new(arr: [[f64; WIDTH]; HEIGHT]) -> Self {
        Self(arr)
    }

    /// returns a matrix where all of the entries are zero.
    pub const fn zero() -> Self {
        Self::all_same(0.)
    }

    /// returns a matrix where all of its entries are the specified number.
    pub const fn all_same(num: f64) -> Self {
        Matrix([[num; WIDTH]; HEIGHT])
    }

    /// returns a matrix whose columns are the inputted column vectors.
    pub fn from_columns(vec: [ColumnVec<HEIGHT>; WIDTH]) -> Self {
        Matrix(vec.map(std::convert::Into::into)).transpose()
    }

    /// returns true if the two matrices are equivalent within a floating-point rounding error. useful for tests.
    pub fn close_to(self, rhs: Self) -> bool {
        for i in 0..HEIGHT {
            for j in 0..WIDTH {
                if !self[i][j].close_to(rhs[i][j]) {
                    return false;
                }
            }
        }
        true
    }

    /// returns the transpose of the matrix.
    pub fn transpose(self) -> Matrix<WIDTH, HEIGHT> {
        let mut result = Matrix::default();
        for i in 0..HEIGHT {
            for j in 0..WIDTH {
                result[j][i] = self[i][j];
            }
        }
        result
    }

    /// computes the pseudo-inverse of the matrix.
    /// returns `None` if the matrix does not have a pseudo-inverse.
    pub fn pseudo_inverse(self) -> Option<Matrix<WIDTH, HEIGHT>> {
        Some((self.transpose() * self).inverse()? * self.transpose())
    }

    /// multiplies all entries in the specified row by the given factor.
    /// Returns `None` if the specified row does not exist.
    pub fn multiply_row(mut self, row: usize, factor: f64) -> Option<Self> {
        if row >= HEIGHT {
            return None;
        }
        for i in 0..WIDTH {
            self[row][i] *= factor;
        }
        Some(self)
    }

    /// swaps all entries in 2 rows of the matrix.
    /// Returns `None` if the specified row does not exist.
    pub fn swap_rows(mut self, row1: usize, row2: usize) -> Option<Self> {
        if row1 >= HEIGHT || row2 >= HEIGHT {
            return None;
        }
        for i in 0..WIDTH {
            (self[row1][i], self[row2][i]) = (self[row2][i], self[row1][i]);
        }
        Some(self)
    }

    /// adds all entries of one row scaled by the factor to another row.
    /// Returns `None` if either of the specified rows do not exist.
    pub fn add_scaled_row(mut self, factor: f64, row1: usize, row2: usize) -> Option<Self> {
        if row1 >= HEIGHT || row2 >= HEIGHT {
            return None;
        }
        for i in 0..WIDTH {
            self[row2][i] += self[row1][i] * factor;
        }
        Some(self)
    }

    /// multiplies each row of the matrix by the corresponding element of the given vector.
    pub fn multiply_rows(mut self, vec: ColumnVec<WIDTH>) -> Self {
        for i in 0..HEIGHT {
            for j in 0..WIDTH {
                self[i][j] *= vec[j];
            }
        }
        self
    }

    /// creates a new matrix where each element is the function applied to the original element.
    pub fn map<F>(mut self, func: F) -> Self
    where
        F: Fn(f64) -> f64,
    {
        for i in 0..HEIGHT {
            for j in 0..WIDTH {
                self[i][j] = func(self[i][j]);
            }
        }
        self
    }

    /// converts the matrix to a matrix of a different size.
    /// If the new matrix is smaller, the last elements will be removed.
    /// If the new matrix is larger, the new elements will be zero.
    pub fn resize<const NEW_HEIGHT: usize, const NEW_WIDTH: usize>(
        self,
    ) -> Matrix<NEW_HEIGHT, NEW_WIDTH> {
        let mut result = Matrix::<NEW_HEIGHT, NEW_WIDTH>::default();
        for i in 0..NEW_HEIGHT {
            for j in 0..NEW_WIDTH {
                result[i][j] = if i > HEIGHT || j > WIDTH {
                    0.
                } else {
                    self[i][j]
                }
            }
        }
        result
    }
}
