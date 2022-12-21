use std::ops::{Add, Div, Index, IndexMut, Mul, Neg, Sub};

use crate::{complex::Complex, linalg::ColumnVec};

/// a fixed-size matrix which can be of any size.
#[derive(Debug, PartialEq, Clone, Copy)]
pub struct Matrix<const A: usize, const B: usize>(pub [[f64; B]; A]);

impl<const A: usize, const B: usize> From<Matrix<A, B>> for [[f64; B]; A] {
    fn from(matrix: Matrix<A, B>) -> Self {
        matrix.0
    }
}

impl<const A: usize, const B: usize> From<[[f64; B]; A]> for Matrix<A, B> {
    fn from(arr: [[f64; B]; A]) -> Self {
        Matrix(arr)
    }
}

impl From<Complex> for Matrix<2, 2> {
    fn from(Complex { real, im }: Complex) -> Self {
        Self([[real, -im], [im, real]])
    }
}

impl<const A: usize> From<ColumnVec<A>> for Matrix<A, 1> {
    fn from(vec: ColumnVec<A>) -> Self {
        Matrix(vec.0.map(|x| [x]))
    }
}

impl<const A: usize> From<Matrix<A, 1>> for ColumnVec<A> {
    fn from(mat: Matrix<A, 1>) -> Self {
        ColumnVec(mat.0.map(|[x]| x))
    }
}

impl<const A: usize, const B: usize> Default for Matrix<A, B> {
    fn default() -> Self {
        Self([[0.; B]; A])
    }
}

impl<const A: usize, const B: usize> Index<usize> for Matrix<A, B> {
    type Output = [f64; B];

    fn index(&self, index: usize) -> &Self::Output {
        &self.0[index]
    }
}

impl<const A: usize, const B: usize> IndexMut<usize> for Matrix<A, B> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.0[index]
    }
}

impl<const A: usize, const B: usize> Add for Matrix<A, B> {
    type Output = Matrix<A, B>;

    fn add(self, rhs: Self) -> Self::Output {
        let mut result = self;
        for i in 0..A {
            for j in 0..B {
                result[i][j] += rhs[i][j];
            }
        }
        result
    }
}

impl<const A: usize, const B: usize> Sub for Matrix<A, B> {
    type Output = Matrix<A, B>;

    fn sub(self, rhs: Self) -> Self::Output {
        let mut result = self;
        for i in 0..A {
            for j in 0..B {
                result[i][j] -= rhs[i][j];
            }
        }
        result
    }
}

impl<const A: usize, const B: usize, const C: usize> Mul<Matrix<B, C>> for Matrix<A, B> {
    type Output = Matrix<A, C>;

    fn mul(self, rhs: Matrix<B, C>) -> Self::Output {
        let mut result = Matrix::zero();
        for i in 0..A {
            for j in 0..C {
                for k in 0..B {
                    result[i][j] += self[i][k] * rhs[k][j];
                }
            }
        }
        result
    }
}

impl<const A: usize, const B: usize> Mul<f64> for Matrix<A, B> {
    type Output = Matrix<A, B>;

    fn mul(self, rhs: f64) -> Self::Output {
        Matrix(self.0.map(|x| x.map(|y| y * rhs)))
    }
}

impl<const A: usize, const B: usize> Mul<Matrix<A, B>> for f64 {
    type Output = Matrix<A, B>;

    fn mul(self, rhs: Matrix<A, B>) -> Self::Output {
        Matrix(rhs.0.map(|x| x.map(|y| y / self)))
    }
}

impl<const A: usize, const B: usize> Neg for Matrix<A, B> {
    type Output = Self;

    fn neg(self) -> Self::Output {
        -1. * self
    }
}

impl<const A: usize, const B: usize> Div<f64> for Matrix<A, B> {
    type Output = Matrix<A, B>;

    fn div(self, rhs: f64) -> Self::Output {
        Matrix(self.0.map(|x| x.map(|y| y / rhs)))
    }
}

impl<const A: usize> Div<Matrix<A, A>> for Matrix<A, A> {
    type Output = Option<Matrix<A, A>>;

    /// Performs the `/` operation, multiplying by the inverse of the matrix.
    /// Note that, because matrix multiplication is not commutative, `a / b` is equivalent to `a * b.inverse()` and not `b.inverse() * a`.
    /// Returns `None` if the matrix is not invertible.
    #[allow(clippy::suspicious_arithmetic_impl)] // needed due to multiplication in a division impl
    fn div(self, rhs: Matrix<A, A>) -> Self::Output {
        Some(self * rhs.inverse()?)
    }
}

impl<const A: usize> Div<Matrix<A, A>> for f64 {
    type Output = Option<Matrix<A, A>>;

    /// Performs the `/` operation, multiplying by the inverse of the matrix.
    /// Returns `None` if the matrix is not invertible.
    #[allow(clippy::suspicious_arithmetic_impl)] // needed due to multiplication in a division impl
    fn div(self, rhs: Matrix<A, A>) -> Self::Output {
        Some(rhs.inverse()? * self)
    }
}

impl Matrix<2, 2> {
    /// constructs a rotation matrix which rotates vectors by the specified angle (in radians) counterclockwise.
    pub fn rotation(angle: f64) -> Self {
        Matrix([[angle.cos(), -angle.sin()], [angle.sin(), angle.cos()]])
    }

    /// constructs a rotation matrix which rotates the specified vector so it lands on the positive x-axis.
    pub fn rotation_vec(vec: ColumnVec<2>) -> Self {
        let vec = vec.normalized();
        Matrix([[vec[0], vec[1]], [-vec[1], vec[0]]])
    }

    /// constructs a matrix which reflects vectors using a line of reflections of the specified angle.
    pub fn reflection(angle: f64) -> Self {
        let angle = 2. * angle;
        Matrix([[angle.cos(), angle.sin()], [angle.sin(), -angle.cos()]])
    }

    /// constructs a matrix which reflects vectors using a line of reflections which contains the given vector.
    pub fn reflection_vec(vec: ColumnVec<2>) -> Self {
        let vec = vec.normalized();
        Matrix([
            [vec[0] * vec[0] - vec[1] * vec[1], 2. * vec[0] * vec[1]],
            [2. * vec[0] * vec[1], vec[1] * vec[1] - vec[0] * vec[0]],
        ])
    }

    /// constructs a matrix which projects vectors onto a line of the specified angle.
    pub fn projection(angle: f64) -> Self {
        Matrix([
            [angle.cos() * angle.cos(), angle.cos() * angle.sin()],
            [angle.cos() * angle.sin(), angle.sin() * angle.sin()],
        ])
    }

    /// constructs a matrix which shears by the specified amount in the specified direction (counterclockwise from the positive x-axis in radians).
    pub fn rotated_shear(angle: f64, amount: f64) -> Self {
        Matrix([
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
        let vec = vec.normalized();
        Matrix([
            [
                vec[0] * vec[0] - amount * vec[0] * vec[1] + vec[1] * vec[1],
                amount * vec[0] * vec[0],
            ],
            [
                -amount * vec[1] * vec[1],
                vec[0] * vec[0] + amount * vec[0] * vec[1] + vec[1] * vec[1],
            ],
        ])
    }

    /// scales the x and y components in a basis rotated by the given angle.
    pub fn rotated_scale(angle: f64, x: f64, y: f64) -> Self {
        Matrix([
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

    /// constructs a matrix which scales vectors by the x amount inthe direction of the specified vector, and by the y amount in the perpendicular component.
    pub fn rotated_scale_vec(vec: ColumnVec<2>, x: f64, y: f64) -> Self {
        let vec = vec.normalized();
        Matrix([
            [
                x * vec[0] * vec[0] + y * vec[1] * vec[1],
                x * vec[1] * vec[0] - y * vec[1] * vec[0],
            ],
            [
                x * vec[1] * vec[0] - y * vec[1] * vec[0],
                x * vec[1] * vec[1] + y * vec[0] * vec[0],
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
        Self([[a[0], b[0], c[0]], [a[1], b[1], c[1]], [a[2], b[2], c[2]]])
    }
}

impl<const A: usize> Matrix<A, A> {
    /// constructs the identity matrix.
    pub fn identity() -> Self {
        let mut result = Self::zero();
        for i in 0..A {
            result[i][i] = 1.;
        }
        result
    }

    /// constructs a matrix that scales any vector by the specified amount.
    pub fn scaling(n: f64) -> Self {
        Self::identity() * n
    }

    /// constructs a matrix which projects vectors onto the line of the specified vector.
    pub fn projection_vec(vec: ColumnVec<A>) -> Self {
        let vec = vec.normalized();
        let mut result = Matrix([[0.; A]; A]);
        for i in 0..A {
            for j in 0..A {
                result[i][j] = vec[i] * vec[j];
            }
        }
        result
    }

    /// constructs a matrix which reflects vectors across a (n-1)-dimensional plane perpendicular to the specified vector.
    pub fn reflection_normal_vec(vec: ColumnVec<A>) -> Self {
        let vec = vec.normalized();
        let mut result = Matrix([[0.; A]; A]);
        for i in 0..A {
            for j in 0..A {
                result[i][j] = -2. * vec[i] * vec[j];
            }
        }
        result + Self::identity()
    }

    /// uses row reduction to reduce the matrix to upper-triangluar form.
    pub fn reduce_upper(self) -> Self {
        let mut result = self;
        for i in 0..A {
            if result[i][i] == 0. {
                let mut seen_nonzero = false;
                for j in i + 1..A {
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
            for j in i + 1..A {
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
        let a = (0..A).map(|i| self.reduce_upper()[i][i]).product::<f64>();
        if a == 0. {
            0.
        } else {
            1. / a
        }
    }

    /// computes the inverse. Returns `None` if the matrix does not have an inverse.
    pub fn inverse(mut self) -> Option<Self> {
        if self.determinant() == 0. {
            return None;
        }
        let mut result = Matrix::<A, A>::identity();
        for i in 0..A {
            if self[i][i] == 0. {
                let mut seen_nonzero = false;
                for j in i + 1..A {
                    if self[j][i] != 0. {
                        self = self.add_scaled_row(1., j, i)?;
                        result = result.add_scaled_row(1., j, i)?;
                        seen_nonzero = true;
                        break;
                    }
                }
                if !seen_nonzero {
                    return None;
                }
            }
            for j in 0..A {
                if j != i && self[j][i] != 0. {
                    result = result.add_scaled_row(-self[j][i] / self[i][i], i, j)?;
                    self = self.add_scaled_row(-self[j][i] / self[i][i], i, j)?;
                }
            }
        }
        for i in 0..A {
            result = result.multiply_row(i, 1. / self[i][i])?;
        }
        Some(result)
    }

    /// computes the n-th power of the matrix. Equivalent to multiplying the matrix by itself n times.
    /// Any zeroth power returns the identity matrix, and negative powers return the corresponding positive power of the inverse matrix.
    /// Returns `None` if the exponent is negative and the matrix does not have an inverse.
    pub fn pow(self, n: i32) -> Option<Self> {
        match (n, n < 0, n % 2 == 0) {
            (_, true, _) => self.inverse()?.pow(-n),
            (0, ..) => Some(Self::identity()),
            (1, ..) => Some(self),
            (.., true) => {
                let a = self.pow(n / 2)?;
                Some(a * a)
            }
            (.., false) => Some(self.pow(n - 1)? * self),
        }
    }

    /// returns the sum of the entries along the diagonal (top-left to bottom-right).
    pub fn trace(self) -> f64 {
        (0..A).map(|i| self[i][i]).sum()
    }
}

impl<const A: usize, const B: usize> Matrix<A, B> {
    /// returns a matrix where all of the entries are zero.
    pub fn zero() -> Self {
        Matrix([[0.; B]; A])
    }

    pub fn from_columns(vec: [ColumnVec<A>; B]) -> Self {
        Matrix(vec.map(|x| x.0)).transpose()
    }

    /// returns true if the two matrices are equivalent within a floating-point rounding error. useful for tests.
    pub fn is_close_enough(self, rhs: Self) -> bool {
        for i in 0..A {
            for j in 0..B {
                if (self[i][j] - rhs[i][j]).abs() > f64::EPSILON {
                    return false;
                }
            }
        }
        true
    }

    /// returns the transpose of the matrix.
    pub fn transpose(self) -> Matrix<B, A> {
        let mut result = Matrix::<B, A>::zero();
        for i in 0..A {
            for j in 0..B {
                result[j][i] = self[i][j];
            }
        }
        result
    }

    /// multiplies all entries in the specified row by the given factor.
    /// Returns `None` if the specified row does not exist.
    pub fn multiply_row(self, row: usize, factor: f64) -> Option<Self> {
        if row >= A {
            return None;
        }
        let mut result = self;
        for i in 0..B {
            result[row][i] *= factor;
        }
        Some(result)
    }

    /// swaps all entries in 2 rows of the matrix.
    /// Returns `None` if the specified row does not exist.
    pub fn swap_rows(self, row1: usize, row2: usize) -> Option<Self> {
        if row1 >= A || row2 >= A {
            return None;
        }
        let mut result = self;
        for i in 0..B {
            (result[row1][i], result[row2][i]) = (result[row2][i], result[row1][i]);
        }
        Some(result)
    }

    /// adds all entries of one row scaled by the factor to another row.
    /// Returns `None` if the specified row does not exist.
    pub fn add_scaled_row(self, factor: f64, row1: usize, row2: usize) -> Option<Self> {
        if row1 >= A || row2 >= A {
            return None;
        }
        let mut result = self;
        for i in 0..B {
            result[row2][i] += result[row1][i] * factor;
        }
        Some(result)
    }
}
