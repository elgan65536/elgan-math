pub use crate::affine::*;
pub use crate::columnvec::*;
pub use crate::matrix::*;

#[cfg(test)]
mod test {
    use super::{ColumnVec, Matrix};

    #[test]
    fn test_matr_add_sub() {
        let a = Matrix::new([[2., 7., 1.], [8., 2., 8.]]);
        let b = Matrix::new([[3., 1., 4.], [1., 5., 9.]]);
        assert_eq!(a + b, Matrix::new([[5., 8., 5.], [9., 7., 17.]]));
        assert_eq!(a - b, Matrix::new([[-1., 6., -3.], [7., -3., -1.]]));
    }

    #[test]
    fn test_matr_mul() {
        let a = Matrix::new([[2., 7., 1.], [8., 2., 8.]]);
        let b = Matrix::new([[3., 1.], [4., 1.], [5., 9.]]);
        assert_eq!(a * b, Matrix::new([[39., 18.], [72., 82.]]));
    }

    #[test]
    fn test_matr_scalar_mul() {
        let a = Matrix::new([[3., 1.], [4., 1.], [5., 9.]]);
        assert_eq!(a * 2., Matrix::new([[6., 2.], [8., 2.], [10., 18.]]));
        assert_eq!(a / 2., Matrix::new([[1.5, 0.5], [2., 0.5], [2.5, 4.5]]));
    }

    #[test]
    fn test_vec_add_sub() {
        let a = ColumnVec::new([3., 1., 4., 1.]);
        let b = ColumnVec::new([2., 7., 1., 8.]);
        assert_eq!(a + b, ColumnVec::new([5., 8., 5., 9.]));
        assert_eq!(a - b, ColumnVec::new([1., -6., 3., -7.]));
    }

    #[test]
    fn test_dot_product() {
        let a = ColumnVec::new([3., 1., 4., 1.]);
        let b = ColumnVec::new([2., 7., 1., 8.]);
        assert_eq!(a * b, 25.);
    }

    #[test]
    fn test_outer_product() {
        let a = ColumnVec::new([3., 1., 4.]);
        let b = ColumnVec::new([2., 7., 1.]);
        assert_eq!(
            a.outer_product(b),
            Matrix::new([[6., 21., 3.], [2., 7., 1.], [8., 28., 4.]])
        );
        assert_eq!(
            b.outer_product(a),
            Matrix::new([[6., 2., 8.,], [21., 7., 28.], [3., 1., 4.]])
        );
    }

    #[test]
    fn test_vec_scalar_mul() {
        let a = ColumnVec::new([3., 1., 4., 1.]);
        assert_eq!(a * 2., ColumnVec::new([6., 2., 8., 2.]));
        assert_eq!(a / 2., ColumnVec::new([1.5, 0.5, 2., 0.5]));
    }

    #[test]
    fn test_vec_matr_mul() {
        let a = Matrix::new([[3., 1., 4.], [1., 5., 9.]]);
        let b = ColumnVec::new([2., 7., 1.]);
        assert_eq!(a * b, ColumnVec::new([17., 46.]));
    }

    #[test]
    fn test_vec_length() {
        assert_eq!(ColumnVec::new([8., 15.]).length(), 17.);
        assert_eq!(ColumnVec::new([3., 4., 12.]).length(), 13.);
    }

    #[test]
    fn rotation_test() {
        assert_eq!(Matrix::rotation(0.), Matrix::new([[1., 0.], [0., 1.]]));
    }

    #[test]
    fn angle_test() {
        assert_eq!(
            ColumnVec::new([1., 0.]).angle_between(ColumnVec::new([6., 0.])),
            0.
        );
        assert_eq!(
            ColumnVec::new([3., 0.]).angle_between(ColumnVec::new([0., 5.])),
            90f64.to_radians()
        );
        assert_eq!(
            ColumnVec::new([0., 5.]).angle_between(ColumnVec::new([3., 0.])),
            90f64.to_radians()
        );
        assert_eq!(
            ColumnVec::new([3., 0.]).angle_between(ColumnVec::new([-5., 5.])),
            135f64.to_radians()
        );
    }

    #[test]
    fn multiply_row_test() {
        let a = Matrix::new([[3., 1., 4.], [1., 5., 9.]]);
        assert_eq!(a.multiply_row(2, 1.), None);
        assert_eq!(
            a.multiply_row(1, 1.5),
            Some(Matrix::new([[3., 1., 4.], [1.5, 7.5, 13.5]]))
        );
    }

    #[test]
    fn swap_row_test() {
        let a = Matrix::new([[3., 1.], [4., 1.], [5., 9.]]);
        assert_eq!(a.swap_rows(0, 3), None);
        assert_eq!(a.swap_rows(3, 0), None);
        assert_eq!(
            a.swap_rows(0, 1),
            Some(Matrix::new([[4., 1.], [3., 1.], [5., 9.]]))
        );
    }

    #[test]
    fn add_row_test() {
        let a = Matrix::new([[3., 1.], [4., 1.], [5., 9.]]);
        assert_eq!(a.add_scaled_row(1., 0, 3), None);
        assert_eq!(a.add_scaled_row(1., 3, 0), None);
        assert_eq!(
            a.add_scaled_row(1.5, 0, 1),
            Some(Matrix::new([[3., 1.], [8.5, 2.5], [5., 9.]]))
        );
    }

    #[test]
    fn reduction_test() {
        let a = Matrix::new([[2., 1., 4.], [4., 5., 9.], [6., 6., 5.]]);
        assert_eq!(
            a.reduce_upper(),
            Matrix::new([[2., 1., 4.], [0., 3., 1.], [0., 0., -8.]])
        );
    }

    #[test]
    fn determinant_test() {
        let a = Matrix::new([[2., 1., 4.], [4., 5., 9.], [6., 6., 5.]]);
        let b = Matrix::new([[0., 1.], [4., 6.]]);
        let c = Matrix::new([[2., -5.], [-1., 2.5]]);
        assert_eq!(a.determinant(), -48.);
        assert_eq!(b.determinant(), -4.);
        assert_eq!(c.determinant(), 0.);
    }

    #[test]
    fn inverse_test() {
        let a = Matrix::new([[2., 1., 4.], [4., 5., 9.], [6., 6., 5.]]);
        let b = Matrix::new([[0., 1.], [4., 6.]]);
        let c = Matrix::new([[2., -5.], [-1., 2.5]]);
        assert!(a
            .inverse()
            .unwrap()
            .close_to(Matrix::new([[29., -19., 11.], [-34., 14., 2.], [6., 6., -6.]]) / 48.));
        assert_eq!(b.inverse(), Some(Matrix::new([[-1.5, 0.25], [1., 0.]])));
        assert_eq!(c.inverse(), None);
    }
}
