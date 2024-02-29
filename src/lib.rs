#![deny(missing_docs)]
//! The elgan-Math library.

mod affine;
mod arithmetic;
mod columnvec;
mod complex;
mod fraction;
mod linalg;
mod matrix;
mod polynomial;

pub use arithmetic::*;
pub use complex::*;
pub use fraction::*;
pub use linalg::*;
pub use polynomial::*;
