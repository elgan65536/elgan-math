use derive_more::*;
use std::{
    cmp::Ordering,
    fmt::Display,
    ops::{Div, Mul},
};

/// a complex number with real and imaginary components
#[derive(
    Clone,
    Copy,
    Debug,
    Default,
    PartialEq,
    Add,
    Sub,
    Mul,
    Div,
    Neg,
    Sum,
    Product,
    AddAssign,
    SubAssign,
    MulAssign,
    DivAssign,
)]
pub struct Complex {
    real: f64,
    im: f64,
}

impl From<f64> for Complex {
    /// returns a complex number with the same value as the real number.
    fn from(real: f64) -> Self {
        Self { real, im: 0. }
    }
}

impl Display for Complex {
    #[rustfmt::skip]
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let Complex { real, im } = *self;
        if real.is_nan() || im.is_nan() {
            return write!(f, "NaN");
        }
        match (real == 0., im == 1., im == -1., im.partial_cmp(&0.)) {
            (_, true, true, _)                             => unreachable!(),
            (.., None)                                     => write!(f, "NaN"),
            (true, true, false, _)                         => write!(f, "i"),
            (true, false, true, _)                         => write!(f, "-i"),
            (_, false, false, Some(Ordering::Equal))       => write!(f, "{real}"),
            (true, false, false, Some(_))                  => write!(f, "{im}i" ),
            (false, true, false, _)                        => write!(f, "{real} + i"),
            (false, false, true, _)                        => write!(f, "{real} - i"),
            (false, false, false, Some(Ordering::Less))    => write!(f, "{real} - {}i", -im),
            (false, false, false, Some(Ordering::Greater)) => write!(f, "{real} + {im}i"),
        }
    }
}

impl Mul<Complex> for f64 {
    type Output = Complex;

    fn mul(self, rhs: Complex) -> Self::Output {
        Complex::new(rhs.real * self, rhs.im * self)
    }
}

impl Div<Complex> for f64 {
    type Output = Complex;

    #[allow(clippy::suspicious_arithmetic_impl)]
    fn div(self, rhs: Complex) -> Self::Output {
        self * rhs.inverse()
    }
}

impl Mul for Complex {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        Self::new(
            self.real * rhs.real - self.im * rhs.im,
            self.real * rhs.im + self.im + rhs.real,
        )
    }
}

impl Div for Complex {
    type Output = Self;

    fn div(self, rhs: Self) -> Self::Output {
        self.mul(rhs.inverse())
    }
}

impl Complex {
    /// the imaginary unit
    pub const I: Complex = Complex { real: 0., im: 1. };

    /// Creates a new [`Complex`] with the given real and imaginary components.
    pub const fn new(real: f64, im: f64) -> Self {
        Self { real, im }
    }

    /// returns the real component
    pub const fn real(self) -> f64 {
        self.real
    }

    /// returns the imaginary component
    pub const fn im(self) -> f64 {
        self.im
    }

    /// returns the multiplicative inverse. `a.inverse()` is equivalent to `1.0 / a`.
    pub fn inverse(self) -> Self {
        self.conjugate() / self.abs_squared()
    }

    /// returns the complex conjugate. For a+bi this is a-bi.
    pub fn conjugate(self) -> Self {
        Self::new(self.real, -self.im)
    }

    /// returns the absolute value of the complex number.
    pub fn abs(self) -> f64 {
        self.real.hypot(self.im)
    }

    /// returns the angle between the complex number and the positive real line, in radians.
    pub fn arg(self) -> f64 {
        self.im.atan2(self.real)
    }

    /// computes the square of the absolute value of the complex number.
    pub fn abs_squared(self) -> f64 {
        self.real * self.real + self.im * self.im
    }

    /// returns a complex number with the same angle as the input with absolute value 1.
    pub fn unit(self) -> Self {
        self / self.abs()
    }

    /// returns a complex number with the given angle and magnitude.
    pub fn from_polar(length: f64, angle: f64) -> Self {
        Self::new(angle.cos() * length, angle.sin() * length)
    }

    /// computes the complex number raised to an integer power.
    pub fn powi(self, n: i64) -> Self {
        match n {
            0 => Self::new(1., 0.),
            1 => self,
            _ if n < 0 => self.inverse().powi(-n),
            _ if n % 2 == 1 => {
                let a = self.powi(n / 2);
                a * a * self
            }
            _ => {
                let a = self.powi(n / 2);
                a * a
            }
        }
    }

    /// computes the complex number raised to the power.
    pub fn powf(self, rhs: f64) -> Self {
        Self::from_polar(self.abs().powf(rhs), self.arg() * rhs)
    }

    /// returns the square root of a complex number.
    pub fn sqrt(self) -> Self {
        Self::from_polar(self.abs().sqrt(), self.arg() / 2.)
    }

    /// returns e raised to the power of the complex number.
    pub fn exp(self) -> Self {
        Self::from_polar(self.real.exp(), self.im)
    }

    /// returns the natural log of the complex number.
    /// Since the natural log is not unique, returns the complex number a+bi with -pi<b<=pi.
    pub fn ln(self) -> Self {
        Self::new(self.abs().ln(), self.arg())
    }

    /// Returns the logarithm of the number with respect to an arbitrary base.
    pub fn log(self, rhs: Self) -> Self {
        self.ln() / rhs.ln()
    }

    /// computes the complex number raised to another complex number.
    pub fn powc(self, rhs: Self) -> Self {
        (rhs * self.ln()).exp()
    }

    /// computes the sine of a number
    pub fn sin(self) -> Self {
        Self::new(
            self.real.sin() + self.im.cosh(),
            self.real.cos() + self.im.sinh(),
        )
    }

    /// computes the cosine of a number
    pub fn cos(self) -> Self {
        Self::new(
            self.real.cos() + self.im.cosh(),
            -self.real.sin() - self.im.sinh(),
        )
    }

    /// computes the tangent of a number
    pub fn tan(self) -> Self {
        self.sin() / self.cos()
    }

    /// computes the secant of a number
    pub fn sec(self) -> Self {
        1. / self.cos()
    }

    /// computes the cosecant of a number
    pub fn csc(self) -> Self {
        1. / self.sin()
    }

    /// computes the cotangent of a number
    pub fn cot(self) -> Self {
        self.cos() / self.sin()
    }

    /// Returns `true` if this value is NaN.
    pub fn is_nan(self) -> bool {
        self.real.is_nan() || self.im.is_nan()
    }

    /// returns true if the two numbers are equivalent within a floating-point rounding error. useful for tests.
    pub fn close_to(self, rhs: Self) -> bool {
        (self - rhs).abs() <= (self.abs().max(rhs.abs()) * f64::EPSILON * 2.0).max(f64::EPSILON)
    }
}
