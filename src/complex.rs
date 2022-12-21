use std::{
    fmt::Display,
    ops::{Add, Div, Mul, Neg, Sub},
};

/// a complex number with real and imaginary components
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Complex {
    pub real: f64,
    pub im: f64,
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
        use std::cmp::Ordering::*;
        let Complex { real, im } = *self;
        if real.is_nan() || im.is_nan() {
            return write!(f, "NaN");
        }
        match (real == 0., im == 1., im == -1., im.partial_cmp(&0.)) {
            (_, true, true, _)                   => unreachable!(),
            (.., None)                           => write!(f, "NaN"),
            (true, true, false, _)               => write!(f, "i"),
            (true, false, true, _)               => write!(f, "-i"),
            (true, false, false, Some(Less))     => write!(f, "-{}i", -im),
            (true, false, false, Some(Equal))    => write!(f, "0"),
            (true, false, false, Some(Greater))  => write!(f, "{}i", im),
            (false, true, false, _)              => write!(f, "{} + i", real),
            (false, false, true, _)              => write!(f, "{} - i", real),
            (false, false, false, Some(Less))    => write!(f, "{} - {}i", real, -im),
            (false, false, false, Some(Equal))   => write!(f, "{}", real),
            (false, false, false, Some(Greater)) => write!(f, "{} + {}i", real, im),
        }
    }
}

impl Neg for Complex {
    type Output = Self;

    fn neg(self) -> Self::Output {
        Self::new(-self.real, -self.im)
    }
}

impl Add for Complex {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self::new(self.real + rhs.real, self.im + rhs.im)
    }
}

impl Sub for Complex {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        self + -rhs
    }
}

impl Mul<f64> for Complex {
    type Output = Self;

    fn mul(self, rhs: f64) -> Self::Output {
        Self::new(self.real * rhs, self.im * rhs)
    }
}

impl Mul<Complex> for f64 {
    type Output = Complex;

    fn mul(self, rhs: Complex) -> Self::Output {
        Complex::new(rhs.real * self, rhs.im * self)
    }
}

impl Div<f64> for Complex {
    type Output = Self;

    fn div(self, rhs: f64) -> Self::Output {
        Self::new(self.real / rhs, self.im / rhs)
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

    #[allow(clippy::suspicious_arithmetic_impl)]
    fn div(self, rhs: Self) -> Self::Output {
        self * rhs.inverse()
    }
}

pub const I: Complex = Complex { real: 0., im: 1. };

impl Complex {
    /// Creates a new [`Complex`] with the given real and imaginary components.
    pub fn new(real: f64, im: f64) -> Self {
        Self { real, im }
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
        self.abs_squared().sqrt()
    }

    /// returns the angle between the complex number and the positive real line.
    pub fn angle(self) -> f64 {
        self.im.atan2(self.real)
    }

    /// computes the square of the absolute value of the complex number.
    /// This is faster than computing the absolute value and then squaring it.
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

    /// computes the complex number raised to the power.
    pub fn powf(self, rhs: f64) -> Self {
        Self::from_polar(self.abs().powf(rhs), self.angle() * rhs)
    }

    /// returns the square root of a complex number.
    pub fn sqrt(self) -> Self {
        Self::from_polar(self.abs().sqrt(), self.angle() / 2.)
    }

    /// returns e raised to the power of the complex number.
    pub fn exp(self) -> Self {
        Self::from_polar(self.real.exp(), self.im)
    }

    /// returns the natural log of the complex number.
    /// Since the natural log is not unique, returns the complex number a+bi with -pi<b<=pi.
    pub fn ln(self) -> Self {
        Self::new(self.abs().ln(), self.angle())
    }

    /// Returns the logarithm of the number with respect to an arbitrary base.
    pub fn log(self, rhs: Self) -> Self {
        self.ln() / rhs.ln()
    }

    /// computes the complex number raised to another complex number.
    pub fn powc(self, rhs: Self) -> Self {
        (rhs * self.ln()).exp()
    }

    /// Returns `true` if this value is NaN.
    pub fn is_nan(self) -> bool {
        self.real.is_nan() || self.im.is_nan()
    }

    /// returns true if the two numbers are equivalent within a floating-point rounding error. useful for tests.
    pub fn is_close_enough(self, rhs: Self) -> bool {
        (self - rhs).abs() < f64::EPSILON
    }
}
