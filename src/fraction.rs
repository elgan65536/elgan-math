use std::{
    cmp::Ordering,
    fmt::Display,
    ops::{Add, Div, Mul, Neg, Sub},
};

use crate::arithmetic;

#[derive(Clone, Copy, Debug)]
pub struct Fraction {
    pub num: i64,
    pub den: i64,
}

impl From<Fraction> for f32 {
    fn from(frac: Fraction) -> Self {
        frac.num as f32 / frac.den as f32
    }
}

impl From<Fraction> for f64 {
    fn from(frac: Fraction) -> Self {
        frac.num as f64 / frac.den as f64
    }
}

impl From<i64> for Fraction {
    fn from(n: i64) -> Self {
        Self { num: n, den: 1 }
    }
}

impl Default for Fraction {
    fn default() -> Self {
        Self { num: 0, den: 1 }
    }
}

impl Display for Fraction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} / {}", self.num, self.den)
    }
}

impl Neg for Fraction {
    type Output = Self;

    fn neg(self) -> Self::Output {
        Self {
            num: -self.num,
            den: self.den,
        }
    }
}

impl Add for Fraction {
    type Output = Self;

    /// performs the `+` operation. In the case of overflow, finds a fraction close to the correct answer that does not overflow.
    fn add(self, rhs: Self) -> Self::Output {
        let mut num = self.num as i128 * rhs.den as i128 + self.den as i128 * rhs.num as i128;
        let mut den = self.den as i128 * rhs.den as i128;
        let common_fac = arithmetic::gcd(num.abs(), den.abs());
        num /= common_fac;
        den /= common_fac;
        while num > i64::MAX as i128
            || den > i64::MAX as i128
            || num < i64::MIN as i128
            || den < i64::MIN as i128
        {
            num /= 2;
            den /= 2;
        }
        Self::new(num as i64, den as i64)
    }
}

impl Sub for Fraction {
    type Output = Self;

    /// performs the `-` operation. In the case of overflow, finds a fraction close to the correct answer that does not overflow.
    fn sub(self, rhs: Self) -> Self::Output {
        self + -rhs
    }
}

impl Mul for Fraction {
    type Output = Self;

    /// performs the `*` operation. In the case of overflow, finds a fraction close to the correct answer that does not overflow.
    fn mul(self, rhs: Self) -> Self::Output {
        let mut num = self.num as i128 * rhs.num as i128;
        let mut den = self.den as i128 * rhs.den as i128;
        let common_fac = arithmetic::gcd(num.abs(), den.abs());
        num /= common_fac;
        den /= common_fac;
        while num > i64::MAX as i128
            || den > i64::MAX as i128
            || num < i64::MIN as i128
            || den < i64::MIN as i128
        {
            num /= 2;
            den /= 2;
        }
        Self::new(num as i64, den as i64)
    }
}

impl Div for Fraction {
    type Output = Self;

    /// performs the `/` operation. In the case of overflow, finds a fraction close to the correct answer that does not overflow.
    #[allow(clippy::suspicious_arithmetic_impl)]
    fn div(self, rhs: Self) -> Self::Output {
        self * rhs.inverse()
    }
}

impl PartialEq for Fraction {
    fn eq(&self, other: &Self) -> bool {
        self.num == other.num && self.den == other.den
    }
}

impl Eq for Fraction {
    fn assert_receiver_is_total_eq(&self) {}
}

impl PartialOrd for Fraction {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        if self.den == 0 || other.den == 0 {
            return None;
        }
        let a = f64::from(*self);
        let b = f64::from(*other);
        Some(if a > b {
            Ordering::Greater
        } else if a < b {
            Ordering::Less
        } else {
            Ordering::Equal
        })
    }
}

pub const MAX: Fraction = Fraction {
    num: i64::MAX,
    den: 1,
};
pub const MIN: Fraction = Fraction {
    num: i64::MIN,
    den: 1,
};
pub const MIN_POSITIVE: Fraction = Fraction {
    num: 1,
    den: i64::MAX,
};
pub const MAX_NEGATIVE: Fraction = Fraction {
    num: -1,
    den: i64::MAX,
};

impl Fraction {
    /// Creates a new [`Fraction`] with the given numerator and denominator.
    /// The created Fraciton will always be in simplified form.
    pub fn new(numerator: i64, denominator: i64) -> Self {
        if denominator < 0 {
            Self {
                num: -numerator,
                den: -denominator,
            }
        } else {
            Self {
                num: numerator,
                den: denominator,
            }
        }
        .simplify()
    }

    /// simplifies the fraction.
    pub fn simplify(self) -> Self {
        if self.num == 0 {
            return Self::default();
        }
        let common_fac = arithmetic::gcd(self.num.abs(), self.den.abs());
        Self {
            num: self.num / common_fac,
            den: self.den / common_fac,
        }
    }

    /// returns the multiplicative inverse of the fraction.
    pub fn inverse(self) -> Self {
        Self::new(self.den, self.num)
    }

    /// Computes the absolute value of faction.
    pub fn abs(self) -> Self {
        Self {
            num: self.num.abs(),
            den: self.den.abs(),
        }
    }

    /// finds the closest fraction to the float with the given denominator.
    ///
    /// # Examples
    ///
    /// ```
    /// use lmath::fraction::Fraction;
    ///
    /// let float = 0.463;
    /// assert_eq!(Fraction::from_float_with_denominator(float, 1000), Fraction::new(463,1000));
    /// ```
    pub fn from_float_with_denominator(float: f64, den: i64) -> Self {
        Self::new((float * den as f64).round() as i64, den)
    }

    /// finds the closest fraction to the given float with denominator less than or equal to the given denominator.
    ///
    /// # Examples
    ///
    /// ```
    /// use lmath::fraction::Fraction;
    ///
    /// let float = 0.3333;
    /// assert_eq!(Fraction::from_float_closest(float, 100), Fraction::new(1, 3));
    /// ```
    pub fn from_float_closest(float: f64, max_denom: i64) -> Self {
        let mut closest = Self::default();
        let mut distance = f64::INFINITY;
        for i in 1..=max_denom {
            let frac = Self::from_float_with_denominator(float, i);
            if (float - f64::from(frac)).abs() < distance {
                closest = frac;
                distance = (float - f64::from(frac)).abs();
            }
        }
        closest.simplify()
    }

    /// finds the closest fraction to the float whose denominator is a power of 2.
    ///
    /// # Examples
    ///
    /// ```
    /// use lmath::fraction::Fraction;
    ///
    /// let float = 0.625;
    /// assert_eq!(Fraction::from_float_closest_p2(float), Fraction::new(5, 8));
    /// ```
    pub fn from_float_closest_p2(float: f64) -> Self {
        let mut closest = Self::default();
        let mut distance = f64::INFINITY;
        for p in 0..63 {
            let i = 1 << p;
            let frac = Self::from_float_with_denominator(float, i);
            if (float - f64::from(frac)).abs() < distance {
                closest = frac;
                distance = (float - f64::from(frac)).abs();
            }
        }
        closest.simplify()
    }

    /// computes the nth power of the fraction.
    pub fn pow(self, n: i64) -> Self {
        match (n, n < 0, n % 2 == 0) {
            (_, true, _) => self.inverse().pow(-n),
            (0, ..) => Self::new(1, 1),
            (1, ..) => self,
            (.., true) => {
                let a = self.pow(n / 2);
                a * a
            }
            (.., false) => self.pow(n - 1) * self,
        }
    }
}

#[cfg(test)]
mod test {
    use super::Fraction;

    #[test]
    fn arithmetic_test() {
        let a = Fraction::new(7, 4);
        let b = Fraction::new(12, 5);
        let c = Fraction::new(-9, -24);
        assert_eq!(-a, Fraction::new(-7, 4));
        assert_eq!(c, Fraction::new(3, 8));
        assert_eq!(a + b, Fraction::new(83, 20));
        assert_eq!(a + c, Fraction::new(17, 8));
        assert_eq!(b + c, Fraction::new(111, 40));
        assert_eq!(a - b, Fraction::new(-13, 20));
        assert_eq!(b - a, Fraction::new(13, 20));
        assert_eq!(a * b, Fraction::new(21, 5));
        assert_eq!(a / b, Fraction::new(35, 48));
    }

    #[test]
    fn from_float_test() {
        assert_eq!(Fraction::from_float_closest(43., 100), Fraction::new(43, 1));
        assert_eq!(
            Fraction::from_float_closest(0.6666, 1000),
            Fraction::new(2, 3)
        );
        assert_eq!(
            Fraction::from_float_closest(std::f64::consts::PI, 1000),
            Fraction::new(355, 113)
        );
    }
}
