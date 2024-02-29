use std::{
    cmp::Ordering,
    fmt::Display,
    ops::{Add, Div, Mul, Neg, Sub},
};

use crate::arithmetic::IntArithmetic;

/// A fraction of the form a/b for integers a and b.
#[derive(Clone, Copy, Debug)]
pub struct Fraction {
    num: i64,
    den: i64,
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
        let gcd = self.den.gcd(rhs.den) as i128;
        let num =
            rhs.den as i128 / gcd * self.num as i128 + self.den as i128 / gcd * rhs.num as i128;
        let den = (self.den as i128).lcm(rhs.den as i128);
        let max = num.abs().max(den.abs());
        let min = num.abs().min(den.abs());
        let mut fact = (max / i64::MAX as i128) + 1;
        if fact > 1 && min % fact != 0 && min % (fact + 1) == 0 {
            fact += 1;
        }
        let new_num = num / fact;
        let new_den = den / fact;
        if new_den == 0 {
            if new_num > 0 {
                Fraction::MAX
            } else {
                Fraction::MIN
            }
        } else {
            Self::new(new_num as i64, new_den as i64)
        }
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
    fn mul(mut self, mut rhs: Self) -> Self::Output {
        let a = self.num.abs().gcd(rhs.den.abs());
        if a > 1 {
            self.num /= a;
            rhs.den /= a;
        }
        let a = rhs.num.abs().gcd(self.den.abs());
        if a > 1 {
            rhs.num /= a;
            self.den /= a;
        }
        let num = self.num as i128 * rhs.num as i128;
        let den = self.den as i128 * rhs.den as i128;
        let max = num.abs().max(den.abs());
        let mut fact = max / i64::MAX as i128 + 1;
        if fact > 1 && num.abs().min(den.abs()) % (fact + 1) == 0 {
            fact += 1;
        }
        let new_num = num / fact;
        let new_den = den / fact;
        if new_den == 0 {
            if new_num >= 0 {
                Fraction::MAX
            } else {
                Fraction::MIN
            }
        } else {
            Self::new(new_num as i64, new_den as i64)
        }
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

impl Eq for Fraction {}

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

impl Fraction {
    /// highest possible Fraction
    pub const MAX: Fraction = Fraction {
        num: i64::MAX,
        den: 1,
    };
    /// lowest possible fraction
    pub const MIN: Fraction = Fraction {
        num: i64::MIN,
        den: 1,
    };
    /// lowest possible positive fraction
    pub const MIN_POSITIVE: Fraction = Fraction {
        num: 1,
        den: i64::MAX,
    };
    /// highest possible negative fraction
    pub const MAX_NEGATIVE: Fraction = Fraction {
        num: -1,
        den: i64::MAX,
    };
    /// approximation of Archimedes' constant (π)
    pub const PI: Fraction = Fraction {
        num: 3378027104,
        den: 1075259423,
    };
    /// approximation of sqrt(π)
    pub const SQRT_PI: Fraction = Fraction {
        num: 1910320744,
        den: 1077783065,
    };
    /// approximation of sqrt(2)
    pub const SQRT_2: Fraction = Fraction {
        num: 1527427483,
        den: 1080054331,
    };
    /// approximation of sqrt(3)
    pub const SQRT_3: Fraction = Fraction {
        num: 1866608895,
        den: 1077687148,
    };
    /// approximation of sqrt(5)
    pub const SQRT_5: Fraction = Fraction {
        num: 2405109757,
        den: 1075597782,
    };
    /// approximation of Euler's number (e)
    pub const E: Fraction = Fraction {
        num: 2927228642,
        den: 1076867237,
    };
    /// approximation of log<sub>2</sub>(10)
    pub const LOG2_10: Fraction = Fraction {
        num: 3571843386,
        den: 1075231999,
    };
    /// approximation of ln(2)
    pub const LN_2: Fraction = Fraction {
        num: 748264996,
        den: 1079518199,
    };
    /// approximation of ln(10)
    pub const LN_10: Fraction = Fraction {
        num: 2479617416,
        den: 1076884161,
    };

    /// Creates a new [`Fraction`] with the given numerator and denominator.
    /// The created Fraciton will always be in simplified form.
    ///
    /// # Examples
    ///
    /// ```
    /// use elgan_math::Fraction;
    /// assert_eq!(Fraction::new(3, 2), Fraction::new(9,6))
    /// ```
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

    /// returns the numerator, assuming simplified form.
    ///
    /// # Examples
    ///
    /// ```
    /// use elgan_math::Fraction;
    /// assert_eq!(Fraction::new(3, 2).num(), 3);
    /// assert_eq!(Fraction::new(8, 6).num(), 4);
    /// ```
    pub const fn num(self) -> i64 {
        self.num
    }

    /// returns the denominator, assuming simplified form.
    ///
    /// # Examples
    ///
    /// ```
    /// use elgan_math::Fraction;
    /// assert_eq!(Fraction::new(3, 2).den(), 2);
    /// assert_eq!(Fraction::new(8, 6).den(), 3);
    /// ```
    pub const fn den(self) -> i64 {
        self.den
    }

    /// simplifies the fraction.
    fn simplify(self) -> Self {
        if self.num == 0 {
            return Self::default();
        }
        let common_fac = self.num.abs().gcd(self.den.abs());
        Self {
            num: self.num / common_fac,
            den: self.den / common_fac,
        }
    }

    /// returns the multiplicative inverse of the fraction.
    ///
    /// # Examples
    ///
    /// ```
    /// use elgan_math::Fraction;
    /// assert_eq!(Fraction::new(2, 3).inverse(), Fraction::new(3, 2))
    /// ```
    pub fn inverse(self) -> Self {
        if self.num < 0 {
            Self {
                num: -self.den,
                den: -self.num,
            }
        } else {
            Self {
                num: self.den,
                den: self.num,
            }
        }
    }

    /// Computes the absolute value of the faction.
    ///
    /// # Examples
    ///
    /// ```
    /// use elgan_math::Fraction;
    /// assert_eq!(Fraction::new(-3, 2).abs(), Fraction::new(3, 2))
    /// ```
    pub fn abs(self) -> Self {
        Self {
            num: self.num.abs(),
            den: self.den.abs(),
        }
    }

    /// Computes `self + rhs`, wrapping around at the
    /// boundary of the type for both numerator and denominator.
    ///
    /// # Examples
    ///
    /// ```
    /// use elgan_math::Fraction;
    /// let frac = Fraction::new(i64::MAX, 2);
    /// assert_eq!(frac.wrapping_add(frac), Fraction::new(-2, 2))
    /// ```
    pub fn wrapping_add(self, rhs: Self) -> Self {
        Self {
            num: (self.num.wrapping_mul(rhs.den)).wrapping_add(rhs.num.wrapping_mul(self.den)),
            den: self.den.wrapping_mul(rhs.den),
        }
        .simplify()
    }

    /// Computes `self - rhs`, wrapping around at the
    /// boundary of the type for both numerator and denominator.
    ///
    /// # Examples
    ///
    /// ```
    /// use elgan_math::Fraction;
    /// let frac = Fraction::new(i64::MAX, 2);
    /// assert_eq!(frac.wrapping_sub(-frac), Fraction::new(-2, 2))
    /// ```
    pub fn wrapping_sub(self, rhs: Self) -> Self {
        self.wrapping_add(-rhs)
    }

    /// Computes `self * rhs`, wrapping around at the
    /// boundary of the type for both numerator and denominator.
    ///
    /// # Examples
    ///
    /// ```
    /// use elgan_math::Fraction;
    /// let frac = Fraction::new(i64::MAX, 2);
    /// assert_eq!(frac.wrapping_mul(Fraction::from(3)), Fraction::new(i64::MAX - 2, 2))
    /// ```
    pub fn wrapping_mul(self, rhs: Self) -> Self {
        Self {
            num: self.num.wrapping_mul(rhs.num),
            den: self.den.wrapping_mul(rhs.den),
        }
        .simplify()
    }

    /// Computes `self / rhs`, wrapping around at the
    /// boundary of the type for both numberator and denominator.
    ///
    /// # Examples
    ///
    /// ```
    /// use elgan_math::Fraction;
    /// let frac = Fraction::new(i64::MAX, 2);
    /// assert_eq!(frac.wrapping_div(Fraction::new(1, 3)), Fraction::new(i64::MAX - 2, 2))
    /// ```
    pub fn wrapping_div(self, rhs: Self) -> Self {
        self.wrapping_mul(rhs.inverse())
    }

    /// finds the closest fraction to the float with the given denominator.
    ///
    /// # Examples
    ///
    /// ```
    /// use elgan_math::Fraction;
    ///
    /// let float = 0.463;
    /// assert_eq!(Fraction::from_float_with_denominator(float, 20), Fraction::new(9, 20));
    /// assert_eq!(Fraction::from_float_with_denominator(float, 1000), Fraction::new(463, 1000));
    /// ```
    pub fn from_float_with_denominator(float: f64, den: i64) -> Self {
        Self::new((float * den as f64).round() as i64, den)
    }

    /// finds the closest fraction to the given float with denominator less than or equal to the given denominator.
    /// This can be very slow if `max_denom` is high.
    ///
    /// # Examples
    ///
    /// ```
    /// use elgan_math::Fraction;
    ///
    /// let float = 0.3333;
    /// assert_eq!(Fraction::from_float_closest(float, 100), Fraction::new(1, 3));
    /// ```
    pub fn from_float_closest(float: f64, max_denom: i64) -> Self {
        let mut closest = Self::default();
        let mut distance = f64::INFINITY;
        for i in 1.max(max_denom / 2)..=max_denom {
            let frac = Self::from_float_with_denominator(float, i);
            if (float - f64::from(frac)).abs() < distance {
                closest = frac;
                distance = (float - f64::from(frac)).abs();
                if distance == 0.0 {
                    break;
                }
            }
        }
        closest
    }

    /// finds the closest fraction to the float whose denominator is a power of 2.
    ///
    /// # Examples
    ///
    /// ```
    /// use elgan_math::Fraction;
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
                if distance == 0.0 {
                    break;
                }
            }
        }
        closest
    }

    /// computes the nth power of the fraction.
    ///
    /// # Examples
    ///
    /// ```
    /// use elgan_math::Fraction;
    /// assert_eq!(Fraction::new(3, 2).pow(4), Fraction::new(81, 16))
    /// ```
    pub fn pow(self, n: i64) -> Self {
        match n {
            0 => Self::new(1, 1),
            1 => self,
            _ if n < 0 => self.inverse().pow(-n),
            _ if n % 2 == 1 => {
                let a = self.pow(n / 2);
                a * a * self
            }
            _ => {
                let a = self.pow(n / 2);
                a * a
            }
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
        assert_eq!(c.num(), 3);
        assert_eq!(c.den(), 8);
        assert_eq!(-a, Fraction::new(-7, 4));
        assert_eq!(a.inverse(), Fraction::new(4, 7));
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
            Fraction::from_float_closest(std::f64::consts::PI, 500),
            Fraction::new(355, 113)
        );
    }
}
