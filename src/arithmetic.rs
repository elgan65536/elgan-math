/// Arithmetic operations for integers
pub trait IntArithmetic {
    /// computes the greatest common factor of the two numbers.
    ///
    /// # Examples
    /// ```
    /// use elgan_math::IntArithmetic;
    /// assert_eq!(14.gcd(21), 7);
    /// ```
    fn gcd(self, rhs: Self) -> Self;

    /// computes the lowest common multiple of the two numbers.
    ///
    /// # Examples
    /// ```
    /// use elgan_math::IntArithmetic;
    /// assert_eq!(14.lcm(21), 42);
    /// ```
    fn lcm(self, rhs: Self) -> Self;

    /// computes the greatest factor of `self` which is not a factor of `rhs`.
    ///
    /// # Examples
    /// ```
    /// use elgan_math::IntArithmetic;
    /// assert_eq!(14.uncommon_factor(21), 2);
    /// ```
    fn uncommon_factor(self, rhs: Self) -> Self;
}

/// Arithmetic operations for floating point numbers
pub trait FloatArithmetic {
    /// returns true if the two numbers are equivalent within a floating-point rounding error.
    /// Use this instead of `==` if you want to accomodate for rounding errors.
    ///
    /// # Examples
    /// ```
    /// use elgan_math::FloatArithmetic;
    ///
    /// assert!(1.0.close_to(0.9999999999999999))
    /// ```
    fn close_to(self, rhs: Self) -> bool;
}

macro_rules! int_arithmetic_impl {
    ($Int: ty) => {
        impl IntArithmetic for $Int {
            fn gcd(self, rhs: Self) -> Self {
                match (self <= rhs, self == 0) {
                    (false, _) => rhs.gcd(self),
                    (true, true) => rhs,
                    (true, false) => (rhs % self).gcd(self),
                }
            }

            fn lcm(self, rhs: Self) -> Self {
                self / self.gcd(rhs) * rhs
            }

            fn uncommon_factor(self, rhs: Self) -> Self {
                self / self.gcd(rhs)
            }
        }
    };
}

int_arithmetic_impl!(i8);
int_arithmetic_impl!(i16);
int_arithmetic_impl!(i32);
int_arithmetic_impl!(i64);
int_arithmetic_impl!(i128);
int_arithmetic_impl!(isize);
int_arithmetic_impl!(u8);
int_arithmetic_impl!(u16);
int_arithmetic_impl!(u32);
int_arithmetic_impl!(u64);
int_arithmetic_impl!(u128);
int_arithmetic_impl!(usize);

macro_rules! float_arithmetic_impl {
    ($Float: ty) => {
        impl FloatArithmetic for $Float {
            fn close_to(self, rhs: Self) -> bool {
                (self - rhs).abs()
                    <= (self.abs().max(rhs.abs()) * <$Float>::EPSILON * 2.0).max(<$Float>::EPSILON)
            }
        }
    };
}

float_arithmetic_impl!(f32);
float_arithmetic_impl!(f64);
