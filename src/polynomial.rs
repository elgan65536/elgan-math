use std::ops::{Add, Div, Mul, Neg, Sub};

use crate::FloatArithmetic;

/// a ploynomial
#[derive(Clone, PartialEq, Debug)]
pub struct Polynomial {
    coefficients: Vec<f64>,
}

impl Neg for Polynomial {
    type Output = Self;

    fn neg(self) -> Self::Output {
        Self::new(&self.coefficients.iter().map(|x| -x).collect::<Vec<_>>())
    }
}

impl Add for Polynomial {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        let mut result = vec![];
        for i in 0..self.len().max(rhs.len()) {
            result.push(
                self.coefficients.get(i).unwrap_or(&0.0) + rhs.coefficients.get(i).unwrap_or(&0.0),
            )
        }
        Self::new(&result)
    }
}

impl Add<f64> for Polynomial {
    type Output = Self;

    fn add(self, rhs: f64) -> Self::Output {
        self + Self::new(&[rhs])
    }
}

impl Sub for Polynomial {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        self + -rhs
    }
}

impl Sub<f64> for Polynomial {
    type Output = Self;

    fn sub(self, rhs: f64) -> Self::Output {
        self - Self::new(&[rhs])
    }
}

impl Mul for Polynomial {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        let mut result = vec![0.0; self.len() + rhs.len()];
        for i in 0..self.len() {
            for j in 0..rhs.len() {
                result[i + j] = self.coefficients[i].mul_add(rhs.coefficients[j], result[i + j])
            }
        }
        Self::new(&result)
    }
}

impl Mul<f64> for Polynomial {
    type Output = Self;

    fn mul(self, rhs: f64) -> Self::Output {
        Self::new(
            &self
                .coefficients
                .iter()
                .map(|x| x * rhs)
                .collect::<Vec<_>>(),
        )
    }
}

impl Div<f64> for Polynomial {
    type Output = Self;

    fn div(self, rhs: f64) -> Self::Output {
        Self::new(
            &self
                .coefficients
                .iter()
                .map(|x| x / rhs)
                .collect::<Vec<_>>(),
        )
    }
}

impl Polynomial {
    /// removes leading zeroes
    fn remove_unnecessary(mut self) -> Self {
        while self.coefficients.last() == Some(&0.0) {
            self.coefficients.pop();
        }
        self
    }

    /// creates a new [`Polynomial`] from the given coefficients.
    /// The coefficients should be in order from lowest degree to highest degree term.
    ///
    /// # Examples
    /// ```
    /// use elgan_math::Polynomial;
    ///
    /// let polynomial = Polynomial::new(&[3.0, 2.0, 1.0]); // P(x) = x^2 + 2x + 3
    /// ```
    pub fn new(coefficients: &[f64]) -> Self {
        Self {
            coefficients: coefficients.to_vec(),
        }
        .remove_unnecessary()
    }

    /// creates a [`Polynomial`] which evaluates to 0 at every input which is in `roots`, and
    /// evaluates to a nonzero number for any other input.
    ///
    /// # Examples
    ///  
    /// ```
    /// use elgan_math::Polynomial;
    ///
    /// let polynomial = Polynomial::from_roots(&[1.0, 2.0]);
    /// assert_eq!(polynomial.eval(1.0), 0.0);
    /// assert_eq!(polynomial.eval(2.0), 0.0);
    /// assert!(polynomial.eval(0.0) != 0.0);
    /// assert!(polynomial.eval(1.5) != 0.0);
    /// ```
    pub fn from_roots(roots: &[f64]) -> Self {
        let mut result = Polynomial::new(&[1.]);
        for root in roots {
            result = result * Polynomial::new(&[-root, 1.])
        }
        result
    }

    /// returns the number of coefficients in the polynomial.
    ///
    /// # Examples
    ///  
    /// ```
    /// use elgan_math::Polynomial;
    ///
    /// let polynomial = Polynomial::new(&[3.0, 2.0, 1.0]); // P(x) = x^2 + 2x + 3
    /// assert_eq!(polynomial.len(), 3);
    /// ```
    #[allow(clippy::all)]
    pub fn len(&self) -> usize {
        self.coefficients.len()
    }

    /// returns the highest exponent in the polynomial.
    ///
    /// # Examples
    ///  
    /// ```
    /// use elgan_math::Polynomial;
    ///
    /// let polynomial = Polynomial::new(&[3.0, 2.0, 1.0]); // P(x) = x^2 + 2x + 3
    /// assert_eq!(polynomial.degree(), 2);
    /// ```
    pub fn degree(&self) -> usize {
        self.coefficients.len().max(1) - 1
    }

    /// returns the derivative of the polynomial
    ///
    /// # Examples
    ///  
    /// ```
    /// use elgan_math::Polynomial;
    ///
    /// let polynomial = Polynomial::new(&[3.0, 2.0, 1.0]); // P(x) = x^2 + 2x + 3
    /// assert_eq!(polynomial.derivative(), Polynomial::new(&[2., 2.])); // P'(x) = 2x + 2
    /// ```
    pub fn derivative(&self) -> Self {
        let mut result = vec![0.0; self.len() - 1];
        for i in 1..self.len() {
            result[i - 1] = self.coefficients[i] * i as f64;
        }
        Self::new(&result)
    }

    /// returns the antiderivative of the polynomial, omitting the constant.
    ///
    /// # Examples
    ///  
    /// ```
    /// use elgan_math::Polynomial;
    ///
    /// let polynomial = Polynomial::new(&[5.0, 4.0, 3.0]); // P(x) = 3x^2 + 4x + 5
    /// assert_eq!(polynomial.integral(), Polynomial::new(&[0.0, 5.0, 2.0, 1.0])); // P(x) = x^3 + 2x^2 + 5x (+C)
    /// ```
    pub fn integral(&self) -> Self {
        let mut result = vec![0.0; self.len() + 1];
        for i in 0..self.len() {
            result[i + 1] = self.coefficients[i] / (i + 1) as f64;
        }
        Self::new(&result)
    }

    /// evaluates the polynomial at the given input value.
    ///
    /// # Examples
    ///  
    /// ```
    /// use elgan_math::Polynomial;
    ///
    /// let polynomial = Polynomial::new(&[3.0, 2.0, 1.0]); // P(x) = x^2 + 2x + 3
    /// assert_eq!(polynomial.eval(0.0), 3.0);
    /// assert_eq!(polynomial.eval(1.0), 6.0);
    /// assert_eq!(polynomial.eval(2.0), 11.0);
    /// ```
    pub fn eval(&self, x: f64) -> f64 {
        let mut total = *self.coefficients.last().unwrap_or(&0.0);
        for &i in self.coefficients.iter().rev().skip(1) {
            total = total.mul_add(x, i);
        }
        total
    }

    /// finds roots of the polynomial using Newton's method.
    /// the `guess` input is used as the initial guess for iteration.
    /// Typically, this function will return the root closest to the provided guess if there are multiple roots.
    ///
    /// # Examples
    ///
    /// ```
    /// use elgan_math::Polynomial;
    /// use elgan_math::FloatArithmetic;
    ///
    /// let polynomial = Polynomial::new(&[-2.0, -1.0, 1.0]); // P(x) = x^2 - x - 2
    /// assert_eq!(polynomial.eval(-1.0), 0.0);
    /// assert_eq!(polynomial.eval(2.0), 0.0);
    /// assert!(polynomial.root_newton(3.0).close_to(2.0));
    /// assert!(polynomial.root_newton(-3.0).close_to(-1.0));
    /// ```
    pub fn root_newton(&self, mut guess: f64) -> f64 {
        for _ in 0..40 {
            if self.eval(guess).close_to(0.) {
                return guess;
            }
            guess = guess - self.eval(guess) / self.derivative().eval(guess)
        }
        guess
    }

    /// from the given arrays of x and y coordinates, returns a polynomial P such that P(x[i]) == y[i] for all values in the array.
    /// Returns `None` if the arrays are different lengths.
    /// Also returns `None` if there are duplicate x-values.
    ///
    /// #Examples
    /// ```
    /// use elgan_math::Polynomial;
    ///
    /// let p = Polynomial::from_points(&[0.0, 1.0, 2.0], &[0.0, 1.0, 4.0]).unwrap(); // points (0,0), (1,1), (2,4)
    /// assert_eq!(p, Polynomial::new(&[0.0, 0.0, 1.0])); // polynomial is P(x) = x^2
    ///
    /// assert_eq!(Polynomial::from_points(&[0.0, 1.0, 2.0], &[0.0, 1.0]), None);
    /// assert_eq!(Polynomial::from_points(&[0.0, 1.0, 1.0], &[0.0, 1.0, 4.0]), None);
    /// ```
    pub fn from_points(x: &[f64], y: &[f64]) -> Option<Self> {
        if x.len() != y.len() {
            None?
        }
        // need to convert floats to u64 because f64 is not Ord
        let mut set = std::collections::HashSet::new();
        for i in x {
            if set.contains(&i.to_bits()) {
                None?
            }
            set.insert(i.to_bits());
        }
        let len = x.len();
        let mut output = Self::new(&[]);
        for i in 0..len {
            let component = Self::from_roots(
                &x.iter()
                    .enumerate()
                    .filter(|(index, _)| index != &i)
                    .map(|(_, &item)| item)
                    .collect::<Vec<_>>(),
            );
            let factor = component.eval(x[i]) / y[i];
            output = output + (component / factor)
        }
        Some(output)
    }

    /// returns the coefficients of the polynomial, starting with the degree 0 term.
    ///
    /// # Examples
    /// ```
    /// use elgan_math::Polynomial;
    ///
    /// let polynomial = Polynomial::new(&[3.0, 2.0, 1.0]); // P(x) = x^2 + 2x + 3
    /// assert_eq!(polynomial.coefficients(), &[3.0, 2.0, 1.0])
    /// ```
    pub fn coefficients(&self) -> &[f64] {
        &self.coefficients
    }

    /// returns true if the two polynomials are equivalent within a floating-point rounding error.
    /// Use this instead of `==` if you want to accomodate for rounding errors.
    ///
    /// # Examples
    /// ```
    /// use elgan_math::Polynomial;
    ///
    /// let p1 = Polynomial::new(&[1.0, 1.0]);
    /// let p2 = Polynomial::new(&[1.0000000000000001, 0.9999999999999999]);
    /// assert!(p1.close_to(&p2));
    ///
    /// let p1 = Polynomial::new(&[1.0, 1.0]);
    /// let p2 = Polynomial::new(&[1.0, 1.0, 0.00000000000000000001]);
    /// assert!(p1.close_to(&p2));
    /// ```
    pub fn close_to(&self, rhs: &Self) -> bool {
        for i in 0..(self.len().max(rhs.len())) {
            if !self
                .coefficients
                .get(i)
                .unwrap_or(&0.)
                .close_to(*rhs.coefficients.get(i).unwrap_or(&0.))
            {
                return false;
            }
        }
        true
    }
}

#[cfg(test)]
mod test {
    use crate::Polynomial;

    #[test]
    pub fn ops_test() {
        assert_eq!(
            -Polynomial::new(&[3., 1., 4., 1.]),
            Polynomial::new(&[-3., -1., -4., -1.])
        );
        assert_eq!(
            Polynomial::new(&[]) + Polynomial::new(&[3., 1., 4., 1.]),
            Polynomial::new(&[3., 1., 4., 1.])
        );
        assert_eq!(
            Polynomial::new(&[3., 1., 4., 1.]) + Polynomial::new(&[]),
            Polynomial::new(&[3., 1., 4., 1.])
        );
        assert_eq!(
            Polynomial::new(&[]) * Polynomial::new(&[3., 1., 4., 1.]),
            Polynomial::new(&[])
        );
        assert_eq!(
            Polynomial::new(&[3., 1., 4., 1.]) * Polynomial::new(&[]),
            Polynomial::new(&[])
        );
        assert_eq!(
            Polynomial::new(&[3., 1., 4., 1.]) + Polynomial::new(&[2., 7., 1., 8.]),
            Polynomial::new(&[5., 8., 5., 9.])
        );
        assert_eq!(
            Polynomial::new(&[3., 1., 4., 1.]) + Polynomial::new(&[2.]),
            Polynomial::new(&[5., 1., 4., 1.])
        );
        assert_eq!(
            Polynomial::new(&[3., 1., 4., 1.]) * Polynomial::new(&[2., 7., 1., 8.]),
            Polynomial::new(&[6., 23., 18., 55., 19., 33., 8.])
        );
        assert_eq!(
            Polynomial::new(&[3., 1., 4., 1.]) * Polynomial::new(&[2.]),
            Polynomial::new(&[6., 2., 8., 2.])
        );
    }

    #[test]
    pub fn evaluation_test() {
        let poly = Polynomial::new(&[8., 5., 3., 2.]);
        assert_eq!(poly.eval(0.), 8.);
        assert_eq!(poly.eval(1.), 18.);
        assert_eq!(poly.eval(2.), 46.);
    }

    #[test]
    pub fn eq_test() {
        assert_eq!(
            Polynomial::new(&[8., 5., 3., 2.]),
            Polynomial::new(&[8., 5., 3., 2., 0., 0.]),
        );
        assert_ne!(
            Polynomial::new(&[8., 5., 3., 2.]),
            Polynomial::new(&[0., 0., 8., 5., 3., 2.,]),
        );
    }

    #[test]
    pub fn roots_test() {
        assert_eq!(Polynomial::from_roots(&[1., 5., 7.5]).eval(1.), 0.);
        assert_eq!(Polynomial::from_roots(&[1., 5., 7.5]).eval(5.), 0.);
        assert_eq!(Polynomial::from_roots(&[1., 5., 7.5]).eval(7.5), 0.);
        assert_ne!(Polynomial::from_roots(&[1., 5., 7.5]).eval(0.), 0.);
        assert_ne!(Polynomial::from_roots(&[1., 5., 7.5]).eval(-1.), 0.);
        assert_ne!(Polynomial::from_roots(&[1., 5., 7.5]).eval(2.5), 0.);
    }

    #[test]
    pub fn interp_test() {
        assert_eq!(
            Polynomial::from_points(&[0.,], &[1.]).unwrap(),
            Polynomial::new(&[1.])
        );
        assert_eq!(
            Polynomial::from_points(&[0., 1.], &[1., 2.]).unwrap(),
            Polynomial::new(&[1., 1.])
        );
        assert!(Polynomial::from_points(&[0., 1., 3.], &[1., 2., 4.])
            .unwrap()
            .close_to(&Polynomial::new(&[1., 1.])));
        assert!(Polynomial::from_points(&[0., 1., 2.], &[1., 2., 4.])
            .unwrap()
            .close_to(&Polynomial::new(&[1., 0.5, 0.5])));
    }
}
