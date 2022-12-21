use std::ops::{Add, Rem};

pub fn gcd<T>(a: T, b: T) -> T
where
    T: Ord + Rem<Output = T> + Add<Output = T> + Copy,
{
    match (a > b, a + b == b) {
        (true, _) => gcd(b, a),
        (false, true) => b,
        (false, false) => gcd(b % a, a),
    }
}
