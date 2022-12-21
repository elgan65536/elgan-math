#![allow(unused)]
use elgan_math::{complex::Complex, fraction::Fraction};

fn main() {
    println!("{}", Fraction::from_float_closest(2f64.ln(), 1000));
}
