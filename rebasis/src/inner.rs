//! The inner product ⟨φ, ψ⟩ = ∫ φ(x) ψ(x) dx of two elements.
//!
//! Splitting the overlap at every knot of both elements leaves pieces on which both are smooth.
//! Polynomial pieces (at most cubic, so products of degree <= 6) are integrated exactly by the
//! 4-point Gauss–Legendre rule; pieces involving a Gaussian are further cut to at most one
//! standard deviation and integrated with 8 points, which is accurate to rounding.

use crate::element::Element;
use crate::quadrature::{gl4, gl8, integrate};

/// ⟨a, b⟩. `knots` is scratch space, reused between calls to avoid allocation.
///
/// A delta against a function gives the function's value at the delta. Two deltas give 1 if
/// they are at the same point and 0 otherwise, so a set of distinct deltas is orthonormal and
/// projecting onto it is interpolation.
pub fn inner(a: &Element, b: &Element, knots: &mut Vec<f64>) -> f64 {
    match (a, b) {
        (Element::Delta { x: xa }, Element::Delta { x: xb }) => {
            if xa == xb {
                1.0
            } else {
                0.0
            }
        }
        (Element::Delta { x }, other) | (other, Element::Delta { x }) => other.evaluate(*x),
        _ => {
            let lo = a.lower().max(b.lower());
            let hi = a.upper().min(b.upper());
            if hi <= lo {
                return 0.0;
            }
            knots.clear();
            knots.push(lo);
            knots.push(hi);
            a.push_knots(lo, hi, knots);
            b.push_knots(lo, hi, knots);
            knots.sort_unstable_by(f64::total_cmp);
            knots.dedup();

            let sigma = match (a.sigma(), b.sigma()) {
                (Some(s), Some(t)) => Some(s.min(t)),
                (s, t) => s.or(t),
            };
            let f = |x: f64| a.evaluate(x) * b.evaluate(x);
            match sigma {
                None => knots
                    .windows(2)
                    .map(|w| integrate(gl4(), w[0], w[1], f))
                    .sum(),
                Some(sigma) => knots
                    .windows(2)
                    .map(|w| {
                        let n = ((w[1] - w[0]) / sigma).ceil().max(1.0) as usize;
                        let h = (w[1] - w[0]) / n as f64;
                        (0..n)
                            .map(|k| {
                                let x0 = w[0] + k as f64 * h;
                                integrate(gl8(), x0, x0 + h, f)
                            })
                            .sum::<f64>()
                    })
                    .sum(),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::element::Shape;

    fn ip(a: &Element, b: &Element) -> f64 {
        inner(a, b, &mut Vec::new())
    }

    #[test]
    fn box_box_is_overlap_length() {
        let a = Element::boxcar(0.0, 2.0).unwrap();
        let b = Element::boxcar(1.5, 4.0).unwrap();
        assert!((ip(&a, &b) - 0.5).abs() < 1e-15);
        assert!((ip(&b, &a) - 0.5).abs() < 1e-15);
    }

    #[test]
    fn hat_hat_mass_matrix() {
        // Linear finite-element mass matrix on a uniform grid with spacing h: h·[1/6, 2/3, 1/6]
        let h = 0.3;
        let h0 = Element::hat(0.0, h, 2.0 * h).unwrap();
        let h1 = Element::hat(h, 2.0 * h, 3.0 * h).unwrap();
        assert!((ip(&h0, &h0) - 2.0 * h / 3.0).abs() < 1e-15);
        assert!((ip(&h0, &h1) - h / 6.0).abs() < 1e-15);
    }

    #[test]
    fn hat_box_with_kink_inside_is_exact() {
        // ∫_0^1 x dx over the rising side of a unit hat = 1/2; the box covers [0.25, 1.5]
        let hat = Element::hat(0.0, 1.0, 2.0).unwrap();
        let bx = Element::boxcar(0.25, 1.5).unwrap();
        let expected = (1.0 - 0.25f64.powi(2)) / 2.0 + (1.0 - 0.5f64.powi(2)) / 2.0;
        assert!((ip(&hat, &bx) - expected).abs() < 1e-15);
    }

    #[test]
    fn gaussian_box_matches_erf() {
        let sigma = 0.7;
        let g = Element::gaussian(0.2, sigma, 6.0).unwrap();
        let bx = Element::boxcar(-0.5, 1.1).unwrap();
        let s2 = sigma * std::f64::consts::SQRT_2;
        let expected = sigma
            * (std::f64::consts::PI / 2.0).sqrt()
            * (libm::erf((1.1 - 0.2) / s2) - libm::erf((-0.5 - 0.2) / s2));
        assert!((ip(&g, &bx) - expected).abs() < 1e-14 * expected);
    }

    #[test]
    fn gaussian_gaussian_matches_closed_form() {
        let (s1, s2, c1, c2) = (0.4, 1.3, 0.0, 0.9);
        let g1 = Element::gaussian(c1, s1, 12.0).unwrap();
        let g2 = Element::gaussian(c2, s2, 12.0).unwrap();
        let var = s1 * s1 + s2 * s2;
        let expected = (2.0 * std::f64::consts::PI).sqrt() * s1 * s2 / var.sqrt()
            * (-(c1 - c2) * (c1 - c2) / (2.0 * var)).exp();
        assert!((ip(&g1, &g2) - expected).abs() < 1e-13 * expected);
    }

    #[test]
    fn tabulated_shape_equals_explicit_hat() {
        let shape = Arc::new(Shape::piecewise_linear(&[-1.0, 0.0, 1.0], &[0.0, 1.0, 0.0]).unwrap());
        let tab = Element::piecewise(shape, 2.0, 0.5).unwrap();
        let hat = Element::hat(1.5, 2.0, 2.5).unwrap();
        let other = Element::hat(1.7, 2.1, 3.0).unwrap();
        assert!((ip(&tab, &other) - ip(&hat, &other)).abs() < 1e-15);
        assert!((tab.integral() - 0.5).abs() < 1e-15);
    }

    #[test]
    fn deltas() {
        let d = Element::delta(0.25).unwrap();
        let hat = Element::hat(0.0, 0.5, 1.0).unwrap();
        assert_eq!(ip(&d, &hat), 0.5);
        assert_eq!(ip(&d, &Element::delta(0.25).unwrap()), 1.0);
        assert_eq!(ip(&d, &Element::delta(0.3).unwrap()), 0.0);
    }
}
