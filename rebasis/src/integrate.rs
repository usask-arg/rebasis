//! Inner products ⟨φ, ψ⟩_w = ∫ φ(x) ψ(x) w(x) dx and integrals of elements.
//!
//! The overlap is split at every knot of both elements, leaving pieces on which everything is
//! smooth. Without profiles or a weight, the integrand is a polynomial of degree <= 6 and the
//! 4-point Gauss–Legendre rule is exact. Otherwise each piece is marched in sub-intervals no
//! wider than the profiles' (and weight's) step limits and integrated with 8 points, which is
//! accurate to rounding.

use crate::element::Element;
use crate::error::Result;
use crate::profile::Parameter;
use crate::quadrature::{gl4, gl8, integrate};

/// The weight w(x) of the inner product.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum Weight {
    #[default]
    Uniform,
    /// cos(scale · x), e.g. area weighting in latitude with scale = π/180 for degrees.
    Cosine { scale: f64 },
}

impl Weight {
    /// Area weighting for a latitude coordinate in degrees.
    pub fn cos_latitude_degrees() -> Self {
        Weight::Cosine {
            scale: std::f64::consts::PI / 180.0,
        }
    }

    #[inline]
    pub fn value(&self, x: f64) -> f64 {
        match self {
            Weight::Uniform => 1.0,
            Weight::Cosine { scale } => (scale * x).cos(),
        }
    }

    #[inline]
    pub fn slope(&self, x: f64) -> f64 {
        match self {
            Weight::Uniform => 0.0,
            Weight::Cosine { scale } => -scale * (scale * x).sin(),
        }
    }

    fn step(&self) -> Option<f64> {
        match self {
            Weight::Uniform => None,
            Weight::Cosine { scale } => Some(0.25 / scale.abs()),
        }
    }
}

/// Pieces no longer than this fraction of the step limit use the 4-point rule.
const SHORT_PIECE: f64 = 0.05;

/// The largest sub-interval starting at x and moving right that keeps every profile within its
/// step limit (the limit is nondecreasing in the distance from the center, so it is evaluated
/// at the end of the sub-interval nearest the center).
#[inline]
fn step_at(elements: &[&Element], weight: Weight, x: f64) -> f64 {
    let mut h = weight.step().unwrap_or(f64::INFINITY);
    for e in elements {
        if let Some((p, c)) = e.as_profile() {
            let s = if x >= c {
                p.step(x - c)
            } else {
                let d = c - x;
                p.step((d - p.step(d)).max(0.0))
            };
            h = h.min(s);
        }
    }
    h
}

/// ∫ Π elements(x) · w(x) dx over [lo, hi] given the sorted knots inside it (inclusive of lo, hi).
fn integrate_product(elements: &[&Element], weight: Weight, knots: &[f64]) -> f64 {
    integrate_with(elements, weight, knots, |x| {
        elements.iter().map(|e| e.evaluate(x)).product::<f64>()
    })
}

/// ∫ g(x) w(x) dx over the knots, with the step control of `elements` (g must be smooth
/// between knots, on the scale of the elements' profiles).
fn integrate_with(
    elements: &[&Element],
    weight: Weight,
    knots: &[f64],
    g: impl Fn(f64) -> f64,
) -> f64 {
    let f = |x: f64| g(x) * weight.value(x);
    let smooth = weight == Weight::Uniform && elements.iter().all(|e| e.as_profile().is_none());
    if smooth {
        return knots
            .windows(2)
            .map(|w| integrate(gl4(), w[0], w[1], f))
            .sum();
    }
    let mut total = 0.0;
    for w in knots.windows(2) {
        let (mut x, end) = (w[0], w[1]);
        while x < end {
            let allowed = step_at(elements, weight, x);
            // avoid leaving a sliver at the end
            let h = if x + 1.05 * allowed >= end {
                end - x
            } else {
                allowed
            };
            // On a piece much shorter than the profile's scale (a fine grid under a wide line
            // shape) the integrand is nearly polynomial and 4 points are already exact to
            // rounding; see `short_pieces_need_fewer_points` below.
            let rule = if h <= SHORT_PIECE * allowed {
                gl4()
            } else {
                gl8()
            };
            total += integrate(rule, x, x + h, f);
            x += h;
        }
    }
    total
}

fn sorted_knots(elements: &[&Element], lo: f64, hi: f64, knots: &mut Vec<f64>) {
    knots.clear();
    knots.push(lo);
    knots.push(hi);
    for e in elements {
        e.push_knots(lo, hi, knots);
    }
    knots.sort_unstable_by(f64::total_cmp);
    knots.dedup();
}

/// ⟨a, b⟩_w. `knots` is scratch space, reused between calls to avoid allocation.
///
/// A delta against a function gives the (weighted) function value at the delta. Two deltas
/// give 1 if they are at the same point and 0 otherwise, so a set of distinct deltas is
/// orthonormal and projecting onto it is interpolation.
pub fn inner(a: &Element, b: &Element, weight: Weight, knots: &mut Vec<f64>) -> f64 {
    match (a, b) {
        (Element::Delta { x: xa }, Element::Delta { x: xb }) => {
            if xa == xb {
                1.0
            } else {
                0.0
            }
        }
        (Element::Delta { x }, other) | (other, Element::Delta { x }) => {
            other.evaluate(*x) * weight.value(*x)
        }
        _ => {
            let lo = a.lower().max(b.lower());
            let hi = a.upper().min(b.upper());
            if hi <= lo {
                return 0.0;
            }
            sorted_knots(&[a, b], lo, hi, knots);
            integrate_product(&[a, b], weight, knots)
        }
    }
}

/// ∫ φ(x) w(x) dx (w(x₀) for a delta at x₀).
pub fn integral(e: &Element, weight: Weight, knots: &mut Vec<f64>) -> f64 {
    if let Element::Delta { x } = e {
        return weight.value(*x);
    }
    if weight == Weight::Uniform {
        if let Some(v) = e.polynomial_integral() {
            return v;
        }
    }
    sorted_knots(&[e], e.lower(), e.upper(), knots);
    integrate_product(&[e], weight, knots)
}

/// ∂/∂p ⟨k, b⟩_w for a parameter p of the kernel k. `jumps` must be `k.jumps(p)`.
pub fn inner_derivative(
    k: &Element,
    b: &Element,
    p: Parameter,
    weight: Weight,
    jumps: &[(f64, f64)],
    knots: &mut Vec<f64>,
) -> Result<f64> {
    if let Element::Delta { x } = k {
        // ⟨δ_c, b⟩ = b(c) w(c)
        return Ok(match p {
            // two deltas: a Kronecker delta, flat in c
            Parameter::Center => match b {
                Element::Delta { .. } => 0.0,
                _ => b.slope(*x) * weight.value(*x) + b.evaluate(*x) * weight.slope(*x),
            },
            Parameter::Scale => 0.0,
            other => {
                return crate::error::invalid(format!("a delta has no parameter {other:?}"));
            }
        });
    }
    if let Element::Delta { x } = b {
        return Ok(k.d_smooth(*x, p)? * weight.value(*x));
    }
    let point: f64 = jumps
        .iter()
        .filter(|(x, _)| *x >= b.lower() && *x <= b.upper())
        .map(|(x, a)| a * b.evaluate(*x) * weight.value(*x))
        .sum();
    let lo = k.lower().max(b.lower());
    let hi = k.upper().min(b.upper());
    if hi <= lo {
        return Ok(point);
    }
    sorted_knots(&[k, b], lo, hi, knots);
    // d_smooth only errors for an inapplicable parameter, which `jumps` has already rejected
    let smooth = integrate_with(&[k, b], weight, knots, |x| {
        k.d_smooth(x, p).unwrap_or(0.0) * b.evaluate(x)
    });
    Ok(smooth + point)
}

/// ∂/∂p ∫ k w for a parameter p of the kernel k. `jumps` must be `k.jumps(p)`.
pub fn integral_derivative(
    k: &Element,
    p: Parameter,
    weight: Weight,
    jumps: &[(f64, f64)],
    knots: &mut Vec<f64>,
) -> Result<f64> {
    if let Element::Delta { x } = k {
        return Ok(match p {
            Parameter::Center => weight.slope(*x),
            _ => 0.0,
        });
    }
    let point: f64 = jumps.iter().map(|(x, a)| a * weight.value(*x)).sum();
    sorted_knots(&[k], k.lower(), k.upper(), knots);
    let smooth = integrate_with(&[k], weight, knots, |x| k.d_smooth(x, p).unwrap_or(0.0));
    Ok(smooth + point)
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::element::Shape;
    use crate::profile::Profile;

    fn ip(a: &Element, b: &Element) -> f64 {
        inner(a, b, Weight::Uniform, &mut Vec::new())
    }

    fn integral_of(e: &Element) -> f64 {
        integral(e, Weight::Uniform, &mut Vec::new())
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
    fn lorentzian_integral_matches_arctan() {
        let g = 0.05;
        let cutoff = 1000.0;
        let e = Element::profile(Profile::lorentzian(g).unwrap(), 3.0, cutoff).unwrap();
        let expected = 2.0 * g * cutoff.atan();
        assert!((integral_of(&e) - expected).abs() < 1e-14 * expected);
        // against an off-center box: γ (atan((b − c)/γ) − atan((a − c)/γ))
        let bx = Element::boxcar(3.01, 3.5).unwrap();
        let expected = g * ((0.5f64 / g).atan() - (0.01f64 / g).atan());
        assert!((ip(&e, &bx) - expected).abs() < 1e-14 * expected);
    }

    #[test]
    fn voigt_area_matches_gaussian_times_lorentzian_norms() {
        // ∫ V = 1 for the normalized Voigt, so ∫ (peak-1 Voigt) = 1 / V(0), V(0) = Re w(iy) / (σ√2π)
        let (sigma, gamma) = (0.2, 0.01);
        let p = Profile::voigt(sigma, gamma).unwrap();
        let cutoff = 1e5;
        let e = Element::profile(p, 0.0, cutoff).unwrap();
        let y = gamma / (sigma * std::f64::consts::SQRT_2);
        let v0 =
            crate::special::faddeeva_re(0.0, y) / (sigma * (2.0 * std::f64::consts::PI).sqrt());
        // mass beyond the cutoff R·hwhm is ≈ the Lorentzian tail 2γ/(πR·hwhm) of the unit-area Voigt
        let r = cutoff * p.hwhm();
        let tail = 2.0 * gamma / (std::f64::consts::PI * r);
        let expected = (1.0 - tail) / v0;
        assert!((integral_of(&e) - expected).abs() < 1e-9 * expected);
    }

    #[test]
    fn super_gaussian_with_fractional_power_integrates_accurately() {
        // ∫ exp(−ln2 |t/w|^p) dt = 2 w Γ(1 + 1/p) / ln2^(1/p)
        for p in [1.0, 1.5, 2.0, 3.0, 8.0] {
            let w = 0.3;
            let e =
                Element::profile(Profile::super_gaussian(w, w, p).unwrap(), 1.0, 200.0).unwrap();
            let expected =
                2.0 * w * libm::tgamma(1.0 + 1.0 / p) / std::f64::consts::LN_2.powf(1.0 / p);
            assert!(
                (integral_of(&e) - expected).abs() < 1e-13 * expected,
                "p={p}"
            );
        }
    }

    #[test]
    fn short_pieces_need_fewer_points() {
        // Hats and clamped cubic B-splines are partitions of unity, so Σ_j ⟨K, B_j⟩ over a fine
        // grid (4-point rule on every short piece) must equal ∫ K (8-point rule on long pieces),
        // and likewise for ∫ K·x via the B-spline / hat reproduction of linear functions.
        use crate::ops::inner_products;
        use crate::profile::Apodization;
        use crate::space::Space;
        let profiles = [
            Profile::gaussian(0.05).unwrap(),
            Profile::lorentzian(0.02).unwrap(),
            Profile::voigt(0.04, 0.02).unwrap(),
            Profile::pseudo_voigt(0.03, 0.7).unwrap(),
            Profile::super_gaussian(0.04, 0.06, 2.5).unwrap(),
            Profile::fts(20.0, Apodization::HAMMING).unwrap(),
        ];
        let nodes: Vec<f64> = (0..=40_000)
            .map(|i| -2.0 + 4.0 * i as f64 / 40_000.0)
            .collect();
        for basis in [
            Space::hats(&nodes).unwrap(),
            Space::bsplines_clamped(
                &nodes[..=20_000]
                    .iter()
                    .step_by(2)
                    .copied()
                    .collect::<Vec<_>>(),
                3,
            )
            .unwrap(),
        ] {
            for p in profiles {
                let k = Space::profiles(&[-1.013], &[p], 15.0).unwrap();
                assert!(
                    k.lower()[0] > -1.99 && k.upper()[0] < -0.01,
                    "{p:?} must fit the grid"
                );
                let w = inner_products(&k, &basis, Weight::Uniform);
                let total: f64 = w.row(0).1.iter().sum();
                let reference = integral_of(&k.elements()[0]);
                // the tolerance is summation rounding over ~10³ terms; the 8-point rule everywhere
                // gives the same differences
                assert!(
                    (total - reference).abs() < 5e-14 * reference.abs(),
                    "{p:?}: {total} vs {reference}"
                );
            }
        }
    }

    #[test]
    fn cosine_weight() {
        // ∫_{-90}^{90} cos(lat) dlat (degrees) = 360/π
        let bx = Element::boxcar(-90.0, 90.0).unwrap();
        let w = Weight::cos_latitude_degrees();
        let got = integral(&bx, w, &mut Vec::new());
        assert!((got - 360.0 / std::f64::consts::PI).abs() < 1e-12);
        // box ⊗ box weighted = ∫ over the overlap
        let a = Element::boxcar(10.0, 40.0).unwrap();
        let b = Element::boxcar(30.0, 60.0).unwrap();
        let s = std::f64::consts::PI / 180.0;
        let expected = ((40.0 * s).sin() - (30.0 * s).sin()) / s;
        assert!((inner(&a, &b, w, &mut Vec::new()) - expected).abs() < 1e-12);
    }

    #[test]
    fn tabulated_shape_equals_explicit_hat() {
        let shape = Arc::new(Shape::piecewise_linear(&[-1.0, 0.0, 1.0], &[0.0, 1.0, 0.0]).unwrap());
        let tab = Element::piecewise(shape, 2.0, 0.5).unwrap();
        let hat = Element::hat(1.5, 2.0, 2.5).unwrap();
        let other = Element::hat(1.7, 2.1, 3.0).unwrap();
        assert!((ip(&tab, &other) - ip(&hat, &other)).abs() < 1e-15);
        assert!((integral_of(&tab) - 0.5).abs() < 1e-15);
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
