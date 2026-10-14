//! Separable stationary covariances, and the exact covariances between weighted averages that
//! optimal estimation needs.
//!
//! A prior covariance C(x, x') = σ² Π_d c_d(x_d − x'_d) over a D-dimensional coordinate. An
//! *atom* along dimension d is an element E with a weight w; it stands for the normalized
//! average ∫ E w f / ∫ E w (a delta stands for the point value). The covariance between two
//! product atoms a, b is σ² Π_d ∫∫ a_d(x) w_a(x) b_d(y) w_b(y) c_d(x − y) dx dy / (N_a N_b),
//! computed exactly per dimension:
//!
//! - delta–delta: c_d(x_a − x_b);
//! - element–delta: an inner product of the element with c_d centred on the point;
//! - element–element: an outer quadrature over the first element of the (exact) inner product
//!   of the second with c_d centred on the quadrature node. The node function is smooth between
//!   the knots of the first element, the knots of the second, and those knots ± the support
//!   radius of c_d, so the pieces between them are integrated to rounding.
//!
//! A periodic dimension (longitude) sums c_d over the images x + jP.

use ndarray::Array2;
use rayon::prelude::*;

use crate::element::Element;
use crate::error::{Result, invalid};
use crate::integrate::{Weight, inner, inner_derivative, integral};
use crate::profile::{MaternOrder, Parameter, Profile};
use crate::quadrature::gl8;
use crate::space::Space;

/// The correlation function c(Δ) of one dimension.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Family {
    /// exp(−Δ² / 2ℓ²): very smooth fields.
    SquaredExponential,
    /// Matérn ½ (exponential), 3/2 or 5/2: rougher fields.
    Matern(MaternOrder),
}

impl Family {
    pub fn by_name(name: &str) -> Result<Self> {
        Ok(match name {
            "squared_exponential" | "gaussian" => Family::SquaredExponential,
            "exponential" | "matern12" => Family::Matern(MaternOrder::Half),
            "matern32" => Family::Matern(MaternOrder::ThreeHalves),
            "matern52" => Family::Matern(MaternOrder::FiveHalves),
            other => return invalid(format!("unknown covariance family {other:?}")),
        })
    }
}

/// c(Δ) along one dimension, optionally periodic.
#[derive(Debug, Clone)]
pub struct Covariance1d {
    profile: Profile,
    /// Truncation in units of `length`, where c has fallen below 1e-17.
    cutoff: f64,
    /// The support radius cutoff · length.
    radius: f64,
    length: f64,
    period: Option<f64>,
    /// Quadrature step for the outer integral.
    step: f64,
}

impl Covariance1d {
    pub fn new(family: Family, length: f64, period: Option<f64>) -> Result<Self> {
        if !(length.is_finite() && length > 0.0) {
            return invalid(format!(
                "correlation length must be finite and > 0, got {length}"
            ));
        }
        if let Some(p) = period {
            if !(p.is_finite() && p > 0.0) {
                return invalid(format!("period must be finite and > 0, got {p}"));
            }
        }
        let (profile, rate) = match family {
            Family::SquaredExponential => (Profile::gaussian(length)?, 1.0),
            Family::Matern(order) => {
                let rate = match order {
                    MaternOrder::Half => 1.0,
                    MaternOrder::ThreeHalves => 3f64.sqrt(),
                    MaternOrder::FiveHalves => 5f64.sqrt(),
                };
                (Profile::matern(order, length)?, rate)
            }
        };
        let mut cutoff = 1.0;
        while profile.value(cutoff * length) > 1e-17 {
            cutoff += 0.5;
        }
        Ok(Covariance1d {
            profile,
            cutoff,
            radius: cutoff * length,
            length,
            period,
            step: 0.5 * length / rate,
        })
    }

    pub fn length(&self) -> f64 {
        self.length
    }

    pub fn period(&self) -> Option<f64> {
        self.period
    }

    /// The image indices j for which c centred on x + jP reaches [lo, hi].
    fn images(&self, lo: f64, hi: f64, x: f64) -> std::ops::RangeInclusive<i64> {
        match self.period {
            None => {
                // 0..=0 if c centred on x reaches [lo, hi], else the empty 0..=-1
                let reaches = i64::from(x + self.radius >= lo && x - self.radius <= hi);
                0..=reaches - 1
            }
            Some(p) => {
                let a = ((lo - self.radius - x) / p).ceil() as i64;
                let b = ((hi + self.radius - x) / p).floor() as i64;
                a..=b
            }
        }
    }

    fn shift(&self, j: i64) -> f64 {
        self.period.map_or(0.0, |p| j as f64 * p)
    }

    /// c(Δ), summed over periodic images.
    pub fn value(&self, delta: f64) -> f64 {
        self.images(delta, delta, 0.0)
            .map(|j| {
                let d = delta - self.shift(j);
                if d.abs() <= self.radius {
                    self.profile.value(d)
                } else {
                    0.0
                }
            })
            .sum()
    }

    fn centred(&self, x: f64) -> Element {
        Element::profile(self.profile, x, self.cutoff).expect("valid profile")
    }

    /// (c(Δ), ℓ ∂c/∂ℓ) summed over periodic images. c depends on Δ/ℓ, so ℓ ∂c/∂ℓ = −Δ c'(Δ).
    pub fn value_and_length_derivative(&self, delta: f64) -> (f64, f64) {
        self.images(delta, delta, 0.0)
            .map(|j| {
                let d = delta - self.shift(j);
                if d.abs() <= self.radius {
                    (self.profile.value(d), -d * self.profile.d_dt(d))
                } else {
                    (0.0, 0.0)
                }
            })
            .fold((0.0, 0.0), |(a, b), (c, d)| (a + c, b + d))
    }

    /// Σ_j ⟨e, c(· − x − jP)⟩_w and its ℓ ∂/∂ℓ: a dilation of the centred profile, i.e. its
    /// `Scale` derivative (including the tiny terms of its moving truncation edges).
    fn element_point_both(
        &self,
        e: &Element,
        w: Weight,
        x: f64,
        derivative: bool,
        knots: &mut Vec<f64>,
    ) -> (f64, f64) {
        let mut v = 0.0;
        let mut d = 0.0;
        for j in self.images(e.lower(), e.upper(), x) {
            let k = self.centred(x + self.shift(j));
            v += inner(e, &k, w, knots);
            if derivative {
                let jumps = k.jumps(Parameter::Scale).expect("profiles have a scale");
                d += inner_derivative(&k, e, Parameter::Scale, w, &jumps, knots)
                    .expect("profiles have a scale");
            }
        }
        (v, d)
    }

    /// Σ_j ⟨e, c(· − x − jP)⟩_w.
    fn element_point(&self, e: &Element, w: Weight, x: f64, knots: &mut Vec<f64>) -> f64 {
        self.element_point_both(e, w, x, false, knots).0
    }

    /// ∫∫ a(x) w_a(x) b(y) w_b(y) c(x − y) dx dy, the unnormalized covariance of two atoms (a
    /// delta stands for its point, ignoring its weight).
    pub fn between(
        &self,
        a: &Element,
        wa: Weight,
        b: &Element,
        wb: Weight,
        scratch: &mut Scratch,
    ) -> f64 {
        match (a, b) {
            (Element::Delta { x: xa }, Element::Delta { x: xb }) => self.value(xa - xb),
            (Element::Delta { x }, e) => self.element_point(e, wb, *x, &mut scratch.inner),
            (e, Element::Delta { x }) => self.element_point(e, wa, *x, &mut scratch.inner),
            _ => self.element_element(a, wa, b, wb, false, scratch).0,
        }
    }

    /// [`Covariance1d::between`] and its derivative ℓ ∂/∂ℓ with respect to the correlation
    /// length (the atoms' normalizations do not depend on ℓ).
    pub fn between_and_length_derivative(
        &self,
        a: &Element,
        wa: Weight,
        b: &Element,
        wb: Weight,
        scratch: &mut Scratch,
    ) -> (f64, f64) {
        match (a, b) {
            (Element::Delta { x: xa }, Element::Delta { x: xb }) => {
                self.value_and_length_derivative(xa - xb)
            }
            (Element::Delta { x }, e) => {
                self.element_point_both(e, wb, *x, true, &mut scratch.inner)
            }
            (e, Element::Delta { x }) => {
                self.element_point_both(e, wa, *x, true, &mut scratch.inner)
            }
            _ => self.element_element(a, wa, b, wb, true, scratch),
        }
    }

    fn element_element(
        &self,
        a: &Element,
        wa: Weight,
        b: &Element,
        wb: Weight,
        derivative: bool,
        scratch: &mut Scratch,
    ) -> (f64, f64) {
        let (lo, hi) = (a.lower(), a.upper());
        let knots = &mut scratch.outer;
        knots.clear();
        knots.push(lo);
        knots.push(hi);
        a.push_knots(lo, hi, knots);
        // knots of b (with its ends), shifted to every image that reaches a, and ± the radius
        let b_knots = &mut scratch.other;
        b_knots.clear();
        b_knots.push(b.lower());
        b_knots.push(b.upper());
        b.push_knots(b.lower(), b.upper(), b_knots);
        let reach = self.images(b.lower() - (hi - lo), b.upper() + (hi - lo), lo);
        for j in reach {
            let s = self.shift(j);
            for &k in b_knots.iter() {
                for x in [k - s, k - s - self.radius, k - s + self.radius] {
                    if x > lo && x < hi {
                        knots.push(x);
                    }
                }
            }
        }
        knots.sort_unstable_by(f64::total_cmp);
        knots.dedup();

        let inner_knots = &mut scratch.inner;
        let mut total = 0.0;
        let mut total_d = 0.0;
        for w in knots.windows(2) {
            let (mut x0, end) = (w[0], w[1]);
            while x0 < end {
                let limit = self.step.min(a.step_limit(x0)).min(match wa {
                    Weight::Uniform => f64::INFINITY,
                    Weight::Cosine { scale } => 0.25 / scale.abs(),
                });
                let h = if x0 + 1.05 * limit >= end {
                    end - x0
                } else {
                    limit
                };
                // one quadrature for the value and the derivative: the rule is linear, so
                // accumulate both from the same nodes
                let rule = gl8();
                let (mid, half) = (x0 + 0.5 * h, 0.5 * h);
                for (t, wt) in rule.nodes.iter().zip(&rule.weights) {
                    let x = mid + half * t;
                    let v = a.evaluate(x) * wa.value(x);
                    if v != 0.0 {
                        let (g, gd) = self.element_point_both(b, wb, x, derivative, inner_knots);
                        total += wt * half * v * g;
                        total_d += wt * half * v * gd;
                    }
                }
                x0 += h;
            }
        }
        (total, total_d)
    }
}

/// Scratch buffers for [`Covariance1d::between`].
#[derive(Debug, Default)]
pub struct Scratch {
    outer: Vec<f64>,
    other: Vec<f64>,
    inner: Vec<f64>,
}

/// σ² Π_d c_d.
#[derive(Debug, Clone)]
pub struct ProductCovariance {
    pub variance: f64,
    pub dims: Vec<Covariance1d>,
}

/// Atoms along each dimension: `spaces[d]` holds the d-th factor of every atom (all of equal
/// length), `weights[d]` its weight.
#[derive(Debug, Clone, Copy)]
pub struct Atoms<'a> {
    pub spaces: &'a [&'a Space],
    pub weights: &'a [Weight],
}

impl Atoms<'_> {
    fn len(&self, dims: usize) -> Result<usize> {
        if self.spaces.len() != dims || self.weights.len() != dims {
            return invalid(format!(
                "atoms need one space and one weight per dimension ({dims}), got {} and {}",
                self.spaces.len(),
                self.weights.len()
            ));
        }
        let n = self.spaces.first().map_or(0, |s| s.len());
        if self.spaces.iter().any(|s| s.len() != n) {
            return invalid("every dimension's space must have one element per atom");
        }
        Ok(n)
    }

    /// N = ∫ E w per dimension and atom (1 for deltas).
    fn norms(&self) -> Vec<Vec<f64>> {
        self.spaces
            .iter()
            .zip(self.weights)
            .map(|(s, &w)| {
                let mut knots = Vec::new();
                s.elements()
                    .iter()
                    .map(|e| match e {
                        Element::Delta { .. } => 1.0,
                        _ => integral(e, w, &mut knots),
                    })
                    .collect()
            })
            .collect()
    }
}

impl ProductCovariance {
    #[allow(clippy::too_many_arguments)]
    fn entry(
        &self,
        a: &Atoms,
        na: &[Vec<f64>],
        i: usize,
        b: &Atoms,
        nb: &[Vec<f64>],
        j: usize,
        scratch: &mut Scratch,
    ) -> f64 {
        let mut v = self.variance;
        for (d, c) in self.dims.iter().enumerate() {
            let f = c.between(
                &a.spaces[d].elements()[i],
                a.weights[d],
                &b.spaces[d].elements()[j],
                b.weights[d],
                scratch,
            );
            if f == 0.0 {
                return 0.0;
            }
            v *= f / (na[d][i] * nb[d][j]);
        }
        v
    }

    /// The covariance matrix between the atoms of `a` (rows) and `b` (columns). With
    /// `symmetric` (a and b the same atoms) only the upper triangle is computed.
    pub fn matrix(&self, a: Atoms, b: Atoms, symmetric: bool) -> Result<Array2<f64>> {
        let dims = self.dims.len();
        let (n, m) = (a.len(dims)?, b.len(dims)?);
        if symmetric && n != m {
            return invalid("a symmetric covariance needs the same atoms on both sides");
        }
        let (na, nb) = (a.norms(), b.norms());
        let rows: Vec<Vec<f64>> = (0..n)
            .into_par_iter()
            .map_init(Scratch::default, |scratch, i| {
                let start = if symmetric { i } else { 0 };
                (start..m)
                    .map(|j| self.entry(&a, &na, i, &b, &nb, j, scratch))
                    .collect()
            })
            .collect();
        let mut out = Array2::zeros((n, m));
        for (i, row) in rows.into_iter().enumerate() {
            let start = if symmetric { i } else { 0 };
            for (k, v) in row.into_iter().enumerate() {
                out[[i, start + k]] = v;
                if symmetric {
                    out[[start + k, i]] = v;
                }
            }
        }
        Ok(out)
    }

    /// The symmetric covariance matrix K of the atoms of `a`, and ℓ_d ∂K/∂ℓ_d for every
    /// dimension d (the variance derivative σ² ∂K/∂σ² is K itself): what the gradient of the
    /// log likelihood with respect to the correlation lengths needs.
    pub fn matrix_with_length_derivatives(
        &self,
        a: Atoms,
    ) -> Result<(Array2<f64>, Vec<Array2<f64>>)> {
        let dims = self.dims.len();
        let n = a.len(dims)?;
        let na = a.norms();
        type Row = (Vec<f64>, Vec<Vec<f64>>);
        let rows: Vec<Row> = (0..n)
            .into_par_iter()
            .map_init(Scratch::default, |scratch, i| {
                let mut vals = Vec::with_capacity(n - i);
                let mut ders = vec![Vec::with_capacity(n - i); dims];
                let mut f = vec![0.0; dims];
                let mut fd = vec![0.0; dims];
                for j in i..n {
                    let mut scale = self.variance;
                    for (d, c) in self.dims.iter().enumerate() {
                        let (v, dv) = c.between_and_length_derivative(
                            &a.spaces[d].elements()[i],
                            a.weights[d],
                            &a.spaces[d].elements()[j],
                            a.weights[d],
                            scratch,
                        );
                        let norm = na[d][i] * na[d][j];
                        f[d] = v / norm;
                        fd[d] = dv / norm;
                    }
                    let k: f64 = f.iter().product::<f64>() * scale;
                    vals.push(k);
                    for d in 0..dims {
                        // σ² f'_d Π_{e≠d} f_e
                        scale = self.variance * fd[d];
                        for (e, fe) in f.iter().enumerate() {
                            if e != d {
                                scale *= fe;
                            }
                        }
                        ders[d].push(scale);
                    }
                }
                (vals, ders)
            })
            .collect();
        let mut k = Array2::zeros((n, n));
        let mut g = vec![Array2::zeros((n, n)); dims];
        for (i, (vals, ders)) in rows.into_iter().enumerate() {
            for (off, v) in vals.into_iter().enumerate() {
                let j = i + off;
                k[[i, j]] = v;
                k[[j, i]] = v;
                for d in 0..dims {
                    g[d][[i, j]] = ders[d][off];
                    g[d][[j, i]] = ders[d][off];
                }
            }
        }
        Ok((k, g))
    }

    /// The variances of the atoms of `a` (the diagonal of `matrix(a, a)`).
    pub fn diagonal(&self, a: Atoms) -> Result<Vec<f64>> {
        let n = a.len(self.dims.len())?;
        let na = a.norms();
        Ok((0..n)
            .into_par_iter()
            .map_init(Scratch::default, |scratch, i| {
                self.entry(&a, &na, i, &a, &na, i, scratch)
            })
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn box_(a: f64, b: f64) -> Element {
        Element::boxcar(a, b).unwrap()
    }

    #[test]
    fn box_box_squared_exponential_matches_closed_form() {
        // ∫_a^b ∫_c^d exp(−(x − y)²/2ℓ²) dy dx in closed form via F(u) = ℓ²(e^(−u²/2ℓ²) − 1) +
        // ℓ√(π/2) u erf(u/(ℓ√2)), the double antiderivative of the Gaussian.
        let l = 0.7f64;
        let f = |u: f64| {
            l * l * ((-u * u / (2.0 * l * l)).exp() - 1.0)
                + l * (std::f64::consts::PI / 2.0).sqrt() * u * libm::erf(u / (l * 2f64.sqrt()))
        };
        let (a, b, c, d) = (0.0, 1.3, 0.4, 2.9);
        let expected = f(b - c) - f(b - d) - f(a - c) + f(a - d);
        let cov = Covariance1d::new(Family::SquaredExponential, l, None).unwrap();
        let got = cov.between(
            &box_(a, b),
            Weight::Uniform,
            &box_(c, d),
            Weight::Uniform,
            &mut Scratch::default(),
        );
        assert!(
            (got - expected).abs() < 1e-14 * expected,
            "{got} vs {expected}"
        );
    }

    #[test]
    fn box_box_exponential_matches_closed_form() {
        // for disjoint boxes, ∫∫ e^(−(y − x)/ℓ) = ℓ² (e^(−(c − b)/ℓ) − e^(−(d − b)/ℓ))
        //                                          · (1 − e^(−(b − a)/ℓ)) e^... (product form)
        let l = 0.5f64;
        let (a, b, c, d) = (0.0, 1.0, 1.5, 2.2);
        let expected =
            l * l * (1.0 - (-(b - a) / l).exp()) * ((-(c - b) / l).exp() - (-(d - b) / l).exp());
        let cov = Covariance1d::new(Family::Matern(MaternOrder::Half), l, None).unwrap();
        let got = cov.between(
            &box_(a, b),
            Weight::Uniform,
            &box_(c, d),
            Weight::Uniform,
            &mut Scratch::default(),
        );
        assert!(
            (got - expected).abs() < 1e-14 * expected,
            "{got} vs {expected}"
        );
    }

    #[test]
    fn overlapping_boxes_exponential_kink_inside() {
        // ∫_0^1 ∫_0^1 e^(−|x − y|/ℓ) dx dy = 2ℓ (1 − ℓ (1 − e^(−1/ℓ)))
        let l = 0.3f64;
        let expected = 2.0 * l * (1.0 - l * (1.0 - (-1.0 / l).exp()));
        let cov = Covariance1d::new(Family::Matern(MaternOrder::Half), l, None).unwrap();
        let got = cov.between(
            &box_(0.0, 1.0),
            Weight::Uniform,
            &box_(0.0, 1.0),
            Weight::Uniform,
            &mut Scratch::default(),
        );
        assert!(
            (got - expected).abs() < 1e-14 * expected,
            "{got} vs {expected}"
        );
    }

    #[test]
    fn periodic_images_wrap() {
        let cov =
            Covariance1d::new(Family::Matern(MaternOrder::ThreeHalves), 20.0, Some(360.0)).unwrap();
        assert!((cov.value(359.0 - 1.0) - cov.value(-2.0)).abs() < 1e-16);
        let mut s = Scratch::default();
        let d = |x: f64| Element::delta(x).unwrap();
        // a box straddling 0 equals the same box written across 360
        let a = cov.between(
            &box_(-10.0, 10.0),
            Weight::Uniform,
            &d(355.0),
            Weight::Uniform,
            &mut s,
        );
        let b = cov.between(
            &box_(350.0, 370.0),
            Weight::Uniform,
            &d(355.0),
            Weight::Uniform,
            &mut s,
        );
        assert!((a - b).abs() < 1e-12 * a);
        // a full-circle box sees every point the same
        let full = |x: f64| {
            cov.between(
                &box_(0.0, 360.0),
                Weight::Uniform,
                &d(x),
                Weight::Uniform,
                &mut Scratch::default(),
            )
        };
        assert!((full(3.0) - full(181.0)).abs() < 1e-12 * full(3.0));
    }

    #[test]
    fn length_derivatives_match_finite_differences() {
        let d = |x: f64| Element::delta(x).unwrap();
        let w = Weight::Cosine { scale: 0.02 };
        let pairs: Vec<(Element, Weight, Element, Weight)> = vec![
            (d(0.3), Weight::Uniform, d(1.1), Weight::Uniform),
            (box_(0.0, 1.3), w, d(0.7), Weight::Uniform),
            (d(2.0), Weight::Uniform, box_(0.5, 1.6), w),
            (box_(0.0, 1.3), w, box_(0.4, 2.9), Weight::Uniform),
            (
                Element::hat(0.0, 0.5, 1.2).unwrap(),
                Weight::Uniform,
                box_(0.9, 1.5),
                w,
            ),
        ];
        let families = [
            Family::SquaredExponential,
            Family::Matern(MaternOrder::Half),
            Family::Matern(MaternOrder::ThreeHalves),
            Family::Matern(MaternOrder::FiveHalves),
        ];
        for family in families {
            for period in [None, Some(3.0)] {
                let l = 0.6;
                let c = Covariance1d::new(family, l, period).unwrap();
                let h = 1e-6;
                let cp = Covariance1d::new(family, l * (1.0 + h), period).unwrap();
                let cm = Covariance1d::new(family, l * (1.0 - h), period).unwrap();
                let mut s = Scratch::default();
                for (a, wa, b, wb) in &pairs {
                    let (v, dv) = c.between_and_length_derivative(a, *wa, b, *wb, &mut s);
                    assert!(
                        (v - c.between(a, *wa, b, *wb, &mut s)).abs() < 1e-15 * v.abs().max(1.0)
                    );
                    let fd = (cp.between(a, *wa, b, *wb, &mut s)
                        - cm.between(a, *wa, b, *wb, &mut s))
                        / (2.0 * h);
                    assert!(
                        (dv - fd).abs() < 1e-7 * fd.abs().max(1e-3),
                        "{family:?} {period:?} {a:?} {b:?}: {dv} vs {fd}"
                    );
                }
            }
        }
    }

    #[test]
    fn element_element_is_symmetric() {
        let cov = Covariance1d::new(Family::Matern(MaternOrder::FiveHalves), 0.3, None).unwrap();
        let mut s = Scratch::default();
        let h = Element::hat(0.0, 0.4, 1.0).unwrap();
        let b = box_(0.2, 0.9);
        let w = Weight::Cosine { scale: 1.0 };
        let ab = cov.between(&h, w, &b, Weight::Uniform, &mut s);
        let ba = cov.between(&b, Weight::Uniform, &h, w, &mut s);
        assert!((ab - ba).abs() < 1e-14 * ab, "{ab} vs {ba}");
    }
}
