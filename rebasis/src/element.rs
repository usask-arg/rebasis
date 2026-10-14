//! Single basis functions (or measurement kernels) on the real line.
//!
//! Every element is a *shape*: rebasis does not attach a normalization to it. Boxes, hats and
//! profiles peak at 1, so a grid of hats or boxes is a partition of unity and its coefficients
//! are function values. Whether an element acts as a basis function (representing f) or as a
//! measurement kernel (an instrument response, unit area) is decided by the operation it is
//! used in, see [`crate::ops`].

use std::sync::Arc;

use crate::error::{Result, invalid};
use crate::profile::Profile;

/// One cubic piece p(x) = Σ c_m (x − a)^m on [a, b).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Segment {
    pub a: f64,
    pub b: f64,
    pub coeffs: [f64; 4],
}

impl Segment {
    #[inline]
    fn eval(&self, x: f64) -> f64 {
        let u = x - self.a;
        let c = &self.coeffs;
        ((c[3] * u + c[2]) * u + c[1]) * u + c[0]
    }

    fn integral(&self) -> f64 {
        let w = self.b - self.a;
        let c = &self.coeffs;
        w * (c[0] + w * (c[1] / 2.0 + w * (c[2] / 3.0 + w * c[3] / 4.0)))
    }
}

/// A piecewise-cubic shape in its own coordinate u, shared by every element that is a shifted
/// and stretched copy of it (e.g. a tabulated instrument line shape used for all channels).
#[derive(Debug, Clone, PartialEq)]
pub struct Shape {
    segments: Vec<Segment>,
}

fn check_table(x: &[f64], y: &[f64]) -> Result<()> {
    if x.len() != y.len() || x.len() < 2 {
        return invalid("a tabulated shape needs x and y of equal length >= 2");
    }
    if x.iter().chain(y).any(|v| !v.is_finite()) {
        return invalid("a tabulated shape must be finite");
    }
    if x.windows(2).any(|w| w[1] <= w[0]) {
        return invalid("a tabulated shape needs strictly increasing x");
    }
    Ok(())
}

impl Shape {
    /// Segments must be finite, non-empty, sorted and non-overlapping (gaps are zero).
    pub fn new(segments: Vec<Segment>) -> Result<Self> {
        if segments.is_empty() {
            return invalid("a shape needs at least one segment");
        }
        for s in &segments {
            if !(s.a.is_finite() && s.b.is_finite() && s.coeffs.iter().all(|c| c.is_finite())) {
                return invalid("shape segments must be finite");
            }
            if s.b <= s.a {
                return invalid("shape segments need a < b");
            }
        }
        if segments.windows(2).any(|w| w[1].a < w[0].b) {
            return invalid("shape segments must be sorted and non-overlapping");
        }
        Ok(Shape { segments })
    }

    /// The piecewise-linear interpolant of (x, y), zero outside [x[0], x[-1]].
    pub fn piecewise_linear(x: &[f64], y: &[f64]) -> Result<Self> {
        check_table(x, y)?;
        let segments = x
            .windows(2)
            .zip(y.windows(2))
            .map(|(xw, yw)| Segment {
                a: xw[0],
                b: xw[1],
                coeffs: [yw[0], (yw[1] - yw[0]) / (xw[1] - xw[0]), 0.0, 0.0],
            })
            .collect();
        Shape::new(segments)
    }

    /// The monotone piecewise-cubic Hermite (PCHIP, Fritsch–Carlson) interpolant of (x, y),
    /// zero outside [x[0], x[-1]]. Smooth (C¹) and free of overshoot, with the same slopes as
    /// `scipy.interpolate.PchipInterpolator`.
    pub fn pchip(x: &[f64], y: &[f64]) -> Result<Self> {
        check_table(x, y)?;
        let n = x.len();
        let h: Vec<f64> = x.windows(2).map(|w| w[1] - w[0]).collect();
        let delta: Vec<f64> = (0..n - 1).map(|k| (y[k + 1] - y[k]) / h[k]).collect();
        let mut d = vec![0.0; n];
        if n == 2 {
            d[0] = delta[0];
            d[1] = delta[0];
        } else {
            for k in 1..n - 1 {
                if delta[k - 1] * delta[k] > 0.0 {
                    let w1 = 2.0 * h[k] + h[k - 1];
                    let w2 = h[k] + 2.0 * h[k - 1];
                    d[k] = (w1 + w2) / (w1 / delta[k - 1] + w2 / delta[k]);
                }
            }
            let edge = |h0: f64, h1: f64, m0: f64, m1: f64| {
                let d = ((2.0 * h0 + h1) * m0 - h0 * m1) / (h0 + h1);
                if d.signum() != m0.signum() {
                    0.0
                } else if m0.signum() != m1.signum() && d.abs() > 3.0 * m0.abs() {
                    3.0 * m0
                } else {
                    d
                }
            };
            d[0] = edge(h[0], h[1], delta[0], delta[1]);
            d[n - 1] = edge(h[n - 2], h[n - 3], delta[n - 2], delta[n - 3]);
        }
        let segments = (0..n - 1)
            .map(|k| Segment {
                a: x[k],
                b: x[k + 1],
                coeffs: [
                    y[k],
                    d[k],
                    (3.0 * delta[k] - 2.0 * d[k] - d[k + 1]) / h[k],
                    (d[k] + d[k + 1] - 2.0 * delta[k]) / (h[k] * h[k]),
                ],
            })
            .collect();
        Shape::new(segments)
    }

    /// 1 on [−top, top], falling linearly to 0 at ±bottom. top = 0 is a triangle, top = bottom
    /// a box.
    pub fn trapezoid(top: f64, bottom: f64) -> Result<Self> {
        if !(top.is_finite() && bottom.is_finite() && 0.0 <= top && top <= bottom && bottom > 0.0) {
            return invalid(format!(
                "trapezoid needs 0 <= top <= bottom, bottom > 0; got {top}, {bottom}"
            ));
        }
        if top == bottom {
            return Shape::new(vec![Segment {
                a: -bottom,
                b: bottom,
                coeffs: [1.0, 0.0, 0.0, 0.0],
            }]);
        }
        let mut x = vec![-bottom, -top];
        let mut y = vec![0.0, 1.0];
        if top > 0.0 {
            x.push(top);
            y.push(1.0);
        }
        x.push(bottom);
        y.push(0.0);
        Shape::piecewise_linear(&x, &y)
    }

    pub fn segments(&self) -> &[Segment] {
        &self.segments
    }

    /// The derivative within each segment; jumps at segment ends are not included.
    pub fn derivative(&self) -> Shape {
        Shape {
            segments: self
                .segments
                .iter()
                .map(|s| {
                    let c = s.coeffs;
                    Segment {
                        a: s.a,
                        b: s.b,
                        coeffs: [c[1], 2.0 * c[2], 3.0 * c[3], 0.0],
                    }
                })
                .collect(),
        }
    }

    fn scaled(&self, factor: f64) -> Shape {
        Shape {
            segments: self
                .segments
                .iter()
                .map(|s| Segment {
                    a: s.a,
                    b: s.b,
                    coeffs: s.coeffs.map(|c| c * factor),
                })
                .collect(),
        }
    }

    pub fn lower(&self) -> f64 {
        self.segments[0].a
    }

    pub fn upper(&self) -> f64 {
        self.segments[self.segments.len() - 1].b
    }

    pub fn evaluate(&self, u: f64) -> f64 {
        let k = self.segments.partition_point(|s| s.a <= u);
        if k == 0 {
            return 0.0;
        }
        let s = &self.segments[k - 1];
        if u < s.b { s.eval(u) } else { 0.0 }
    }

    pub fn integral(&self) -> f64 {
        self.segments.iter().map(Segment::integral).sum()
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum Element {
    /// Point evaluation. As a kernel it samples; as a basis function it is a point mass.
    Delta { x: f64 },
    /// 1 on [a, b), 0 elsewhere.
    Box { a: f64, b: f64 },
    /// Linear from 0 at a to 1 at c and back to 0 at b. a == c or c == b gives a half hat.
    Hat { a: f64, c: f64, b: f64 },
    /// shape((x − center) / stretch).
    Piecewise {
        shape: Arc<Shape>,
        center: f64,
        stretch: f64,
    },
    /// profile(x − center) on [lo, hi], 0 elsewhere.
    Profile {
        profile: Profile,
        center: f64,
        lo: f64,
        hi: f64,
    },
}

impl Element {
    pub fn delta(x: f64) -> Result<Self> {
        if !x.is_finite() {
            return invalid("delta position must be finite");
        }
        Ok(Element::Delta { x })
    }

    pub fn boxcar(a: f64, b: f64) -> Result<Self> {
        if !(a.is_finite() && b.is_finite() && a < b) {
            return invalid(format!("box needs finite a < b, got [{a}, {b}]"));
        }
        Ok(Element::Box { a, b })
    }

    pub fn hat(a: f64, c: f64, b: f64) -> Result<Self> {
        if !(a.is_finite() && b.is_finite() && c.is_finite() && a <= c && c <= b && a < b) {
            return invalid(format!(
                "hat needs finite a <= c <= b with a < b, got ({a}, {c}, {b})"
            ));
        }
        Ok(Element::Hat { a, c, b })
    }

    /// `profile` centered at `center`, truncated at `cutoff` (see [`Profile::support`]).
    pub fn profile(profile: Profile, center: f64, cutoff: f64) -> Result<Self> {
        if !center.is_finite() {
            return invalid("profile center must be finite");
        }
        let (left, right) = profile.support(cutoff)?;
        Ok(Element::Profile {
            profile,
            center,
            lo: center - left,
            hi: center + right,
        })
    }

    /// A Gaussian truncated at `cutoff` standard deviations either side of the center.
    pub fn gaussian(center: f64, sigma: f64, cutoff: f64) -> Result<Self> {
        Element::profile(Profile::gaussian(sigma)?, center, cutoff)
    }

    pub fn piecewise(shape: Arc<Shape>, center: f64, stretch: f64) -> Result<Self> {
        if !(center.is_finite() && stretch.is_finite() && stretch > 0.0) {
            return invalid("piecewise element needs finite center and stretch > 0");
        }
        Ok(Element::Piecewise {
            shape,
            center,
            stretch,
        })
    }

    pub fn lower(&self) -> f64 {
        match self {
            Element::Delta { x } => *x,
            Element::Box { a, .. } | Element::Hat { a, .. } => *a,
            Element::Profile { lo, .. } => *lo,
            Element::Piecewise {
                shape,
                center,
                stretch,
            } => center + stretch * shape.lower(),
        }
    }

    pub fn upper(&self) -> f64 {
        match self {
            Element::Delta { x } => *x,
            Element::Box { b, .. } | Element::Hat { b, .. } => *b,
            Element::Profile { hi, .. } => *hi,
            Element::Piecewise {
                shape,
                center,
                stretch,
            } => center + stretch * shape.upper(),
        }
    }

    pub fn center(&self) -> f64 {
        match self {
            Element::Delta { x } => *x,
            Element::Box { a, b } => 0.5 * (a + b),
            Element::Hat { c, .. } => *c,
            Element::Profile { center, .. } | Element::Piecewise { center, .. } => *center,
        }
    }

    /// Point value. A delta has no point value and evaluates to 0.
    pub fn evaluate(&self, x: f64) -> f64 {
        match self {
            Element::Delta { .. } => 0.0,
            Element::Box { a, b } => {
                if *a <= x && x < *b {
                    1.0
                } else {
                    0.0
                }
            }
            Element::Hat { a, c, b } => {
                if x < *a || x > *b {
                    0.0
                } else if x == *c {
                    1.0
                } else if x < *c {
                    (x - a) / (c - a)
                } else {
                    (b - x) / (b - c)
                }
            }
            Element::Profile {
                profile,
                center,
                lo,
                hi,
            } => {
                if x < *lo || x > *hi {
                    0.0
                } else {
                    profile.value(x - center)
                }
            }
            Element::Piecewise {
                shape,
                center,
                stretch,
            } => shape.evaluate((x - center) / stretch),
        }
    }

    /// The `order`-th derivative of a piecewise-polynomial element, excluding the point masses
    /// at its jumps (so the derivative of a box is zero). Errors for deltas and profiles.
    pub fn derivative(&self, order: usize) -> Result<Element> {
        if order == 0 {
            return Ok(self.clone());
        }
        let (shape, center, stretch) = match self {
            Element::Box { a, b } => (
                Shape::new(vec![Segment {
                    a: *a,
                    b: *b,
                    coeffs: [1.0, 0.0, 0.0, 0.0],
                }])?,
                0.0,
                1.0,
            ),
            Element::Hat { a, c, b } => {
                let mut segments = Vec::new();
                if c > a {
                    segments.push(Segment {
                        a: *a,
                        b: *c,
                        coeffs: [0.0, 1.0 / (c - a), 0.0, 0.0],
                    });
                }
                if b > c {
                    segments.push(Segment {
                        a: *c,
                        b: *b,
                        coeffs: [1.0, -1.0 / (b - c), 0.0, 0.0],
                    });
                }
                (Shape::new(segments)?, 0.0, 1.0)
            }
            Element::Piecewise {
                shape,
                center,
                stretch,
            } => ((**shape).clone(), *center, *stretch),
            _ => return invalid("only piecewise-polynomial elements have derivatives"),
        };
        let mut d = shape;
        for _ in 0..order {
            d = d.derivative();
        }
        let d = d.scaled(stretch.powi(-(order as i32)));
        Element::piecewise(Arc::new(d), center, stretch)
    }

    /// The profile and its center, for the integrator's step control.
    pub(crate) fn as_profile(&self) -> Option<(&Profile, f64)> {
        match self {
            Element::Profile {
                profile, center, ..
            } => Some((profile, *center)),
            _ => None,
        }
    }

    /// ∫ φ(x) dx over the element's support, exactly for piecewise polynomials.
    pub(crate) fn polynomial_integral(&self) -> Option<f64> {
        match self {
            Element::Delta { .. } => Some(1.0),
            Element::Box { a, b } => Some(b - a),
            Element::Hat { a, b, .. } => Some(0.5 * (b - a)),
            Element::Piecewise { shape, stretch, .. } => Some(stretch * shape.integral()),
            Element::Profile { .. } => None,
        }
    }

    /// Appends the points where this element is not smooth and that lie strictly inside (lo, hi).
    pub(crate) fn push_knots(&self, lo: f64, hi: f64, out: &mut Vec<f64>) {
        let mut push = |x: f64| {
            if x > lo && x < hi {
                out.push(x)
            }
        };
        match self {
            Element::Delta { x } => push(*x),
            Element::Box { a, b } => {
                push(*a);
                push(*b);
            }
            Element::Hat { a, c, b } => {
                push(*a);
                push(*c);
                push(*b);
            }
            Element::Profile {
                center,
                lo: l,
                hi: h,
                ..
            } => {
                push(*l);
                push(*center);
                push(*h);
            }
            Element::Piecewise {
                shape,
                center,
                stretch,
            } => {
                for s in shape.segments() {
                    push(center + stretch * s.a);
                    push(center + stretch * s.b);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pchip_interpolates_and_does_not_overshoot() {
        let x = [0.0, 1.0, 2.0, 3.0, 4.0];
        let y = [0.0, 0.0, 1.0, 1.0, 0.5];
        let s = Shape::pchip(&x, &y).unwrap();
        for (xi, yi) in x.iter().zip(y).take(4) {
            assert!((s.evaluate(*xi) - yi).abs() < 1e-15);
        }
        for k in 0..400 {
            let v = s.evaluate(k as f64 * 0.01);
            assert!((-1e-15..=1.0 + 1e-15).contains(&v));
        }
    }

    #[test]
    fn trapezoid_integral() {
        let t = Shape::trapezoid(0.5, 1.5).unwrap();
        assert!((t.integral() - 2.0).abs() < 1e-15); // (1 + 3) / 2
        assert!((Shape::trapezoid(0.0, 1.0).unwrap().integral() - 1.0).abs() < 1e-15);
        assert!((Shape::trapezoid(1.0, 1.0).unwrap().integral() - 2.0).abs() < 1e-15);
    }
}
