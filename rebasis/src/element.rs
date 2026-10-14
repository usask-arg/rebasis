//! Single basis functions (or measurement kernels) on the real line.
//!
//! Every element is a *shape*: rebasis does not attach a normalization to it. Boxes, hats and
//! Gaussians peak at 1, so a grid of hats or boxes is a partition of unity and its coefficients
//! are function values. Whether an element acts as a basis function (representing f) or as a
//! measurement kernel (an instrument response, unit area) is decided by the operation it is
//! used in, see [`crate::ops`].

use std::sync::Arc;

use crate::error::{Result, invalid};
use crate::quadrature::{gl8, integrate};

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
        if x.len() != y.len() || x.len() < 2 {
            return invalid("piecewise_linear needs x and y of equal length >= 2");
        }
        if x.windows(2).any(|w| w[1] <= w[0]) {
            return invalid("piecewise_linear needs strictly increasing x");
        }
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

    pub fn segments(&self) -> &[Segment] {
        &self.segments
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
    /// exp(−(x − center)² / 2σ²) on [lo, hi], 0 elsewhere.
    Gaussian {
        center: f64,
        sigma: f64,
        lo: f64,
        hi: f64,
    },
    /// shape((x − center) / stretch).
    Piecewise {
        shape: Arc<Shape>,
        center: f64,
        stretch: f64,
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

    /// A Gaussian truncated at `cutoff` standard deviations either side of the center.
    pub fn gaussian(center: f64, sigma: f64, cutoff: f64) -> Result<Self> {
        if !(center.is_finite() && sigma.is_finite() && sigma > 0.0) {
            return invalid(format!(
                "gaussian needs finite center and sigma > 0, got {sigma}"
            ));
        }
        if !(cutoff.is_finite() && cutoff > 0.0) {
            return invalid("gaussian cutoff must be finite and > 0");
        }
        Ok(Element::Gaussian {
            center,
            sigma,
            lo: center - cutoff * sigma,
            hi: center + cutoff * sigma,
        })
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
            Element::Gaussian { lo, .. } => *lo,
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
            Element::Gaussian { hi, .. } => *hi,
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
            Element::Gaussian { center, .. } | Element::Piecewise { center, .. } => *center,
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
            Element::Gaussian {
                center,
                sigma,
                lo,
                hi,
            } => {
                if x < *lo || x > *hi {
                    0.0
                } else {
                    let t = (x - center) / sigma;
                    (-0.5 * t * t).exp()
                }
            }
            Element::Piecewise {
                shape,
                center,
                stretch,
            } => shape.evaluate((x - center) / stretch),
        }
    }

    /// ∫ φ(x) dx; 1 for a delta.
    pub fn integral(&self) -> f64 {
        match self {
            Element::Delta { .. } => 1.0,
            Element::Box { a, b } => b - a,
            Element::Hat { a, b, .. } => 0.5 * (b - a),
            Element::Gaussian { sigma, lo, hi, .. } => {
                let rule = gl8();
                let n = ((hi - lo) / sigma).ceil().max(1.0) as usize;
                let h = (hi - lo) / n as f64;
                (0..n)
                    .map(|k| {
                        let x0 = lo + k as f64 * h;
                        integrate(rule, x0, x0 + h, |x| self.evaluate(x))
                    })
                    .sum()
            }
            Element::Piecewise { shape, stretch, .. } => stretch * shape.integral(),
        }
    }

    /// The standard deviation, for Gaussians (the integrator refines on this scale).
    pub(crate) fn sigma(&self) -> Option<f64> {
        match self {
            Element::Gaussian { sigma, .. } => Some(*sigma),
            _ => None,
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
            Element::Gaussian { lo: l, hi: h, .. } => {
                push(*l);
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
