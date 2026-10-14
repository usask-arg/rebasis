//! A space is an ordered set of elements: the basis {B_j} that coefficients f_j refer to, or a
//! set of measurement kernels {K_i}.

use std::sync::Arc;

use crate::element::{Element, Segment, Shape};
use crate::error::{Result, invalid};
use crate::integrate::{Weight, integral};
use crate::profile::Profile;

#[derive(Debug, Clone)]
pub struct Space {
    elements: Vec<Element>,
    lower: Vec<f64>,
    upper: Vec<f64>,
    /// Element indices sorted by lower limit, for overlap queries.
    by_lower: Vec<usize>,
    sorted_lower: Vec<f64>,
    max_width: f64,
}

/// How a tabulated line shape is interpolated between its points.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Interpolation {
    Linear,
    /// Monotone cubic (PCHIP).
    Pchip,
}

/// Broadcasts a length-1 parameter array to `n`, or checks it has length `n`.
fn broadcast<'a, T: Copy>(
    name: &str,
    values: &'a [T],
    n: usize,
) -> Result<impl Fn(usize) -> T + 'a> {
    let m = values.len();
    if m != 1 && m != n {
        return invalid(format!("{name} has length {m}, expected 1 or {n}"));
    }
    Ok(move |i: usize| if m == 1 { values[0] } else { values[i] })
}

fn check_increasing(name: &str, x: &[f64], min_len: usize) -> Result<()> {
    if x.len() < min_len {
        return invalid(format!("{name} needs at least {min_len} values"));
    }
    if x.iter().any(|v| !v.is_finite()) {
        return invalid(format!("{name} must be finite"));
    }
    if x.windows(2).any(|w| w[1] <= w[0]) {
        return invalid(format!("{name} must be strictly increasing"));
    }
    Ok(())
}

/// B_{i,d}(x) on the knot vector t (Cox–de Boor, right-continuous).
fn bspline_value(t: &[f64], i: usize, d: usize, x: f64) -> f64 {
    let mut n = [0.0; 4];
    for j in 0..=d {
        n[j] = if t[i + j] <= x && x < t[i + j + 1] {
            1.0
        } else {
            0.0
        };
    }
    for k in 1..=d {
        for j in 0..=d - k {
            let (a, b, c, e) = (t[i + j], t[i + j + k], t[i + j + 1], t[i + j + k + 1]);
            let left = if b > a { (x - a) / (b - a) * n[j] } else { 0.0 };
            let right = if e > c {
                (e - x) / (e - c) * n[j + 1]
            } else {
                0.0
            };
            n[j] = left + right;
        }
    }
    n[0]
}

/// Coefficients q of the degree-d polynomial through (s_k, v_k), by Gaussian elimination.
fn fit_polynomial(s: &[f64], v: &[f64]) -> [f64; 4] {
    let n = s.len();
    let mut a = [[0.0; 5]; 4];
    for r in 0..n {
        for c in 0..n {
            a[r][c] = s[r].powi(c as i32);
        }
        a[r][4] = v[r];
    }
    for col in 0..n {
        let piv = (col..n)
            .max_by(|&x, &y| a[x][col].abs().total_cmp(&a[y][col].abs()))
            .unwrap();
        a.swap(col, piv);
        for r in 0..n {
            if r != col {
                let f = a[r][col] / a[col][col];
                for c in col..5 {
                    a[r][c] -= f * a[col][c];
                }
            }
        }
    }
    let mut q = [0.0; 4];
    for r in 0..n {
        q[r] = a[r][4] / a[r][r];
    }
    q
}

impl Space {
    pub fn new(elements: Vec<Element>) -> Self {
        let lower: Vec<f64> = elements.iter().map(Element::lower).collect();
        let upper: Vec<f64> = elements.iter().map(Element::upper).collect();
        let mut by_lower: Vec<usize> = (0..elements.len()).collect();
        by_lower.sort_by(|&i, &j| lower[i].total_cmp(&lower[j]));
        let sorted_lower = by_lower.iter().map(|&i| lower[i]).collect();
        let max_width = lower
            .iter()
            .zip(&upper)
            .map(|(l, u)| u - l)
            .fold(0.0, f64::max);
        Space {
            elements,
            lower,
            upper,
            by_lower,
            sorted_lower,
            max_width,
        }
    }

    /// Point evaluations (as kernels) or point masses (as a basis) at `x`.
    pub fn deltas(x: &[f64]) -> Result<Self> {
        Ok(Space::new(
            x.iter()
                .map(|&v| Element::delta(v))
                .collect::<Result<_>>()?,
        ))
    }

    /// Piecewise-constant basis: box i covers [edges[i], edges[i+1]).
    pub fn boxes(edges: &[f64]) -> Result<Self> {
        check_increasing("edges", edges, 2)?;
        Ok(Space::new(
            edges
                .windows(2)
                .map(|w| Element::Box { a: w[0], b: w[1] })
                .collect(),
        ))
    }

    /// Piecewise-linear basis (hat functions) on `nodes`, half hats at the two ends. The
    /// coefficients are the values at the nodes, and the represented function is the linear
    /// interpolant: the natural basis for a function known by its samples.
    pub fn hats(nodes: &[f64]) -> Result<Self> {
        check_increasing("nodes", nodes, 2)?;
        let n = nodes.len();
        Ok(Space::new(
            (0..n)
                .map(|i| Element::Hat {
                    a: nodes[i.saturating_sub(1)],
                    c: nodes[i],
                    b: nodes[(i + 1).min(n - 1)],
                })
                .collect(),
        ))
    }

    /// B-splines of `degree` (0..=3) on the nondecreasing knot vector `knots`: len(knots) −
    /// degree − 1 functions, function i supported on [knots[i], knots[i + degree + 1]].
    pub fn bsplines(knots: &[f64], degree: usize) -> Result<Self> {
        if degree > 3 {
            return invalid(format!("B-spline degree must be 0..=3, got {degree}"));
        }
        if knots.len() < degree + 2 {
            return invalid(format!(
                "degree {degree} B-splines need at least {} knots",
                degree + 2
            ));
        }
        if knots.iter().any(|v| !v.is_finite()) || knots.windows(2).any(|w| w[1] < w[0]) {
            return invalid("knots must be finite and nondecreasing");
        }
        let n = knots.len() - degree - 1;
        let s: Vec<f64> = (0..=degree)
            .map(|k| (k as f64 + 0.5) / (degree as f64 + 1.0))
            .collect();
        let mut elements = Vec::with_capacity(n);
        for i in 0..n {
            let (lo, hi) = (knots[i], knots[i + degree + 1]);
            if hi <= lo {
                return invalid(format!(
                    "B-spline {i} has empty support (knot multiplicity > degree + 1)"
                ));
            }
            let center = 0.5 * (lo + hi);
            let mut segments = Vec::with_capacity(degree + 1);
            for j in i..=i + degree {
                let (a, b) = (knots[j], knots[j + 1]);
                if b <= a {
                    continue;
                }
                let h = b - a;
                let v: Vec<f64> = s
                    .iter()
                    .map(|&sk| bspline_value(knots, i, degree, a + sk * h))
                    .collect();
                let q = fit_polynomial(&s, &v);
                let mut coeffs = [0.0; 4];
                for m in 0..=degree {
                    coeffs[m] = q[m] / h.powi(m as i32);
                }
                segments.push(Segment {
                    a: a - center,
                    b: b - center,
                    coeffs,
                });
            }
            elements.push(Element::piecewise(
                Arc::new(Shape::new(segments)?),
                center,
                1.0,
            )?);
        }
        Ok(Space::new(elements))
    }

    /// B-splines of `degree` with the end breakpoints repeated: len(breakpoints) + degree − 1
    /// functions forming a partition of unity on [breakpoints[0], breakpoints[-1]]. Degree 0
    /// gives boxes and degree 1 hats.
    pub fn bsplines_clamped(breakpoints: &[f64], degree: usize) -> Result<Self> {
        check_increasing("breakpoints", breakpoints, 2)?;
        let m = breakpoints.len();
        let mut knots = vec![breakpoints[0]; degree];
        knots.extend_from_slice(breakpoints);
        knots.extend(std::iter::repeat_n(breakpoints[m - 1], degree));
        Space::bsplines(&knots, degree)
    }

    /// Line-shape profiles at `centers`; `profiles` has length 1 (shared) or len(centers).
    pub fn profiles(centers: &[f64], profiles: &[Profile], cutoff: f64) -> Result<Self> {
        let p = broadcast("profiles", profiles, centers.len())?;
        Ok(Space::new(
            centers
                .iter()
                .enumerate()
                .map(|(i, &c)| Element::profile(p(i), c, cutoff))
                .collect::<Result<_>>()?,
        ))
    }

    /// Gaussians at `centers` with standard deviation `sigma` (length 1 or len(centers)),
    /// truncated at `cutoff` standard deviations.
    pub fn gaussians(centers: &[f64], sigma: &[f64], cutoff: f64) -> Result<Self> {
        let profiles = sigma
            .iter()
            .map(|&s| Profile::gaussian(s))
            .collect::<Result<Vec<_>>>()?;
        Space::profiles(centers, &profiles, cutoff)
    }

    /// Boxes [c − w/2, c + w/2) at `centers` with full widths `widths`.
    pub fn rectangles(centers: &[f64], widths: &[f64]) -> Result<Self> {
        let w = broadcast("widths", widths, centers.len())?;
        Ok(Space::new(
            centers
                .iter()
                .enumerate()
                .map(|(i, &c)| Element::boxcar(c - 0.5 * w(i), c + 0.5 * w(i)))
                .collect::<Result<_>>()?,
        ))
    }

    /// Symmetric triangles at `centers` with half width at half maximum `hwhm` (so the base is
    /// 4·hwhm wide).
    pub fn triangles(centers: &[f64], hwhm: &[f64]) -> Result<Self> {
        let h = broadcast("hwhm", hwhm, centers.len())?;
        Ok(Space::new(
            centers
                .iter()
                .enumerate()
                .map(|(i, &c)| Element::hat(c - 2.0 * h(i), c, c + 2.0 * h(i)))
                .collect::<Result<_>>()?,
        ))
    }

    /// Trapezoids at `centers`: 1 within ±top, 0 beyond ±bottom.
    pub fn trapezoids(centers: &[f64], top: &[f64], bottom: &[f64]) -> Result<Self> {
        let (t, b) = (
            broadcast("top", top, centers.len())?,
            broadcast("bottom", bottom, centers.len())?,
        );
        let shared = (top.len() == 1 && bottom.len() == 1)
            .then(|| Shape::trapezoid(top[0], bottom[0]).map(Arc::new))
            .transpose()?;
        Ok(Space::new(
            centers
                .iter()
                .enumerate()
                .map(|(i, &c)| {
                    let shape = match &shared {
                        Some(s) => s.clone(),
                        None => Arc::new(Shape::trapezoid(t(i), b(i))?),
                    };
                    Element::piecewise(shape, c, 1.0)
                })
                .collect::<Result<_>>()?,
        ))
    }

    /// Copies of one shape, element i being shape((x − centers[i]) / stretch[i]).
    pub fn shifted(shape: Shape, centers: &[f64], stretch: &[f64]) -> Result<Self> {
        let stretch = broadcast("stretch", stretch, centers.len())?;
        let shape = Arc::new(shape);
        Ok(Space::new(
            centers
                .iter()
                .enumerate()
                .map(|(i, &c)| Element::piecewise(shape.clone(), c, stretch(i)))
                .collect::<Result<_>>()?,
        ))
    }

    /// A separately tabulated shape per element: element i is the interpolant of
    /// (x[i], y[i]) as a function of the offset from centers[i].
    pub fn tabulated(
        centers: &[f64],
        x: &[Vec<f64>],
        y: &[Vec<f64>],
        interpolation: Interpolation,
    ) -> Result<Self> {
        let n = centers.len();
        if y.len() != n {
            return invalid(format!("y has {} shapes, expected {n}", y.len()));
        }
        if x.len() != 1 && x.len() != n {
            return invalid(format!("x has {} shapes, expected 1 or {n}", x.len()));
        }
        Ok(Space::new(
            centers
                .iter()
                .enumerate()
                .map(|(i, &c)| {
                    let xi = if x.len() == 1 { &x[0] } else { &x[i] };
                    let shape = match interpolation {
                        Interpolation::Linear => Shape::piecewise_linear(xi, &y[i])?,
                        Interpolation::Pchip => Shape::pchip(xi, &y[i])?,
                    };
                    Element::piecewise(Arc::new(shape), c, 1.0)
                })
                .collect::<Result<_>>()?,
        ))
    }

    /// The `order`-th derivatives of the elements (see [`Element::derivative`]).
    pub fn derivative(&self, order: usize) -> Result<Space> {
        Ok(Space::new(
            self.elements
                .iter()
                .map(|e| e.derivative(order))
                .collect::<Result<_>>()?,
        ))
    }

    pub fn len(&self) -> usize {
        self.elements.len()
    }

    pub fn is_empty(&self) -> bool {
        self.elements.is_empty()
    }

    pub fn elements(&self) -> &[Element] {
        &self.elements
    }

    pub fn lower(&self) -> &[f64] {
        &self.lower
    }

    pub fn upper(&self) -> &[f64] {
        &self.upper
    }

    pub fn centers(&self) -> Vec<f64> {
        self.elements.iter().map(Element::center).collect()
    }

    /// ∫ φ_i(x) w(x) dx for each element.
    pub fn integrals(&self, weight: Weight) -> Vec<f64> {
        let mut knots = Vec::new();
        self.elements
            .iter()
            .map(|e| integral(e, weight, &mut knots))
            .collect()
    }

    /// Writes into `out`, in increasing order, the indices of elements whose support [lower,
    /// upper] intersects the closed interval [lo, hi].
    pub(crate) fn candidates(&self, lo: f64, hi: f64, out: &mut Vec<usize>) {
        out.clear();
        let start = self
            .sorted_lower
            .partition_point(|&l| l < lo - self.max_width);
        let end = self.sorted_lower.partition_point(|&l| l <= hi);
        for &j in &self.by_lower[start..end.max(start)] {
            if self.upper[j] >= lo {
                out.push(j);
            }
        }
        out.sort_unstable();
    }
}
