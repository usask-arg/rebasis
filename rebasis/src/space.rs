//! A space is an ordered set of elements: the basis {B_j} that coefficients f_j refer to, or a
//! set of measurement kernels {K_i}.

use std::sync::Arc;

use crate::element::{Element, Shape};
use crate::error::{Result, invalid};

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

/// Broadcasts a length-1 parameter array to `n`, or checks it has length `n`.
fn broadcast<'a>(
    name: &str,
    values: &'a [f64],
    n: usize,
) -> Result<Box<dyn Fn(usize) -> f64 + 'a>> {
    match values.len() {
        1 => Ok(Box::new(move |_| values[0])),
        m if m == n => Ok(Box::new(move |i| values[i])),
        m => invalid(format!("{name} has length {m}, expected 1 or {n}")),
    }
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

    /// Gaussians at `centers` with standard deviation `sigma` (length 1 or len(centers)),
    /// truncated at `cutoff` standard deviations.
    pub fn gaussians(centers: &[f64], sigma: &[f64], cutoff: f64) -> Result<Self> {
        let sigma = broadcast("sigma", sigma, centers.len())?;
        Ok(Space::new(
            centers
                .iter()
                .enumerate()
                .map(|(i, &c)| Element::gaussian(c, sigma(i), cutoff))
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

    pub fn integrals(&self) -> Vec<f64> {
        self.elements.iter().map(Element::integral).collect()
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
