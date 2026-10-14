//! Linear maps between spaces.
//!
//! With f(x) = Σ_j f_j B_j(x) and an inner product weighted by w(x) ([`Weight`], usually
//! uniform):
//!
//! - [`measure`]: g_i = ∫ K_i f w / ∫ K_i w, the response of normalized kernels K_i, e.g. an
//!   instrument line shape applied to a high-resolution spectrum. Exact, sparse, no inversion.
//! - [`evaluate`]: g_i = f(x_i), the special case of delta kernels.
//! - [`Projection`]: the g with Σ_i g_i G_i closest to f in the weighted L2 norm, i.e. the Gram
//!   system ⟨G, G⟩ g = ⟨G, B⟩ f. Use it to change grids (regrid). It conserves ∫ f w whenever
//!   the target basis can represent a constant (boxes, hats, clamped B-splines), and for boxes
//!   it is standard conservative (area-weighted) regridding.
//! - [`Fit`]: the coefficients c in a model space whose measurements best match given data,
//!   min ‖W c − d‖² + λ‖Dᵣ c‖²: smooth conservative upsampling (histopolation), smoothing
//!   splines, and the inverse of [`measure`] in general.

use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};
use rayon::prelude::*;

use crate::banded::BandedCholesky;
use crate::error::{Result, invalid};
use crate::integrate::{Weight, inner};
use crate::space::Space;
use crate::sparse::Csr;

/// The matrix ⟨a_i, b_j⟩_w, shape (len(a), len(b)).
pub fn inner_products(a: &Space, b: &Space, weight: Weight) -> Csr {
    let rows: Vec<(Vec<usize>, Vec<f64>)> = a
        .elements()
        .par_iter()
        .map_init(
            || (Vec::new(), Vec::new()),
            |(candidates, knots), e| {
                b.candidates(e.lower(), e.upper(), candidates);
                let mut idx = Vec::with_capacity(candidates.len());
                let mut val = Vec::with_capacity(candidates.len());
                for &j in candidates.iter() {
                    let v = inner(e, &b.elements()[j], weight, knots);
                    if v != 0.0 {
                        idx.push(j);
                        val.push(v);
                    }
                }
                (idx, val)
            },
        )
        .collect();
    Csr::from_rows(a.len(), b.len(), rows)
}

/// The Gram (mass) matrix ⟨a_i, a_j⟩_w.
pub fn gram(a: &Space, weight: Weight) -> Csr {
    inner_products(a, a, weight)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Normalization {
    /// Divide row i by ∫ K_i w: the kernels are unit-area instrument responses. A kernel that
    /// extends past the end of the basis then sees zero there.
    Integral,
    /// Divide row i by its sum: the weights of each output sum to one, which renormalizes
    /// kernels truncated by the end of the basis (and is what to use for a delta basis).
    RowSum,
    /// Plain inner products ⟨K_i, B_j⟩_w.
    None,
}

/// The matrix W with g = W f, g_i the response of kernel i to f = Σ_j f_j B_j.
pub fn measure(
    kernels: &Space,
    basis: &Space,
    normalization: Normalization,
    weight: Weight,
) -> Csr {
    let mut w = inner_products(kernels, basis, weight);
    let norms = match normalization {
        Normalization::Integral => kernels.integrals(weight),
        Normalization::RowSum => w.row_sums(),
        Normalization::None => return w,
    };
    let factors: Vec<f64> = norms
        .iter()
        .map(|&n| if n != 0.0 { 1.0 / n } else { 0.0 })
        .collect();
    w.scale_rows(&factors);
    w
}

/// The matrix E with E f = f(x): the basis functions evaluated at x, shape (len(x), len(basis)).
pub fn evaluate(basis: &Space, x: &[f64]) -> Result<Csr> {
    Ok(inner_products(&Space::deltas(x)?, basis, Weight::Uniform))
}

/// Solves each row of `rhs` in place with `chol`.
fn solve_rows(chol: &BandedCholesky, rhs: &mut Array2<f64>) {
    rhs.axis_iter_mut(Axis(0))
        .into_par_iter()
        .for_each(|mut row| chol.solve_in_place(row.as_slice_mut().expect("contiguous")));
}

/// The L2 (Galerkin) projection from `src` onto `dst`, kept factored as G⁻¹ M with G the
/// (banded) Gram matrix of `dst` and M = ⟨dst, src⟩; G⁻¹ M itself is generally dense.
#[derive(Debug, Clone)]
pub struct Projection {
    gram: BandedCholesky,
    cross: Csr,
    /// ∫ dst_i w, the mass carried by a unit coefficient.
    dst_mass: Vec<f64>,
}

impl Projection {
    pub fn new(src: &Space, dst: &Space, weight: Weight) -> Result<Self> {
        Ok(Projection {
            gram: BandedCholesky::factor(&gram(dst, weight))?,
            cross: inner_products(dst, src, weight),
            dst_mass: dst.integrals(weight),
        })
    }

    pub fn n_src(&self) -> usize {
        self.cross.ncols()
    }

    pub fn n_dst(&self) -> usize {
        self.cross.nrows()
    }

    /// M = ⟨dst_i, src_j⟩_w.
    pub fn cross(&self) -> &Csr {
        &self.cross
    }

    pub fn apply(&self, f: ArrayView1<f64>) -> Array1<f64> {
        let mut g = self.cross.apply(f);
        self.gram
            .solve_in_place(g.as_slice_mut().expect("fresh array is contiguous"));
        g
    }

    /// Projects each row of f, shape (k, n_src), giving (k, n_dst).
    pub fn apply_last_axis(&self, f: ArrayView2<f64>) -> Array2<f64> {
        let mut g = self.cross.apply_last_axis(f);
        solve_rows(&self.gram, &mut g);
        g
    }

    /// As [`Projection::apply_last_axis`], then limited so that each output coefficient lies
    /// within the range of the source coefficients it overlaps, with the clipped mass
    /// redistributed so that Σ g_i ∫ dst_i w is unchanged (clip and assured sum, Bradley et al.
    /// 2019). Meaningful when coefficients are values: boxes, hats, B-splines.
    pub fn apply_bounded_last_axis(&self, f: ArrayView2<f64>) -> Array2<f64> {
        let mut g = self.apply_last_axis(f);
        ndarray::Zip::from(g.axis_iter_mut(Axis(0)))
            .and(f.axis_iter(Axis(0)))
            .par_for_each(|mut gr, fr| {
                let fr = fr.to_vec();
                caas(
                    gr.as_slice_mut().expect("contiguous"),
                    &fr,
                    &self.cross,
                    &self.dst_mass,
                );
            });
        g
    }

    /// The dense matrix G⁻¹ M, shape (n_dst, n_src).
    pub fn to_dense(&self) -> Array2<f64> {
        // Rows of Mᵀ are columns of M; solving each gives the columns of G⁻¹ M.
        let mut mt = self.cross.transpose().to_dense();
        solve_rows(&self.gram, &mut mt);
        mt.reversed_axes().as_standard_layout().into_owned()
    }
}

/// Clip-and-assured-sum limiter, in place on one projected vector g.
fn caas(g: &mut [f64], f: &[f64], cross: &Csr, mass: &[f64]) {
    let n = g.len();
    let mut lo = vec![0.0; n];
    let mut hi = vec![0.0; n];
    let mut deficit = 0.0; // mass removed by clipping
    for i in 0..n {
        let idx = cross.row(i).0;
        if idx.is_empty() {
            // no source overlap: leave as is
            lo[i] = g[i];
            hi[i] = g[i];
            continue;
        }
        let (mut a, mut b) = (f64::INFINITY, f64::NEG_INFINITY);
        for &j in idx {
            a = a.min(f[j]);
            b = b.max(f[j]);
        }
        lo[i] = a;
        hi[i] = b;
        let c = g[i].clamp(a, b);
        deficit += mass[i] * (g[i] - c);
        g[i] = c;
    }
    if deficit > 0.0 {
        let room: f64 = (0..n).map(|i| mass[i] * (hi[i] - g[i])).sum();
        if room > 0.0 {
            let r = (deficit / room).min(1.0);
            for i in 0..n {
                g[i] += r * (hi[i] - g[i]);
            }
        }
    } else if deficit < 0.0 {
        let room: f64 = (0..n).map(|i| mass[i] * (g[i] - lo[i])).sum();
        if room > 0.0 {
            let r = (-deficit / room).min(1.0);
            for i in 0..n {
                g[i] -= r * (g[i] - lo[i]);
            }
        }
    }
}

/// The r-th order difference matrix, shape (n − r, n).
fn difference_matrix(n: usize, order: usize) -> Csr {
    let mut coeffs = vec![1.0];
    for _ in 0..order {
        let mut next = vec![0.0; coeffs.len() + 1];
        for (k, c) in coeffs.iter().enumerate() {
            next[k + 1] += c;
            next[k] -= c;
        }
        coeffs = next;
    }
    let m = n.saturating_sub(order);
    let rows = (0..m)
        .map(|i| ((i..=i + order).collect(), coeffs.clone()))
        .collect();
    Csr::from_rows(m, n, rows)
}

/// The roughness penalty of a [`Fit`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Penalty {
    /// ∫ (f⁽ʳ⁾)² dx of the fitted function: a smoothing spline. Zero for polynomials of degree
    /// < r whatever the knots; needs a model of degree >= r (r <= 1 for hats).
    Derivative(usize),
    /// Σ (Δʳ c)², differences of neighbouring coefficients (P-splines, Eilers & Marx 1996);
    /// r = 0 is ridge regression. Assumes evenly spaced coefficients.
    Difference(usize),
}

/// Penalized least-squares fit of model coefficients to measurements.
///
/// Finds c minimizing ‖W c − d‖² + λ P(c), W = measure(kernels, model) and P the [`Penalty`].
/// λ = smoothing · tr(WᵀW) / tr(P), so `smoothing` is dimensionless.
#[derive(Debug, Clone)]
pub struct Fit {
    normal: BandedCholesky,
    design: Csr,
    design_t: Csr,
}

impl Fit {
    pub fn new(
        kernels: &Space,
        model: &Space,
        normalization: Normalization,
        weight: Weight,
        smoothing: f64,
        penalty: Penalty,
    ) -> Result<Self> {
        if !(smoothing.is_finite() && smoothing >= 0.0) {
            return invalid("smoothing must be finite and >= 0");
        }
        let design = measure(kernels, model, normalization, weight);
        let mut normal = design.gram_columns();
        if smoothing > 0.0 {
            let p = match penalty {
                Penalty::Difference(r) if model.len() > r => {
                    Some(difference_matrix(model.len(), r).gram_columns())
                }
                Penalty::Difference(_) => None,
                Penalty::Derivative(r) => Some(gram(&model.derivative(r)?, Weight::Uniform)),
            };
            if let Some(p) = p {
                let tr = p.trace();
                if tr <= 0.0 {
                    return invalid(format!(
                        "the {penalty:?} penalty is zero for this model (degree too low?)"
                    ));
                }
                normal = normal.add_scaled(&p, smoothing * normal.trace() / tr);
            }
        }
        let normal = BandedCholesky::factor(&normal).map_err(|e| match e {
            crate::Error::Singular(m) => crate::Error::Singular(format!(
                "{m}; the data do not determine every model coefficient, so add smoothing"
            )),
            other => other,
        })?;
        let design_t = design.transpose();
        Ok(Fit {
            normal,
            design,
            design_t,
        })
    }

    pub fn n_data(&self) -> usize {
        self.design.nrows()
    }

    pub fn n_model(&self) -> usize {
        self.design.ncols()
    }

    /// W = measure(kernels, model): the fitted model's predicted data are W c.
    pub fn design(&self) -> &Csr {
        &self.design
    }

    pub fn apply(&self, data: ArrayView1<f64>) -> Array1<f64> {
        let mut c = self.design_t.apply(data);
        self.normal
            .solve_in_place(c.as_slice_mut().expect("contiguous"));
        c
    }

    /// Fits each row of `data`, shape (k, n_data), giving (k, n_model).
    pub fn apply_last_axis(&self, data: ArrayView2<f64>) -> Array2<f64> {
        let mut c = self.design_t.apply_last_axis(data);
        solve_rows(&self.normal, &mut c);
        c
    }

    /// The dense map from data to coefficients, shape (n_model, n_data).
    pub fn to_dense(&self) -> Array2<f64> {
        // rows of W are columns of Wᵀ
        let mut w = self.design.to_dense();
        solve_rows(&self.normal, &mut w);
        w.reversed_axes().as_standard_layout().into_owned()
    }
}
