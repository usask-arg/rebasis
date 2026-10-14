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
use crate::integrate::{Weight, inner, inner_derivative, integral, integral_derivative};
use crate::profile::Parameter;
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

/// The measurement matrix W of [`measure`] and its derivative ∂W/∂p, row i with respect to the
/// parameter p of kernel i (each kernel has its own parameter, so the Jacobian of g = W f with
/// respect to the kernels' p is diagonal, with diagonal ∂W/∂p f).
///
/// Derivatives are of the kernels as truncated (their supports move with p) and normalized
/// (`Integral` divides by ∫ K w, `RowSum` by the row sum), so they are the exact derivatives
/// of what [`measure`] computes. With `Normalization::None` they depend on each profile's
/// peak normalization (1 at the center).
pub fn measure_derivative(
    kernels: &Space,
    basis: &Space,
    normalization: Normalization,
    weight: Weight,
    p: Parameter,
) -> Result<(Csr, Csr)> {
    type Row = (Vec<usize>, Vec<f64>, Vec<f64>, f64, f64);
    let rows: Vec<Result<Row>> = kernels
        .elements()
        .par_iter()
        .map_init(
            || (Vec::new(), Vec::new()),
            |(candidates, knots), k| {
                let jumps = k.jumps(p)?;
                basis.candidates(k.lower(), k.upper(), candidates);
                let mut idx = Vec::with_capacity(candidates.len());
                let mut val = Vec::with_capacity(candidates.len());
                let mut der = Vec::with_capacity(candidates.len());
                for &j in candidates.iter() {
                    let b = &basis.elements()[j];
                    let v = inner(k, b, weight, knots);
                    let d = inner_derivative(k, b, p, weight, &jumps, knots)?;
                    if v != 0.0 || d != 0.0 {
                        idx.push(j);
                        val.push(v);
                        der.push(d);
                    }
                }
                let (n, dn) = match normalization {
                    Normalization::Integral => (
                        integral(k, weight, knots),
                        integral_derivative(k, p, weight, &jumps, knots)?,
                    ),
                    Normalization::RowSum => (val.iter().sum(), der.iter().sum()),
                    Normalization::None => (1.0, 0.0),
                };
                Ok((idx, val, der, n, dn))
            },
        )
        .collect();
    let mut w_rows = Vec::with_capacity(rows.len());
    let mut d_rows = Vec::with_capacity(rows.len());
    for row in rows {
        let (idx, val, der, n, dn) = row?;
        let inv = if n != 0.0 { 1.0 / n } else { 0.0 };
        // W = A / N, ∂W = (∂A − W ∂N) / N
        let w: Vec<f64> = val.iter().map(|v| v * inv).collect();
        let d: Vec<f64> = der
            .iter()
            .zip(&w)
            .map(|(a, wi)| (a - wi * dn) * inv)
            .collect();
        w_rows.push((idx.clone(), w));
        d_rows.push((idx, d));
    }
    Ok((
        Csr::from_rows(kernels.len(), basis.len(), w_rows),
        Csr::from_rows(kernels.len(), basis.len(), d_rows),
    ))
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

/// Bounds for [`Projection::apply_bounded_last_axis`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Bounds {
    /// Each output coefficient within the range of the source coefficients it overlaps: no new
    /// extrema. Meaningful when coefficients are values (boxes, hats, B-splines).
    Local,
    /// Every output coefficient in [lo, hi], e.g. lo = 0, hi = ∞ for positivity.
    Global { lo: f64, hi: f64 },
}

/// The L2 (Galerkin) projection from `src` onto `dst`, kept factored as G⁻¹ M with G the
/// (banded) Gram matrix of `dst` and M = ⟨dst, src⟩; G⁻¹ M itself is generally dense.
#[derive(Debug, Clone)]
pub struct Projection {
    gram_matrix: Csr,
    gram: BandedCholesky,
    cross: Csr,
    /// ∫ dst_i w, the mass carried by a unit coefficient.
    dst_mass: Vec<f64>,
}

impl Projection {
    pub fn new(src: &Space, dst: &Space, weight: Weight) -> Result<Self> {
        let gram_matrix = gram(dst, weight);
        Ok(Projection {
            gram: BandedCholesky::factor(&gram_matrix)?,
            gram_matrix,
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

    /// The best bounded approximation: the coefficients c minimizing the weighted L2 error
    /// ‖f − Σ c_i dst_i‖ subject to the bounds and to conservation, Σ c_i ∫ dst_i w = Σ g_i
    /// ∫ dst_i w for the unbounded projection g. Equivalently, c minimizes (c − g)ᵀ G (c − g),
    /// a quadratic program with a banded Hessian, solved by a primal-dual active-set method.
    /// Because the error is measured on the function, corrections stay local: clipped mass is
    /// put back next to where it was removed.
    pub fn apply_bounded_last_axis(&self, f: ArrayView2<f64>, bounds: Bounds) -> Array2<f64> {
        let mut rhs = self.cross.apply_last_axis(f);
        let mut g = rhs.clone();
        solve_rows(&self.gram, &mut g);
        ndarray::Zip::from(g.axis_iter_mut(Axis(0)))
            .and(rhs.axis_iter_mut(Axis(0)))
            .and(f.axis_iter(Axis(0)))
            .par_for_each(|mut gr, r, fr| {
                let (lo, hi) = self.bounds(&fr.to_vec(), bounds);
                let g0 = gr.to_vec();
                let r = r.to_vec();
                let c = bounded_least_squares(&self.gram_matrix, &r, &g0, &self.dst_mass, &lo, &hi)
                    .unwrap_or_else(|| {
                        let mut c = g0.clone();
                        caas(&mut c, &self.dst_mass, &lo, &hi);
                        c
                    });
                gr.assign(&ArrayView1::from(&c));
            });
        g
    }

    /// Per-coefficient bounds for one source vector.
    fn bounds(&self, f: &[f64], bounds: Bounds) -> (Vec<f64>, Vec<f64>) {
        let n = self.n_dst();
        match bounds {
            Bounds::Global { lo, hi } => (vec![lo; n], vec![hi; n]),
            Bounds::Local => (0..n)
                .map(|i| {
                    let idx = self.cross.row(i).0;
                    if idx.is_empty() {
                        (f64::NEG_INFINITY, f64::INFINITY)
                    } else {
                        idx.iter()
                            .fold((f64::INFINITY, f64::NEG_INFINITY), |(a, b), &j| {
                                (a.min(f[j]), b.max(f[j]))
                            })
                    }
                })
                .unzip(),
        }
    }

    /// The dense matrix G⁻¹ M, shape (n_dst, n_src).
    pub fn to_dense(&self) -> Array2<f64> {
        // Rows of Mᵀ are columns of M; solving each gives the columns of G⁻¹ M.
        let mut mt = self.cross.transpose().to_dense();
        solve_rows(&self.gram, &mut mt);
        mt.reversed_axes().as_standard_layout().into_owned()
    }
}

/// min ½ cᵀGc − rᵀc subject to lo <= c <= hi and mᵀc = mᵀg (g = G⁻¹r, the unconstrained
/// minimizer), by the primal-dual active-set method (Hintermüller, Ito & Kunisch 2002). Returns
/// None if it does not converge or a reduced system is singular.
fn bounded_least_squares(
    gram: &Csr,
    r: &[f64],
    g: &[f64],
    m: &[f64],
    lo: &[f64],
    hi: &[f64],
) -> Option<Vec<f64>> {
    #[derive(Clone, Copy, PartialEq)]
    enum State {
        Free,
        Lower,
        Upper,
    }
    let n = g.len();
    let target: f64 = m.iter().zip(g).map(|(a, b)| a * b).sum();
    let diag: Vec<f64> = (0..n)
        .map(|i| {
            let (idx, val) = gram.row(i);
            idx.iter().position(|&j| j == i).map_or(1.0, |k| val[k])
        })
        .collect();
    let mut c: Vec<f64> = (0..n).map(|i| g[i].clamp(lo[i], hi[i])).collect();
    let mut lambda = vec![0.0; n];
    let mut state = vec![State::Free; n];
    // start from the unconstrained solution's violations
    for i in 0..n {
        if g[i] > hi[i] {
            state[i] = State::Upper;
        } else if g[i] < lo[i] {
            state[i] = State::Lower;
        }
    }
    for _ in 0..100 {
        // solve with the active set fixed at its bounds
        for i in 0..n {
            match state[i] {
                State::Lower => c[i] = lo[i],
                State::Upper => c[i] = hi[i],
                State::Free => {}
            }
        }
        let free: Vec<usize> = (0..n).filter(|&i| state[i] == State::Free).collect();
        let mu;
        if free.is_empty() {
            mu = 0.0;
        } else {
            let sub = gram.submatrix(&free);
            let chol = BandedCholesky::factor(&sub).ok()?;
            // a = G_FF⁻¹ (r_F − G_FA c_A), b = G_FF⁻¹ m_F, c_F = a − μ b
            let mut a: Vec<f64> = free
                .iter()
                .map(|&i| {
                    let (idx, val) = gram.row(i);
                    let fixed: f64 = idx
                        .iter()
                        .zip(val)
                        .filter(|&(&j, _)| state[j] != State::Free)
                        .map(|(&j, v)| v * c[j])
                        .sum();
                    r[i] - fixed
                })
                .collect();
            let mut b: Vec<f64> = free.iter().map(|&i| m[i]).collect();
            chol.solve_in_place(&mut a);
            chol.solve_in_place(&mut b);
            let mb: f64 = free.iter().zip(&b).map(|(&i, v)| m[i] * v).sum();
            let ma: f64 = free.iter().zip(&a).map(|(&i, v)| m[i] * v).sum();
            let fixed_mass: f64 = (0..n)
                .filter(|&i| state[i] != State::Free)
                .map(|i| m[i] * c[i])
                .sum();
            mu = if mb.abs() > 0.0 {
                (ma + fixed_mass - target) / mb
            } else {
                0.0
            };
            for (k, &i) in free.iter().enumerate() {
                c[i] = a[k] - mu * b[k];
            }
        }
        // multipliers λ = r − G c − μ m on the active set, 0 on the free set
        for i in 0..n {
            lambda[i] = if state[i] == State::Free {
                0.0
            } else {
                let (idx, val) = gram.row(i);
                let gc: f64 = idx.iter().zip(val).map(|(&j, v)| v * c[j]).sum();
                r[i] - gc - mu * m[i]
            };
        }
        // new active set: λ_i + κ_i (c_i − bound_i) pushes outward
        let mut changed = false;
        for i in 0..n {
            let new = if lo[i] == hi[i] {
                State::Lower
            } else if lambda[i] + diag[i] * (c[i] - hi[i]) > 0.0 {
                State::Upper
            } else if lambda[i] + diag[i] * (c[i] - lo[i]) < 0.0 {
                State::Lower
            } else {
                State::Free
            };
            if new != state[i] {
                changed = true;
                state[i] = new;
            }
        }
        if !changed {
            return Some(c);
        }
    }
    None
}

/// Clip and assured sum (Bradley et al. 2019): clip to [lo, hi], then put the clipped mass
/// back in proportion to each coefficient's room. A fallback for the active-set solver.
fn caas(g: &mut [f64], mass: &[f64], lo: &[f64], hi: &[f64]) {
    let n = g.len();
    let mut deficit = 0.0; // mass removed by clipping
    for i in 0..n {
        let c = g[i].clamp(lo[i], hi[i]);
        deficit += mass[i] * (g[i] - c);
        g[i] = c;
    }
    let room = |i: usize, up: bool| {
        let r = if up { hi[i] - g[i] } else { g[i] - lo[i] };
        if r.is_finite() { r } else { 0.0 }
    };
    if deficit != 0.0 {
        let up = deficit > 0.0;
        let total: f64 = (0..n).map(|i| mass[i] * room(i, up)).sum();
        if total > 0.0 {
            let f = (deficit.abs() / total).min(1.0);
            let shifts: Vec<f64> = (0..n).map(|i| f * room(i, up)).collect();
            for i in 0..n {
                g[i] += if up { shifts[i] } else { -shifts[i] };
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
