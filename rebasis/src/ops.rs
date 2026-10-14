//! Linear maps between spaces.
//!
//! With f(x) = Σ_j f_j B_j(x):
//!
//! - [`measure`]: g_i = ∫ K_i f / ∫ K_i, the response of normalized kernels K_i, e.g. an
//!   instrument line shape applied to a high-resolution spectrum. Exact, sparse, no inversion.
//! - [`evaluate`]: g_i = f(x_i), the special case of delta kernels.
//! - [`Projection`]: the g with Σ_i g_i G_i closest to f in L2, i.e. the Gram system
//!   ⟨G, G⟩ g = ⟨G, B⟩ f. Use it to change grids (regrid) rather than to simulate an
//!   instrument. It conserves ∫ f whenever the target basis can represent a constant, as boxes
//!   and hats can, and for boxes it is standard conservative (area-weighted) regridding.

use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};
use rayon::prelude::*;

use crate::banded::BandedCholesky;
use crate::error::Result;
use crate::inner::inner;
use crate::space::Space;
use crate::sparse::Csr;

/// The matrix ⟨a_i, b_j⟩, shape (len(a), len(b)).
pub fn inner_products(a: &Space, b: &Space) -> Csr {
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
                    let v = inner(e, &b.elements()[j], knots);
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

/// The Gram (mass) matrix ⟨a_i, a_j⟩.
pub fn gram(a: &Space) -> Csr {
    inner_products(a, a)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Normalization {
    /// Divide row i by ∫ K_i: the kernels are unit-area instrument responses. A kernel that
    /// extends past the end of the basis then sees zero there.
    Integral,
    /// Divide row i by its sum: the weights of each output sum to one, which renormalizes
    /// kernels truncated by the end of the basis (and is what to use for a delta basis).
    RowSum,
    /// Plain inner products ⟨K_i, B_j⟩.
    None,
}

/// The matrix W with g = W f, g_i the response of kernel i to f = Σ_j f_j B_j.
pub fn measure(kernels: &Space, basis: &Space, normalization: Normalization) -> Csr {
    let mut w = inner_products(kernels, basis);
    let norms = match normalization {
        Normalization::Integral => kernels.integrals(),
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
    Ok(inner_products(&Space::deltas(x)?, basis))
}

/// The L2 (Galerkin) projection from `src` onto `dst`, kept factored as G⁻¹ M with G the
/// (banded) Gram matrix of `dst` and M = ⟨dst, src⟩; G⁻¹ M itself is generally dense.
#[derive(Debug, Clone)]
pub struct Projection {
    gram: BandedCholesky,
    cross: Csr,
}

impl Projection {
    pub fn new(src: &Space, dst: &Space) -> Result<Self> {
        Ok(Projection {
            gram: BandedCholesky::factor(&gram(dst))?,
            cross: inner_products(dst, src),
        })
    }

    pub fn n_src(&self) -> usize {
        self.cross.ncols()
    }

    pub fn n_dst(&self) -> usize {
        self.cross.nrows()
    }

    /// M = ⟨dst_i, src_j⟩.
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
        g.axis_iter_mut(Axis(0))
            .into_par_iter()
            .for_each(|mut row| {
                self.gram
                    .solve_in_place(row.as_slice_mut().expect("contiguous"))
            });
        g
    }

    /// The dense matrix G⁻¹ M, shape (n_dst, n_src).
    pub fn to_dense(&self) -> Array2<f64> {
        // Rows of Mᵀ are columns of M; solving each gives the columns of G⁻¹ M.
        let mut mt = self.cross.transpose().to_dense();
        mt.axis_iter_mut(Axis(0))
            .into_par_iter()
            .for_each(|mut row| {
                self.gram
                    .solve_in_place(row.as_slice_mut().expect("contiguous"))
            });
        mt.reversed_axes().as_standard_layout().into_owned()
    }
}
