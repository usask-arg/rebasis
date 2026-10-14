//! Linear maps between function bases.
//!
//! A quantity f(x) = Σ_j f_j B_j(x) is described by coefficients in a [`Space`] of
//! [`Element`]s. rebasis computes the sparse matrices that take f_j to coefficients or
//! measurements in another space: instrument line shapes ([`measure`]), point evaluation
//! ([`evaluate`]) and L2 projection for regridding ([`Projection`]).
//!
//! All inner products are exact for piecewise-polynomial elements (boxes, hats, tabulated
//! shapes) and accurate to rounding for Gaussians.

// Index loops read most clearly in the banded and sparse kernels.
#![allow(clippy::needless_range_loop)]

mod banded;
pub mod element;
pub mod error;
mod inner;
pub mod ops;
mod quadrature;
pub mod space;
pub mod sparse;

pub use banded::BandedCholesky;
pub use element::{Element, Segment, Shape};
pub use error::{Error, Result};
pub use inner::inner;
pub use ops::{Normalization, Projection, evaluate, gram, inner_products, measure};
pub use space::Space;
pub use sparse::Csr;
