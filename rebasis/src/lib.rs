//! Linear maps between function bases.
//!
//! A quantity f(x) = Σ_j f_j B_j(x) is described by coefficients in a [`Space`] of
//! [`Element`]s. rebasis computes the sparse matrices that take f_j to coefficients or
//! measurements in another space: instrument line shapes ([`measure`]), point evaluation
//! ([`evaluate`]), L2 projection for regridding ([`Projection`]) and penalized least-squares
//! fitting for smooth conservative upsampling and smoothing ([`Fit`]).
//!
//! All inner products are exact for piecewise-polynomial elements (boxes, hats, B-splines,
//! tabulated shapes) and accurate to rounding for the analytic profiles ([`Profile`]).

// Index loops read most clearly in the banded and sparse kernels.
#![allow(clippy::needless_range_loop)]

mod banded;
pub mod element;
pub mod error;
mod integrate;
pub mod ops;
pub mod profile;
mod quadrature;
pub mod space;
pub mod sparse;
pub mod special;

pub use banded::BandedCholesky;
pub use element::{Element, Segment, Shape};
pub use error::{Error, Result};
pub use integrate::{Weight, inner, inner_derivative, integral, integral_derivative};
pub use ops::{
    Bounds, Fit, Normalization, Penalty, Projection, evaluate, gram, inner_products, measure,
    measure_derivative,
};
pub use profile::{Apodization, FWHM_PER_SIGMA, Parameter, Profile};
pub use space::{Interpolation, Space};
pub use sparse::Csr;
