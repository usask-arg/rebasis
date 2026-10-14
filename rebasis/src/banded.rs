//! Cholesky factorization of symmetric positive-definite banded matrices (Gram matrices of
//! local bases).

use crate::error::{Error, Result};
use crate::sparse::Csr;

#[derive(Debug, Clone)]
pub struct BandedCholesky {
    n: usize,
    /// Half bandwidth.
    p: usize,
    /// Row i holds L[i, i-p..=i] at l[i*(p+1)..(i+1)*(p+1)], left-padded for i < p.
    l: Vec<f64>,
}

impl BandedCholesky {
    #[inline]
    fn at(&self, i: usize, j: usize) -> f64 {
        self.l[i * (self.p + 1) + (j + self.p - i)]
    }

    /// Factors the symmetric matrix `a`; only its lower triangle is read.
    #[allow(clippy::neg_cmp_op_on_partial_ord)]
    pub fn factor(a: &Csr) -> Result<Self> {
        assert_eq!(a.nrows(), a.ncols(), "matrix must be square");
        let n = a.nrows();
        let p = a.bandwidth();
        let w = p + 1;
        let mut l = vec![0.0; n * w];
        for i in 0..n {
            let (idx, val) = a.row(i);
            for (&j, &v) in idx.iter().zip(val) {
                if j <= i {
                    l[i * w + (j + p - i)] = v;
                }
            }
        }
        let diag: Vec<f64> = (0..n).map(|i| l[i * w + p]).collect();
        let mut chol = BandedCholesky { n, p, l };
        for i in 0..n {
            let k0 = i.saturating_sub(p);
            for j in k0..=i {
                let mut s = chol.at(i, j);
                for k in k0..j {
                    s -= chol.at(i, k) * chol.at(j, k);
                }
                let pos = i * w + (j + p - i);
                if i == j {
                    let scale = diag[i].abs();
                    // negated so that a NaN pivot is also rejected
                    if !(s > 1e-13 * scale) {
                        return Err(Error::Singular(format!(
                            "pivot {i} is {s:e} (diagonal {scale:e}): the target elements are \
                             linearly dependent, or one of them has zero norm"
                        )));
                    }
                    chol.l[pos] = s.sqrt();
                } else {
                    chol.l[pos] = s / chol.at(j, j);
                }
            }
        }
        Ok(chol)
    }

    pub fn n(&self) -> usize {
        self.n
    }

    /// Overwrites b with A⁻¹ b.
    pub fn solve_in_place(&self, b: &mut [f64]) {
        assert_eq!(b.len(), self.n);
        for i in 0..self.n {
            let mut s = b[i];
            for k in i.saturating_sub(self.p)..i {
                s -= self.at(i, k) * b[k];
            }
            b[i] = s / self.at(i, i);
        }
        for i in (0..self.n).rev() {
            let mut s = b[i];
            for k in i + 1..(i + self.p + 1).min(self.n) {
                s -= self.at(k, i) * b[k];
            }
            b[i] = s / self.at(i, i);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn solves_tridiagonal() {
        // A = tridiag(1, 4, 1), n = 5
        let n = 5;
        let rows = (0..n)
            .map(|i| {
                let mut idx = vec![];
                let mut val = vec![];
                if i > 0 {
                    idx.push(i - 1);
                    val.push(1.0);
                }
                idx.push(i);
                val.push(4.0);
                if i + 1 < n {
                    idx.push(i + 1);
                    val.push(1.0);
                }
                (idx, val)
            })
            .collect();
        let a = Csr::from_rows(n, n, rows);
        let chol = BandedCholesky::factor(&a).unwrap();
        let x_true = [1.0, -2.0, 3.0, 0.5, 7.0];
        let mut b: Vec<f64> = a.apply(ndarray::ArrayView1::from(&x_true)).to_vec();
        chol.solve_in_place(&mut b);
        for (x, t) in b.iter().zip(x_true) {
            assert!((x - t).abs() < 1e-14);
        }
    }

    #[test]
    fn singular_is_an_error() {
        let a = Csr::from_rows(
            2,
            2,
            vec![(vec![0, 1], vec![1.0, 1.0]), (vec![0, 1], vec![1.0, 1.0])],
        );
        assert!(matches!(
            BandedCholesky::factor(&a),
            Err(Error::Singular(_))
        ));
    }
}
