//! Compressed sparse row matrices: the output of every rebasis operation.

use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis, Zip};

#[derive(Debug, Clone, PartialEq)]
pub struct Csr {
    nrows: usize,
    ncols: usize,
    indptr: Vec<usize>,
    indices: Vec<usize>,
    data: Vec<f64>,
}

impl Csr {
    /// Builds from per-row (column indices, values); indices within a row must be increasing.
    pub fn from_rows(nrows: usize, ncols: usize, rows: Vec<(Vec<usize>, Vec<f64>)>) -> Self {
        debug_assert_eq!(rows.len(), nrows);
        let nnz = rows.iter().map(|r| r.0.len()).sum();
        let mut indptr = Vec::with_capacity(nrows + 1);
        let mut indices = Vec::with_capacity(nnz);
        let mut data = Vec::with_capacity(nnz);
        indptr.push(0);
        for (idx, val) in rows {
            indices.extend_from_slice(&idx);
            data.extend_from_slice(&val);
            indptr.push(indices.len());
        }
        Csr {
            nrows,
            ncols,
            indptr,
            indices,
            data,
        }
    }

    pub fn nrows(&self) -> usize {
        self.nrows
    }

    pub fn ncols(&self) -> usize {
        self.ncols
    }

    pub fn nnz(&self) -> usize {
        self.data.len()
    }

    pub fn indptr(&self) -> &[usize] {
        &self.indptr
    }

    pub fn indices(&self) -> &[usize] {
        &self.indices
    }

    pub fn data(&self) -> &[f64] {
        &self.data
    }

    pub fn row(&self, i: usize) -> (&[usize], &[f64]) {
        let r = self.indptr[i]..self.indptr[i + 1];
        (&self.indices[r.clone()], &self.data[r])
    }

    pub fn row_sums(&self) -> Vec<f64> {
        (0..self.nrows)
            .map(|i| self.row(i).1.iter().sum())
            .collect()
    }

    pub fn scale_rows(&mut self, factors: &[f64]) {
        for i in 0..self.nrows {
            for v in &mut self.data[self.indptr[i]..self.indptr[i + 1]] {
                *v *= factors[i];
            }
        }
    }

    /// Largest |i − j| over the stored entries.
    pub fn bandwidth(&self) -> usize {
        (0..self.nrows)
            .flat_map(|i| self.row(i).0.iter().map(move |&j| i.abs_diff(j)))
            .max()
            .unwrap_or(0)
    }

    #[inline]
    fn row_dot(&self, i: usize, x: &[f64]) -> f64 {
        let (idx, val) = self.row(i);
        idx.iter().zip(val).map(|(&j, v)| v * x[j]).sum()
    }

    /// A x for x of length ncols.
    pub fn apply(&self, x: ArrayView1<f64>) -> Array1<f64> {
        assert_eq!(x.len(), self.ncols, "x has the wrong length");
        let x = x.to_vec();
        Array1::from_shape_fn(self.nrows, |i| self.row_dot(i, &x))
    }

    /// Applies A along the last axis: x has shape (k, ncols), the result (k, nrows). This is the
    /// layout of sasktran2 outputs, with the spectral dimension last.
    pub fn apply_last_axis(&self, x: ArrayView2<f64>) -> Array2<f64> {
        assert_eq!(x.ncols(), self.ncols, "x has the wrong number of columns");
        let mut out = Array2::zeros((x.nrows(), self.nrows));
        Zip::from(out.axis_iter_mut(Axis(0)))
            .and(x.axis_iter(Axis(0)))
            .par_for_each(|mut o, xr| {
                let xr = xr.to_vec();
                for (i, v) in o.iter_mut().enumerate() {
                    *v = self.row_dot(i, &xr);
                }
            });
        out
    }

    pub fn transpose(&self) -> Csr {
        let mut rows: Vec<(Vec<usize>, Vec<f64>)> = vec![(Vec::new(), Vec::new()); self.ncols];
        for i in 0..self.nrows {
            let (idx, val) = self.row(i);
            for (&j, &v) in idx.iter().zip(val) {
                rows[j].0.push(i);
                rows[j].1.push(v);
            }
        }
        Csr::from_rows(self.ncols, self.nrows, rows)
    }

    pub fn to_dense(&self) -> Array2<f64> {
        let mut out = Array2::zeros((self.nrows, self.ncols));
        for i in 0..self.nrows {
            let (idx, val) = self.row(i);
            for (&j, &v) in idx.iter().zip(val) {
                out[[i, j]] = v;
            }
        }
        out
    }
}
