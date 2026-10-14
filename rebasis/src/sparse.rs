//! Compressed sparse row matrices: the output of every rebasis operation.

use ndarray::parallel::prelude::*;
use ndarray::{
    Array1, Array2, ArrayD, ArrayView1, ArrayView2, ArrayViewD, ArrayViewMut1, Axis, Zip,
};

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

    /// o = A x for one lane x of length ncols (copied only if it is strided).
    fn apply_lane(&self, x: ArrayView1<f64>, mut o: ArrayViewMut1<f64>) {
        let copied;
        let x = match x.as_slice() {
            Some(s) => s,
            None => {
                copied = x.to_vec();
                &copied
            }
        };
        for (i, v) in o.iter_mut().enumerate() {
            *v = self.row_dot(i, x);
        }
    }

    /// A x for x of length ncols.
    pub fn apply(&self, x: ArrayView1<f64>) -> Array1<f64> {
        assert_eq!(x.len(), self.ncols, "x has the wrong length");
        let mut out = Array1::zeros(self.nrows);
        self.apply_lane(x, out.view_mut());
        out
    }

    /// Applies A along the last axis: x has shape (k, ncols), the result (k, nrows), e.g.
    /// sasktran2 optical properties of shape (location, wavelength).
    pub fn apply_last_axis(&self, x: ArrayView2<f64>) -> Array2<f64> {
        assert_eq!(x.ncols(), self.ncols, "x has the wrong number of columns");
        self.apply_axis(x.into_dyn(), x.ndim() - 1)
            .into_dimensionality()
            .expect("two dimensions in, two out")
    }

    /// Applies A along `axis` of x, whose length there must be ncols; the result has nrows
    /// there and the shape of x elsewhere. E.g. sasktran2 radiances (wavelength, los, stokes)
    /// along axis 0, or their derivatives (parameter, wavelength, los, stokes) along axis 1.
    pub fn apply_axis(&self, x: ArrayViewD<f64>, axis: usize) -> ArrayD<f64> {
        assert!(
            axis < x.ndim(),
            "axis {axis} out of range for {} dimensions",
            x.ndim()
        );
        assert_eq!(
            x.len_of(Axis(axis)),
            self.ncols,
            "x has the wrong length along axis"
        );
        let mut shape = x.shape().to_vec();
        shape[axis] = self.nrows;
        let mut out = ArrayD::zeros(shape);
        if axis + 1 == x.ndim() {
            // One sparse dot product per output value, over contiguous lanes.
            Zip::from(out.lanes_mut(Axis(axis)))
                .and(x.lanes(Axis(axis)))
                .par_for_each(|o, xl| self.apply_lane(xl, o));
        } else {
            // Output slab i is Σ_j a_ij (input slab j): whole slabs at a time, which keeps
            // the memory access contiguous when the axis is an outer one.
            out.axis_iter_mut(Axis(axis))
                .into_par_iter()
                .enumerate()
                .for_each(|(i, mut o)| {
                    let (idx, val) = self.row(i);
                    for (&j, &v) in idx.iter().zip(val) {
                        o.scaled_add(v, &x.index_axis(Axis(axis), j));
                    }
                });
        }
        out
    }

    /// Applies Aᵀ along `axis` of x, whose length there must be nrows (the adjoint of
    /// [`Csr::apply_axis`], for reverse-mode derivatives).
    pub fn apply_transpose_axis(&self, x: ArrayViewD<f64>, axis: usize) -> ArrayD<f64> {
        self.transpose().apply_axis(x, axis)
    }

    /// AᵀA, accumulated in band storage (its half bandwidth is the widest column span of a row
    /// of A, so this suits matrices whose rows are local).
    pub fn gram_columns(&self) -> Csr {
        let n = self.ncols;
        let p = (0..self.nrows)
            .filter_map(|i| {
                let idx = self.row(i).0;
                Some(idx.last()? - idx.first()?)
            })
            .max()
            .unwrap_or(0);
        let w = 2 * p + 1;
        let mut band = vec![0.0; n * w];
        for r in 0..self.nrows {
            let (idx, val) = self.row(r);
            for (&i, &vi) in idx.iter().zip(val) {
                for (&j, &vj) in idx.iter().zip(val) {
                    band[i * w + (j + p - i)] += vi * vj;
                }
            }
        }
        let rows = (0..n)
            .map(|i| {
                let mut idx = Vec::new();
                let mut val = Vec::new();
                for off in 0..w {
                    let j = i + off;
                    if j < p || j - p >= n {
                        continue;
                    }
                    let v = band[i * w + off];
                    if v != 0.0 {
                        idx.push(j - p);
                        val.push(v);
                    }
                }
                (idx, val)
            })
            .collect();
        Csr::from_rows(n, n, rows)
    }

    /// self + alpha · other (same shape).
    pub fn add_scaled(&self, other: &Csr, alpha: f64) -> Csr {
        assert_eq!((self.nrows, self.ncols), (other.nrows, other.ncols));
        let rows = (0..self.nrows)
            .map(|i| {
                let (ia, va) = self.row(i);
                let (ib, vb) = other.row(i);
                let (mut p, mut q) = (0, 0);
                let mut idx = Vec::with_capacity(ia.len() + ib.len());
                let mut val = Vec::with_capacity(ia.len() + ib.len());
                while p < ia.len() || q < ib.len() {
                    let ja = ia.get(p).copied().unwrap_or(usize::MAX);
                    let jb = ib.get(q).copied().unwrap_or(usize::MAX);
                    match ja.cmp(&jb) {
                        std::cmp::Ordering::Less => {
                            idx.push(ja);
                            val.push(va[p]);
                            p += 1;
                        }
                        std::cmp::Ordering::Greater => {
                            idx.push(jb);
                            val.push(alpha * vb[q]);
                            q += 1;
                        }
                        std::cmp::Ordering::Equal => {
                            idx.push(ja);
                            val.push(va[p] + alpha * vb[q]);
                            p += 1;
                            q += 1;
                        }
                    }
                }
                (idx, val)
            })
            .collect();
        Csr::from_rows(self.nrows, self.ncols, rows)
    }

    pub fn trace(&self) -> f64 {
        (0..self.nrows.min(self.ncols))
            .map(|i| {
                let (idx, val) = self.row(i);
                idx.iter().position(|&j| j == i).map_or(0.0, |k| val[k])
            })
            .sum()
    }

    /// The square submatrix on rows and columns `keep` (increasing), renumbered 0..len.
    pub fn submatrix(&self, keep: &[usize]) -> Csr {
        let mut map = vec![usize::MAX; self.ncols.max(self.nrows)];
        for (k, &i) in keep.iter().enumerate() {
            map[i] = k;
        }
        let rows = keep
            .iter()
            .map(|&i| {
                let (idx, val) = self.row(i);
                idx.iter()
                    .zip(val)
                    .filter(|&(&j, _)| map[j] != usize::MAX)
                    .map(|(&j, &v)| (map[j], v))
                    .unzip()
            })
            .collect();
        Csr::from_rows(keep.len(), keep.len(), rows)
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
