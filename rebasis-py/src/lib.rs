//! Python extension module `rebasis._rebasis`.
//!
//! Functions take contiguous float64 numpy arrays (the `rebasis` Python package converts) and
//! return numpy arrays; sparse matrices are returned as (indptr, indices, data, shape) for the
//! Python side to wrap in scipy. Computations release the GIL.

use numpy::{PyArray1, PyArray2, PyArrayMethods, PyUntypedArrayMethods};
use pyo3::create_exception;
use pyo3::exceptions::{PyBufferError, PyValueError};
use pyo3::prelude::*;

create_exception!(_rebasis, SingularError, PyValueError);

fn to_pyerr(e: rebasis::Error) -> PyErr {
    match e {
        rebasis::Error::InvalidInput(m) => PyValueError::new_err(m),
        rebasis::Error::Singular(m) => SingularError::new_err(m),
    }
}

fn read_vec(x: &Bound<'_, PyArray1<f64>>, name: &str) -> PyResult<Vec<f64>> {
    let r = x
        .try_readonly()
        .map_err(|e| PyBufferError::new_err(format!("{name} cannot be read: {e}")))?;
    Ok(r.as_array().to_vec())
}

type CsrParts<'py> = (
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<f64>>,
    (usize, usize),
);

fn csr_parts<'py>(py: Python<'py>, m: &rebasis::Csr) -> CsrParts<'py> {
    let to_i64 = |v: &[usize]| v.iter().map(|&i| i as i64).collect::<Vec<_>>();
    (
        PyArray1::from_vec(py, to_i64(m.indptr())),
        PyArray1::from_vec(py, to_i64(m.indices())),
        PyArray1::from_slice(py, m.data()),
        (m.nrows(), m.ncols()),
    )
}

fn array2(py: Python<'_>, a: ndarray::Array2<f64>) -> PyResult<Bound<'_, PyArray2<f64>>> {
    let (r, c) = a.dim();
    let data = a.into_raw_vec_and_offset().0;
    PyArray1::from_vec(py, data).reshape([r, c])
}

#[pyclass(frozen, name = "Space", module = "rebasis._rebasis")]
pub struct PySpace {
    inner: rebasis::Space,
}

#[pymethods]
impl PySpace {
    #[staticmethod]
    fn deltas(x: Bound<'_, PyArray1<f64>>) -> PyResult<Self> {
        let x = read_vec(&x, "x")?;
        Ok(PySpace {
            inner: rebasis::Space::deltas(&x).map_err(to_pyerr)?,
        })
    }

    #[staticmethod]
    fn boxes(edges: Bound<'_, PyArray1<f64>>) -> PyResult<Self> {
        let e = read_vec(&edges, "edges")?;
        Ok(PySpace {
            inner: rebasis::Space::boxes(&e).map_err(to_pyerr)?,
        })
    }

    #[staticmethod]
    fn hats(nodes: Bound<'_, PyArray1<f64>>) -> PyResult<Self> {
        let n = read_vec(&nodes, "nodes")?;
        Ok(PySpace {
            inner: rebasis::Space::hats(&n).map_err(to_pyerr)?,
        })
    }

    #[staticmethod]
    fn gaussians(
        centers: Bound<'_, PyArray1<f64>>,
        sigma: Bound<'_, PyArray1<f64>>,
        cutoff: f64,
    ) -> PyResult<Self> {
        let c = read_vec(&centers, "centers")?;
        let s = read_vec(&sigma, "sigma")?;
        Ok(PySpace {
            inner: rebasis::Space::gaussians(&c, &s, cutoff).map_err(to_pyerr)?,
        })
    }

    /// Copies of the piecewise-linear shape (shape_x, shape_y), shifted to `centers` and
    /// stretched by `stretch`.
    #[staticmethod]
    fn shifted_linear(
        shape_x: Bound<'_, PyArray1<f64>>,
        shape_y: Bound<'_, PyArray1<f64>>,
        centers: Bound<'_, PyArray1<f64>>,
        stretch: Bound<'_, PyArray1<f64>>,
    ) -> PyResult<Self> {
        let sx = read_vec(&shape_x, "shape_x")?;
        let sy = read_vec(&shape_y, "shape_y")?;
        let c = read_vec(&centers, "centers")?;
        let s = read_vec(&stretch, "stretch")?;
        let shape = rebasis::Shape::piecewise_linear(&sx, &sy).map_err(to_pyerr)?;
        Ok(PySpace {
            inner: rebasis::Space::shifted(shape, &c, &s).map_err(to_pyerr)?,
        })
    }

    fn __len__(&self) -> usize {
        self.inner.len()
    }

    fn lower<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_slice(py, self.inner.lower())
    }

    fn upper<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_slice(py, self.inner.upper())
    }

    fn centers<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_vec(py, self.inner.centers())
    }

    fn integrals<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_vec(py, self.inner.integrals())
    }
}

#[pyfunction]
fn inner_products<'py>(
    py: Python<'py>,
    a: Bound<'py, PySpace>,
    b: Bound<'py, PySpace>,
) -> CsrParts<'py> {
    let (a, b) = (&a.get().inner, &b.get().inner);
    let m = py.detach(|| rebasis::inner_products(a, b));
    csr_parts(py, &m)
}

#[pyfunction]
fn measure<'py>(
    py: Python<'py>,
    kernels: Bound<'py, PySpace>,
    basis: Bound<'py, PySpace>,
    normalization: &str,
) -> PyResult<CsrParts<'py>> {
    let norm = match normalization {
        "integral" => rebasis::Normalization::Integral,
        "rowsum" => rebasis::Normalization::RowSum,
        "none" => rebasis::Normalization::None,
        other => {
            return Err(PyValueError::new_err(format!(
                "normalization must be 'integral', 'rowsum' or 'none', got {other:?}"
            )));
        }
    };
    let (k, b) = (&kernels.get().inner, &basis.get().inner);
    let m = py.detach(|| rebasis::measure(k, b, norm));
    Ok(csr_parts(py, &m))
}

#[pyfunction]
fn evaluate<'py>(
    py: Python<'py>,
    basis: Bound<'py, PySpace>,
    x: Bound<'py, PyArray1<f64>>,
) -> PyResult<CsrParts<'py>> {
    let x = read_vec(&x, "x")?;
    let b = &basis.get().inner;
    let m = py.detach(|| rebasis::evaluate(b, &x)).map_err(to_pyerr)?;
    Ok(csr_parts(py, &m))
}

#[pyclass(frozen, name = "Projection", module = "rebasis._rebasis")]
pub struct PyProjection {
    inner: rebasis::Projection,
}

#[pymethods]
impl PyProjection {
    #[new]
    fn new(py: Python<'_>, src: Bound<'_, PySpace>, dst: Bound<'_, PySpace>) -> PyResult<Self> {
        let (s, d) = (&src.get().inner, &dst.get().inner);
        let inner = py
            .detach(|| rebasis::Projection::new(s, d))
            .map_err(to_pyerr)?;
        Ok(PyProjection { inner })
    }

    #[getter]
    fn n_src(&self) -> usize {
        self.inner.n_src()
    }

    #[getter]
    fn n_dst(&self) -> usize {
        self.inner.n_dst()
    }

    /// Projects each row of f, shape (k, n_src), returning (k, n_dst).
    fn apply_last_axis<'py>(
        &self,
        py: Python<'py>,
        f: Bound<'py, PyArray2<f64>>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let r = f
            .try_readonly()
            .map_err(|e| PyBufferError::new_err(format!("f cannot be read: {e}")))?;
        let shape = (r.shape()[0], r.shape()[1]);
        if shape.1 != self.inner.n_src() {
            return Err(PyValueError::new_err(format!(
                "f has {} values along the projected axis, expected {}",
                shape.1,
                self.inner.n_src()
            )));
        }
        let view = ndarray::ArrayView2::from_shape(shape, r.as_slice()?)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let g = py.detach(|| self.inner.apply_last_axis(view));
        array2(py, g)
    }

    fn to_dense<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let d = py.detach(|| self.inner.to_dense());
        array2(py, d)
    }

    fn cross<'py>(&self, py: Python<'py>) -> CsrParts<'py> {
        csr_parts(py, self.inner.cross())
    }
}

#[pymodule]
fn _rebasis(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add("SingularError", m.py().get_type::<SingularError>())?;
    m.add_class::<PySpace>()?;
    m.add_class::<PyProjection>()?;
    m.add_function(wrap_pyfunction!(inner_products, m)?)?;
    m.add_function(wrap_pyfunction!(measure, m)?)?;
    m.add_function(wrap_pyfunction!(evaluate, m)?)?;
    Ok(())
}
