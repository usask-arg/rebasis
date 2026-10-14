//! Python extension module `rebasis._rebasis`.
//!
//! Functions take contiguous float64 numpy arrays (the `rebasis` Python package converts and
//! broadcasts per-element parameters to full length) and return numpy arrays; sparse matrices
//! are returned as (indptr, indices, data, shape) for the Python side to wrap in scipy. A
//! `weight` argument is None (uniform) or the scale s of a cos(s·x) weight. Computations
//! release the GIL.

use numpy::{PyArray1, PyArray2, PyArrayMethods, PyUntypedArrayMethods};
use pyo3::create_exception;
use pyo3::exceptions::{PyBufferError, PyValueError};
use pyo3::prelude::*;
use rebasis::{Apodization, Interpolation, Profile, Weight};

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

fn read_rows(x: &Bound<'_, PyArray2<f64>>, name: &str) -> PyResult<Vec<Vec<f64>>> {
    let r = x
        .try_readonly()
        .map_err(|e| PyBufferError::new_err(format!("{name} cannot be read: {e}")))?;
    Ok(r.as_array()
        .rows()
        .into_iter()
        .map(|row| row.to_vec())
        .collect())
}

/// Reads a (k, n) float64 array as a core ndarray view and runs `f` on it with the GIL released.
fn with_matrix<'py, T: Send>(
    py: Python<'py>,
    a: &Bound<'py, PyArray2<f64>>,
    name: &str,
    expected_cols: usize,
    f: impl FnOnce(ndarray::ArrayView2<f64>) -> T + Send,
) -> PyResult<T> {
    let r = a
        .try_readonly()
        .map_err(|e| PyBufferError::new_err(format!("{name} cannot be read: {e}")))?;
    let shape = (r.shape()[0], r.shape()[1]);
    if shape.1 != expected_cols {
        return Err(PyValueError::new_err(format!(
            "{name} has {} values along the transformed axis, expected {expected_cols}",
            shape.1
        )));
    }
    let view = ndarray::ArrayView2::from_shape(shape, r.as_slice()?)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(py.detach(|| f(view)))
}

fn weight(scale: Option<f64>) -> PyResult<Weight> {
    match scale {
        None => Ok(Weight::Uniform),
        Some(s) if s.is_finite() && s != 0.0 => Ok(Weight::Cosine { scale: s }),
        Some(s) => Err(PyValueError::new_err(format!(
            "bad cosine weight scale {s}"
        ))),
    }
}

fn interpolation(name: &str) -> PyResult<Interpolation> {
    match name {
        "linear" => Ok(Interpolation::Linear),
        "pchip" => Ok(Interpolation::Pchip),
        other => Err(PyValueError::new_err(format!(
            "interpolation must be 'linear' or 'pchip', got {other:?}"
        ))),
    }
}

fn normalization(name: &str) -> PyResult<rebasis::Normalization> {
    match name {
        "integral" => Ok(rebasis::Normalization::Integral),
        "rowsum" => Ok(rebasis::Normalization::RowSum),
        "none" => Ok(rebasis::Normalization::None),
        other => Err(PyValueError::new_err(format!(
            "normalization must be 'integral', 'rowsum' or 'none', got {other:?}"
        ))),
    }
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

fn space(r: rebasis::Result<rebasis::Space>) -> PyResult<PySpace> {
    Ok(PySpace {
        inner: r.map_err(to_pyerr)?,
    })
}

/// Profiles at `centers` from per-element parameters (each already of length len(centers)).
fn profiles(
    centers: Vec<f64>,
    cutoff: f64,
    make: impl Fn(usize) -> rebasis::Result<Profile>,
) -> PyResult<PySpace> {
    let p = (0..centers.len())
        .map(make)
        .collect::<rebasis::Result<Vec<_>>>()
        .map_err(to_pyerr)?;
    if p.is_empty() {
        return Ok(PySpace {
            inner: rebasis::Space::new(vec![]),
        });
    }
    space(rebasis::Space::profiles(&centers, &p, cutoff))
}

#[pymethods]
impl PySpace {
    #[staticmethod]
    fn deltas(x: Bound<'_, PyArray1<f64>>) -> PyResult<Self> {
        space(rebasis::Space::deltas(&read_vec(&x, "x")?))
    }

    #[staticmethod]
    fn boxes(edges: Bound<'_, PyArray1<f64>>) -> PyResult<Self> {
        space(rebasis::Space::boxes(&read_vec(&edges, "edges")?))
    }

    #[staticmethod]
    fn hats(nodes: Bound<'_, PyArray1<f64>>) -> PyResult<Self> {
        space(rebasis::Space::hats(&read_vec(&nodes, "nodes")?))
    }

    #[staticmethod]
    fn bsplines(knots: Bound<'_, PyArray1<f64>>, degree: usize) -> PyResult<Self> {
        space(rebasis::Space::bsplines(
            &read_vec(&knots, "knots")?,
            degree,
        ))
    }

    #[staticmethod]
    fn harmonics(n: usize, period: f64, start: f64) -> PyResult<Self> {
        space(rebasis::Space::harmonics(n, period, start))
    }

    #[staticmethod]
    fn bsplines_periodic(breakpoints: Bound<'_, PyArray1<f64>>, degree: usize) -> PyResult<Self> {
        space(rebasis::Space::bsplines_periodic(
            &read_vec(&breakpoints, "breakpoints")?,
            degree,
        ))
    }

    #[staticmethod]
    fn bsplines_clamped(breakpoints: Bound<'_, PyArray1<f64>>, degree: usize) -> PyResult<Self> {
        space(rebasis::Space::bsplines_clamped(
            &read_vec(&breakpoints, "breakpoints")?,
            degree,
        ))
    }

    #[staticmethod]
    fn gaussians(
        centers: Bound<'_, PyArray1<f64>>,
        sigma: Bound<'_, PyArray1<f64>>,
        cutoff: f64,
    ) -> PyResult<Self> {
        let s = read_vec(&sigma, "sigma")?;
        profiles(read_vec(&centers, "centers")?, cutoff, |i| {
            Profile::gaussian(s[i])
        })
    }

    #[staticmethod]
    fn lorentzians(
        centers: Bound<'_, PyArray1<f64>>,
        hwhm: Bound<'_, PyArray1<f64>>,
        cutoff: f64,
    ) -> PyResult<Self> {
        let h = read_vec(&hwhm, "hwhm")?;
        profiles(read_vec(&centers, "centers")?, cutoff, |i| {
            Profile::lorentzian(h[i])
        })
    }

    #[staticmethod]
    fn voigts(
        centers: Bound<'_, PyArray1<f64>>,
        sigma: Bound<'_, PyArray1<f64>>,
        gamma: Bound<'_, PyArray1<f64>>,
        cutoff: f64,
    ) -> PyResult<Self> {
        let (s, g) = (read_vec(&sigma, "sigma")?, read_vec(&gamma, "gamma")?);
        profiles(read_vec(&centers, "centers")?, cutoff, |i| {
            Profile::voigt(s[i], g[i])
        })
    }

    #[staticmethod]
    fn pseudo_voigts(
        centers: Bound<'_, PyArray1<f64>>,
        hwhm: Bound<'_, PyArray1<f64>>,
        eta: Bound<'_, PyArray1<f64>>,
        cutoff: f64,
    ) -> PyResult<Self> {
        let (h, e) = (read_vec(&hwhm, "hwhm")?, read_vec(&eta, "eta")?);
        profiles(read_vec(&centers, "centers")?, cutoff, |i| {
            Profile::pseudo_voigt(h[i], e[i])
        })
    }

    #[staticmethod]
    fn super_gaussians(
        centers: Bound<'_, PyArray1<f64>>,
        hwhm_left: Bound<'_, PyArray1<f64>>,
        hwhm_right: Bound<'_, PyArray1<f64>>,
        power: Bound<'_, PyArray1<f64>>,
        cutoff: f64,
    ) -> PyResult<Self> {
        let l = read_vec(&hwhm_left, "hwhm_left")?;
        let r = read_vec(&hwhm_right, "hwhm_right")?;
        let p = read_vec(&power, "power")?;
        profiles(read_vec(&centers, "centers")?, cutoff, |i| {
            Profile::super_gaussian(l[i], r[i], p[i])
        })
    }

    #[staticmethod]
    fn smoothed_boxes(
        centers: Bound<'_, PyArray1<f64>>,
        half_width: Bound<'_, PyArray1<f64>>,
        sigma: Bound<'_, PyArray1<f64>>,
        cutoff: f64,
    ) -> PyResult<Self> {
        let w = read_vec(&half_width, "half_width")?;
        let s = read_vec(&sigma, "sigma")?;
        profiles(read_vec(&centers, "centers")?, cutoff, |i| {
            Profile::smoothed_box(w[i], s[i])
        })
    }

    /// `apodization` is a name, or None with `cosine_sum` (<= 4) or `polynomial` (<= 5)
    /// coefficients.
    #[staticmethod]
    #[pyo3(signature = (centers, max_opd, cutoff, apodization=None, cosine_sum=None, polynomial=None))]
    fn fts(
        centers: Bound<'_, PyArray1<f64>>,
        max_opd: Bound<'_, PyArray1<f64>>,
        cutoff: f64,
        apodization: Option<&str>,
        cosine_sum: Option<Vec<f64>>,
        polynomial: Option<Vec<f64>>,
    ) -> PyResult<Self> {
        let apod = match (apodization, cosine_sum, polynomial) {
            (Some(name), None, None) => Apodization::by_name(name).map_err(to_pyerr)?,
            (None, Some(c), None) if !c.is_empty() && c.len() <= 4 => {
                let mut a = [0.0; 4];
                a[..c.len()].copy_from_slice(&c);
                Apodization::CosineSum(a)
            }
            (None, None, Some(c)) if !c.is_empty() && c.len() <= 5 => {
                let mut a = [0.0; 5];
                a[..c.len()].copy_from_slice(&c);
                Apodization::Polynomial(a)
            }
            _ => {
                return Err(PyValueError::new_err(
                    "give one of apodization (a name), cosine_sum (1-4 coefficients) or \
                     polynomial (1-5 coefficients)",
                ));
            }
        };
        let l = read_vec(&max_opd, "max_opd")?;
        profiles(read_vec(&centers, "centers")?, cutoff, |i| {
            Profile::fts(l[i], apod)
        })
    }

    #[staticmethod]
    fn rectangles(
        centers: Bound<'_, PyArray1<f64>>,
        widths: Bound<'_, PyArray1<f64>>,
    ) -> PyResult<Self> {
        space(rebasis::Space::rectangles(
            &read_vec(&centers, "centers")?,
            &read_vec(&widths, "widths")?,
        ))
    }

    #[staticmethod]
    fn triangles(
        centers: Bound<'_, PyArray1<f64>>,
        hwhm: Bound<'_, PyArray1<f64>>,
    ) -> PyResult<Self> {
        space(rebasis::Space::triangles(
            &read_vec(&centers, "centers")?,
            &read_vec(&hwhm, "hwhm")?,
        ))
    }

    #[staticmethod]
    fn trapezoids(
        centers: Bound<'_, PyArray1<f64>>,
        top: Bound<'_, PyArray1<f64>>,
        bottom: Bound<'_, PyArray1<f64>>,
    ) -> PyResult<Self> {
        space(rebasis::Space::trapezoids(
            &read_vec(&centers, "centers")?,
            &read_vec(&top, "top")?,
            &read_vec(&bottom, "bottom")?,
        ))
    }

    /// Copies of the tabulated shape (shape_x, shape_y), shifted to `centers` and stretched by
    /// `stretch`.
    #[staticmethod]
    fn shifted(
        shape_x: Bound<'_, PyArray1<f64>>,
        shape_y: Bound<'_, PyArray1<f64>>,
        centers: Bound<'_, PyArray1<f64>>,
        stretch: Bound<'_, PyArray1<f64>>,
        interp: &str,
    ) -> PyResult<Self> {
        let sx = read_vec(&shape_x, "shape_x")?;
        let sy = read_vec(&shape_y, "shape_y")?;
        let shape = match interpolation(interp)? {
            Interpolation::Linear => rebasis::Shape::piecewise_linear(&sx, &sy),
            Interpolation::Pchip => rebasis::Shape::pchip(&sx, &sy),
        }
        .map_err(to_pyerr)?;
        space(rebasis::Space::shifted(
            shape,
            &read_vec(&centers, "centers")?,
            &read_vec(&stretch, "stretch")?,
        ))
    }

    /// Per-element tables: x has 1 or len(centers) rows, y has len(centers) rows.
    #[staticmethod]
    fn tabulated(
        centers: Bound<'_, PyArray1<f64>>,
        x: Bound<'_, PyArray2<f64>>,
        y: Bound<'_, PyArray2<f64>>,
        interp: &str,
    ) -> PyResult<Self> {
        space(rebasis::Space::tabulated(
            &read_vec(&centers, "centers")?,
            &read_rows(&x, "x")?,
            &read_rows(&y, "y")?,
            interpolation(interp)?,
        ))
    }

    /// The elements as functions of x = forward(u): kind "reciprocal" (x = a / u) or "affine"
    /// (x = a + b·u).
    fn mapped(&self, kind: &str, a: f64, b: f64, jacobian: bool) -> PyResult<Self> {
        let map = match kind {
            "reciprocal" => rebasis::CoordinateMap::Reciprocal { k: a },
            "affine" => rebasis::CoordinateMap::Affine {
                offset: a,
                scale: b,
            },
            other => {
                return Err(PyValueError::new_err(format!(
                    "map kind must be 'reciprocal' or 'affine', got {other:?}"
                )));
            }
        };
        space(self.inner.mapped(map, jacobian))
    }

    fn take(&self, indices: Vec<usize>) -> PyResult<Self> {
        space(self.inner.select(&indices))
    }

    fn derivative(&self, order: usize) -> PyResult<Self> {
        space(self.inner.derivative(order))
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

    #[pyo3(signature = (weight=None))]
    fn integrals<'py>(
        &self,
        py: Python<'py>,
        weight: Option<f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let w = self::weight(weight)?;
        let v = py.detach(|| self.inner.integrals(w));
        Ok(PyArray1::from_vec(py, v))
    }
}

#[pyfunction]
#[pyo3(signature = (a, b, weight=None))]
fn inner_products<'py>(
    py: Python<'py>,
    a: Bound<'py, PySpace>,
    b: Bound<'py, PySpace>,
    weight: Option<f64>,
) -> PyResult<CsrParts<'py>> {
    let w = self::weight(weight)?;
    let (a, b) = (&a.get().inner, &b.get().inner);
    let m = py.detach(|| rebasis::inner_products(a, b, w));
    Ok(csr_parts(py, &m))
}

#[pyfunction]
#[pyo3(signature = (kernels, basis, normalization, weight=None))]
fn measure<'py>(
    py: Python<'py>,
    kernels: Bound<'py, PySpace>,
    basis: Bound<'py, PySpace>,
    normalization: &str,
    weight: Option<f64>,
) -> PyResult<CsrParts<'py>> {
    let norm = self::normalization(normalization)?;
    let w = self::weight(weight)?;
    let (k, b) = (&kernels.get().inner, &basis.get().inner);
    let m = py.detach(|| rebasis::measure(k, b, norm, w));
    Ok(csr_parts(py, &m))
}

/// (W, ∂W/∂p) for the kernels' parameter p; see `rebasis.measure_derivative`.
#[pyfunction]
#[pyo3(signature = (kernels, basis, normalization, parameter, weight=None))]
fn measure_derivative<'py>(
    py: Python<'py>,
    kernels: Bound<'py, PySpace>,
    basis: Bound<'py, PySpace>,
    normalization: &str,
    parameter: &str,
    weight: Option<f64>,
) -> PyResult<(CsrParts<'py>, CsrParts<'py>)> {
    let norm = self::normalization(normalization)?;
    let w = self::weight(weight)?;
    let p = rebasis::Parameter::by_name(parameter).map_err(to_pyerr)?;
    let (k, b) = (&kernels.get().inner, &basis.get().inner);
    let (wm, dw) = py
        .detach(|| rebasis::measure_derivative(k, b, norm, w, p))
        .map_err(to_pyerr)?;
    Ok((csr_parts(py, &wm), csr_parts(py, &dw)))
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
    #[pyo3(signature = (src, dst, weight=None))]
    fn new(
        py: Python<'_>,
        src: Bound<'_, PySpace>,
        dst: Bound<'_, PySpace>,
        weight: Option<f64>,
    ) -> PyResult<Self> {
        let w = self::weight(weight)?;
        let (s, d) = (&src.get().inner, &dst.get().inner);
        let inner = py
            .detach(|| rebasis::Projection::new(s, d, w))
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

    /// Projects each row of f, shape (k, n_src), returning (k, n_dst). `bounds` is None,
    /// "local", or (lo, hi).
    #[pyo3(signature = (f, bounds=None))]
    fn apply_last_axis<'py>(
        &self,
        py: Python<'py>,
        f: Bound<'py, PyArray2<f64>>,
        bounds: Option<Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let bounds = match bounds {
            None => None,
            Some(b) => {
                if let Ok(s) = b.extract::<String>() {
                    if s != "local" {
                        return Err(PyValueError::new_err(format!(
                            "bounds must be 'local' or (lo, hi), got {s:?}"
                        )));
                    }
                    Some(rebasis::Bounds::Local)
                } else {
                    let (lo, hi): (f64, f64) = b.extract()?;
                    if lo.is_nan() || hi.is_nan() || lo > hi {
                        return Err(PyValueError::new_err("bounds need lo <= hi"));
                    }
                    Some(rebasis::Bounds::Global { lo, hi })
                }
            }
        };
        let g = with_matrix(py, &f, "f", self.inner.n_src(), |v| match bounds {
            Some(b) => self.inner.apply_bounded_last_axis(v, b),
            None => self.inner.apply_last_axis(v),
        })?;
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

#[pyclass(frozen, name = "Fit", module = "rebasis._rebasis")]
pub struct PyFit {
    inner: rebasis::Fit,
}

#[pymethods]
impl PyFit {
    #[new]
    #[pyo3(signature = (kernels, model, normalization, smoothing, penalty, order, weight=None))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        py: Python<'_>,
        kernels: Bound<'_, PySpace>,
        model: Bound<'_, PySpace>,
        normalization: &str,
        smoothing: f64,
        penalty: &str,
        order: usize,
        weight: Option<f64>,
    ) -> PyResult<Self> {
        let norm = self::normalization(normalization)?;
        let w = self::weight(weight)?;
        let penalty = match penalty {
            "derivative" => rebasis::Penalty::Derivative(order),
            "difference" => rebasis::Penalty::Difference(order),
            other => {
                return Err(PyValueError::new_err(format!(
                    "penalty must be 'derivative' or 'difference', got {other:?}"
                )));
            }
        };
        let (k, m) = (&kernels.get().inner, &model.get().inner);
        let inner = py
            .detach(|| rebasis::Fit::new(k, m, norm, w, smoothing, penalty))
            .map_err(to_pyerr)?;
        Ok(PyFit { inner })
    }

    #[getter]
    fn n_data(&self) -> usize {
        self.inner.n_data()
    }

    #[getter]
    fn n_model(&self) -> usize {
        self.inner.n_model()
    }

    /// Fits each row of data, shape (k, n_data), returning coefficients (k, n_model).
    fn apply_last_axis<'py>(
        &self,
        py: Python<'py>,
        data: Bound<'py, PyArray2<f64>>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let c = with_matrix(py, &data, "data", self.inner.n_data(), |v| {
            self.inner.apply_last_axis(v)
        })?;
        array2(py, c)
    }

    fn to_dense<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let d = py.detach(|| self.inner.to_dense());
        array2(py, d)
    }

    fn design<'py>(&self, py: Python<'py>) -> CsrParts<'py> {
        csr_parts(py, self.inner.design())
    }
}

/// One dimension of a separable covariance: (family, length, period or None, variance profile
/// as (basis, coefficients) or None).
type DimSpec<'py> = (
    String,
    f64,
    Option<f64>,
    Option<(Bound<'py, PySpace>, Vec<f64>)>,
);

fn product_covariance(
    variance: f64,
    dims: Vec<DimSpec<'_>>,
) -> PyResult<rebasis::ProductCovariance> {
    if !(variance.is_finite() && variance > 0.0) {
        return Err(PyValueError::new_err(format!(
            "variance must be finite and > 0, got {variance}"
        )));
    }
    let dims =
        dims.into_iter()
            .map(|(family, length, period, profile)| {
                let f = rebasis::Family::by_name(&family)?;
                let c = rebasis::Covariance1d::new(f, length, period)?;
                Ok(match profile {
                    None => c,
                    Some((basis, coeffs)) => c.with_variance_profile(
                        rebasis::VarianceProfile::new(&basis.get().inner, &coeffs)?,
                    ),
                })
            })
            .collect::<rebasis::Result<Vec<_>>>()
            .map_err(to_pyerr)?;
    Ok(rebasis::ProductCovariance { variance, dims })
}

fn weights(w: Vec<Option<f64>>) -> PyResult<Vec<Weight>> {
    w.into_iter().map(weight).collect()
}

/// The covariance matrix between product atoms a (rows) and b (columns); see
/// `rebasis.covariance`.
#[pyfunction]
#[pyo3(signature = (variance, dims, a, wa, b, wb, symmetric=false))]
#[allow(clippy::too_many_arguments)]
fn covariance_matrix<'py>(
    py: Python<'py>,
    variance: f64,
    dims: Vec<DimSpec<'py>>,
    a: Vec<Bound<'py, PySpace>>,
    wa: Vec<Option<f64>>,
    b: Vec<Bound<'py, PySpace>>,
    wb: Vec<Option<f64>>,
    symmetric: bool,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let cov = product_covariance(variance, dims)?;
    let (wa, wb) = (weights(wa)?, weights(wb)?);
    let a: Vec<&rebasis::Space> = a.iter().map(|s| &s.get().inner).collect();
    let b: Vec<&rebasis::Space> = b.iter().map(|s| &s.get().inner).collect();
    let m = py
        .detach(|| {
            cov.matrix(
                rebasis::Atoms {
                    spaces: &a,
                    weights: &wa,
                },
                rebasis::Atoms {
                    spaces: &b,
                    weights: &wb,
                },
                symmetric,
            )
        })
        .map_err(to_pyerr)?;
    array2(py, m)
}

type MatrixAndDerivatives<'py> = (
    Bound<'py, PyArray2<f64>>,
    Vec<Bound<'py, PyArray2<f64>>>,
    Vec<Vec<Bound<'py, PyArray2<f64>>>>,
);

/// The symmetric covariance K of atoms a and ℓ_d ∂K/∂ℓ_d for each dimension d.
#[pyfunction]
fn covariance_length_derivatives<'py>(
    py: Python<'py>,
    variance: f64,
    dims: Vec<DimSpec<'py>>,
    a: Vec<Bound<'py, PySpace>>,
    wa: Vec<Option<f64>>,
) -> PyResult<MatrixAndDerivatives<'py>> {
    let cov = product_covariance(variance, dims)?;
    let wa = weights(wa)?;
    let a: Vec<&rebasis::Space> = a.iter().map(|s| &s.get().inner).collect();
    let (k, g, gp) = py
        .detach(|| {
            cov.matrix_with_derivatives(rebasis::Atoms {
                spaces: &a,
                weights: &wa,
            })
        })
        .map_err(to_pyerr)?;
    let gp = gp
        .into_iter()
        .map(|ms| {
            ms.into_iter()
                .map(|m| array2(py, m))
                .collect::<PyResult<Vec<_>>>()
        })
        .collect::<PyResult<Vec<_>>>()?;
    let g = g
        .into_iter()
        .map(|m| array2(py, m))
        .collect::<PyResult<Vec<_>>>()?;
    Ok((array2(py, k)?, g, gp))
}

/// The variances of product atoms a.
#[pyfunction]
fn covariance_diagonal<'py>(
    py: Python<'py>,
    variance: f64,
    dims: Vec<DimSpec<'py>>,
    a: Vec<Bound<'py, PySpace>>,
    wa: Vec<Option<f64>>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let cov = product_covariance(variance, dims)?;
    let wa = weights(wa)?;
    let a: Vec<&rebasis::Space> = a.iter().map(|s| &s.get().inner).collect();
    let d = py
        .detach(|| {
            cov.diagonal(rebasis::Atoms {
                spaces: &a,
                weights: &wa,
            })
        })
        .map_err(to_pyerr)?;
    Ok(PyArray1::from_vec(py, d))
}

/// Re w(x + iy) for y >= 0, w the Faddeeva function.
#[pyfunction]
fn faddeeva_re(x: f64, y: f64) -> f64 {
    rebasis::special::faddeeva_re(x, y)
}

#[pymodule]
fn _rebasis(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add("SingularError", m.py().get_type::<SingularError>())?;
    m.add("FWHM_PER_SIGMA", rebasis::FWHM_PER_SIGMA)?;
    m.add_class::<PySpace>()?;
    m.add_class::<PyProjection>()?;
    m.add_class::<PyFit>()?;
    m.add_function(wrap_pyfunction!(inner_products, m)?)?;
    m.add_function(wrap_pyfunction!(measure, m)?)?;
    m.add_function(wrap_pyfunction!(measure_derivative, m)?)?;
    m.add_function(wrap_pyfunction!(evaluate, m)?)?;
    m.add_function(wrap_pyfunction!(faddeeva_re, m)?)?;
    m.add_function(wrap_pyfunction!(covariance_matrix, m)?)?;
    m.add_function(wrap_pyfunction!(covariance_diagonal, m)?)?;
    m.add_function(wrap_pyfunction!(covariance_length_derivatives, m)?)?;
    Ok(())
}
