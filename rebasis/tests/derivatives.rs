//! Analytic derivatives of measurement matrices against central finite differences.

use ndarray::Array2;
use rebasis::{
    Apodization, Element, Normalization, Parameter, Profile, Shape, Space, Weight, measure,
    measure_derivative,
};

fn linspace(a: f64, b: f64, n: usize) -> Vec<f64> {
    (0..n)
        .map(|i| a + (b - a) * i as f64 / (n - 1) as f64)
        .collect()
}

type Build = Box<dyn Fn(f64) -> Space>;

/// Kernels at two centers, built from parameter value θ. The centers are off the basis grids so
/// that no kink of a basis function sits exactly at a kernel edge, where the finite difference
/// (not the derivative) loses accuracy.
fn profiles(f: impl Fn(f64) -> Profile + 'static, cutoff: f64) -> Build {
    Box::new(move |t| Space::profiles(&[0.3712345, 1.1037111], &[f(t)], cutoff).unwrap())
}

fn cases() -> Vec<(&'static str, Parameter, f64, Build)> {
    use Parameter::*;
    let e = f64::exp;
    vec![
        (
            "gaussian",
            Center,
            0.0,
            Box::new(|t| {
                Space::profiles(
                    &[0.3712345 + t, 1.1037111 + t],
                    &[Profile::gaussian(0.05).unwrap()],
                    6.0,
                )
                .unwrap()
            }),
        ),
        (
            "gaussian",
            Scale,
            0.0,
            profiles(move |t| Profile::gaussian(0.05 * e(t)).unwrap(), 6.0),
        ),
        (
            "gaussian",
            Sigma,
            0.05,
            profiles(|t| Profile::gaussian(t).unwrap(), 6.0),
        ),
        (
            "lorentzian",
            Scale,
            0.0,
            profiles(move |t| Profile::lorentzian(0.02 * e(t)).unwrap(), 10.0),
        ),
        (
            "lorentzian",
            Gamma,
            0.02,
            profiles(|t| Profile::lorentzian(t).unwrap(), 10.0),
        ),
        (
            "voigt",
            Center,
            0.0,
            Box::new(|t| {
                Space::profiles(
                    &[0.3712345 + t, 1.1037111 + t],
                    &[Profile::voigt(0.03, 0.01).unwrap()],
                    10.0,
                )
                .unwrap()
            }),
        ),
        (
            "voigt",
            Scale,
            0.0,
            profiles(
                move |t| Profile::voigt(0.03 * e(t), 0.01 * e(t)).unwrap(),
                10.0,
            ),
        ),
        (
            "voigt",
            Sigma,
            0.03,
            profiles(|t| Profile::voigt(t, 0.01).unwrap(), 10.0),
        ),
        (
            "voigt",
            Gamma,
            0.01,
            profiles(|t| Profile::voigt(0.03, t).unwrap(), 10.0),
        ),
        (
            "pseudo_voigt",
            Eta,
            0.3,
            profiles(|t| Profile::pseudo_voigt(0.03, t).unwrap(), 10.0),
        ),
        (
            "pseudo_voigt",
            Scale,
            0.0,
            profiles(
                move |t| Profile::pseudo_voigt(0.03 * e(t), 0.3).unwrap(),
                10.0,
            ),
        ),
        (
            "super_gaussian",
            Center,
            0.0,
            Box::new(|t| {
                Space::profiles(
                    &[0.3712345 + t, 1.1037111 + t],
                    &[Profile::super_gaussian(0.04, 0.06, 2.6).unwrap()],
                    5.0,
                )
                .unwrap()
            }),
        ),
        (
            "super_gaussian",
            Scale,
            0.0,
            profiles(
                move |t| Profile::super_gaussian(0.04 * e(t), 0.06 * e(t), 2.6).unwrap(),
                5.0,
            ),
        ),
        (
            "super_gaussian",
            Power,
            2.6,
            profiles(|t| Profile::super_gaussian(0.04, 0.06, t).unwrap(), 5.0),
        ),
        (
            "super_gaussian",
            Asymmetry,
            0.2,
            profiles(
                |a| Profile::super_gaussian(0.05 * (1.0 - a), 0.05 * (1.0 + a), 4.0).unwrap(),
                5.0,
            ),
        ),
        (
            "smoothed_box",
            Scale,
            0.0,
            profiles(
                move |t| Profile::smoothed_box(0.05 * e(t), 0.01 * e(t)).unwrap(),
                6.0,
            ),
        ),
        (
            "smoothed_box",
            HalfWidth,
            0.05,
            profiles(|t| Profile::smoothed_box(t, 0.01).unwrap(), 6.0),
        ),
        (
            "smoothed_box",
            Sigma,
            0.01,
            profiles(|t| Profile::smoothed_box(0.05, t).unwrap(), 6.0),
        ),
        (
            "fts",
            Center,
            0.0,
            Box::new(|t| {
                Space::profiles(
                    &[0.3712345 + t, 1.1037111 + t],
                    &[Profile::fts(20.0, Apodization::NORTON_BEER_MEDIUM).unwrap()],
                    12.0,
                )
                .unwrap()
            }),
        ),
        (
            "fts",
            Scale,
            0.0,
            profiles(
                move |t| Profile::fts(20.0 * e(-t), Apodization::HAMMING).unwrap(),
                12.0,
            ),
        ),
        (
            "rectangle",
            Center,
            0.0,
            Box::new(|t| Space::rectangles(&[0.3712345 + t, 1.1037111 + t], &[0.1]).unwrap()),
        ),
        (
            "rectangle",
            Scale,
            0.0,
            Box::new(move |t| Space::rectangles(&[0.3712345, 1.1037111], &[0.1 * e(t)]).unwrap()),
        ),
        (
            "triangle",
            Scale,
            0.0,
            Box::new(move |t| Space::triangles(&[0.3712345, 1.1037111], &[0.04 * e(t)]).unwrap()),
        ),
        (
            "trapezoid",
            Center,
            0.0,
            Box::new(|t| {
                Space::trapezoids(&[0.3712345 + t, 1.1037111 + t], &[0.02], &[0.07]).unwrap()
            }),
        ),
        (
            "tabulated",
            Scale,
            0.0,
            Box::new(move |t| {
                let shape =
                    Shape::pchip(&[-0.05, -0.02, 0.0, 0.03, 0.06], &[0.0, 0.5, 1.0, 0.4, 0.0])
                        .unwrap();
                Space::shifted(shape, &[0.3712345, 1.1037111], &[e(t)]).unwrap()
            }),
        ),
        (
            "tabulated_jump",
            Center,
            0.0,
            Box::new(|t| {
                // a shape that jumps at both ends and in the middle
                let shape = Shape::new(vec![
                    rebasis::Segment {
                        a: -0.05,
                        b: 0.0,
                        coeffs: [0.5, 2.0, 0.0, 0.0],
                    },
                    rebasis::Segment {
                        a: 0.0,
                        b: 0.04,
                        coeffs: [1.0, -5.0, 0.0, 0.0],
                    },
                ])
                .unwrap();
                Space::shifted(shape, &[0.3712345 + t, 1.1037111 + t], &[1.0]).unwrap()
            }),
        ),
        (
            "delta",
            Center,
            0.0,
            Box::new(|t| Space::deltas(&[0.371234512 + t, 1.1037111013 + t]).unwrap()),
        ),
    ]
}

fn finite_difference(
    build: &Build,
    t0: f64,
    basis: &Space,
    norm: Normalization,
    weight: Weight,
) -> Array2<f64> {
    let h = 1e-6 * t0.abs().max(1.0) * if t0 == 0.0 { 1.0 } else { t0.abs().min(1.0) };
    let h = h.max(1e-7);
    let p = measure(&build(t0 + h), basis, norm, weight).to_dense();
    let m = measure(&build(t0 - h), basis, norm, weight).to_dense();
    (p - m) / (2.0 * h)
}

fn check(norm: Normalization, weight: Weight, basis: &Space, label: &str) {
    for (name, param, t0, build) in cases() {
        let (w, dw) = measure_derivative(&build(t0), basis, norm, weight, param).unwrap();
        let w_ref = measure(&build(t0), basis, norm, weight).to_dense();
        let fd = finite_difference(&build, t0, basis, norm, weight);
        let dw = dw.to_dense();
        let scale = fd.iter().fold(0.0f64, |a, v| a.max(v.abs()));
        let err = (&dw - &fd).iter().fold(0.0f64, |a, v| a.max(v.abs()));
        // The finite difference is the limiting factor: with h = 1e-6 most cases agree to
        // < 2e-7 (larger h straddles kinks of piecewise-polynomial W(θ); smaller h is
        // dominated by rounding).
        assert!(
            err <= 1e-6 * scale + 1e-9,
            "{label} {name} {param:?}: max error {err:e}, scale {scale:e}"
        );
        assert!((&w.to_dense() - &w_ref).iter().all(|v| v.abs() < 1e-15));
    }
}

#[test]
fn derivatives_on_hats_match_finite_differences() {
    let hats = Space::hats(&linspace(0.0, 1.5, 3001)).unwrap();
    check(
        Normalization::Integral,
        Weight::Uniform,
        &hats,
        "hats/integral",
    );
    check(Normalization::RowSum, Weight::Uniform, &hats, "hats/rowsum");
    check(Normalization::None, Weight::Uniform, &hats, "hats/none");
}

#[test]
fn derivatives_with_a_cosine_weight() {
    let hats = Space::hats(&linspace(0.0, 1.5, 3001)).unwrap();
    check(
        Normalization::Integral,
        Weight::Cosine { scale: 1.3 },
        &hats,
        "hats/cosine",
    );
}

#[test]
fn derivatives_on_cubic_splines_and_boxes() {
    let splines = Space::bsplines_clamped(&linspace(0.0, 1.5, 151), 3).unwrap();
    check(
        Normalization::Integral,
        Weight::Uniform,
        &splines,
        "bsplines",
    );
    let boxes = Space::boxes(&linspace(0.0013, 1.5013, 301)).unwrap();
    check(Normalization::Integral, Weight::Uniform, &boxes, "boxes");
}

#[test]
fn shift_of_a_symmetric_kernel_on_a_linear_spectrum_is_the_slope() {
    // g(c) = f(c) for symmetric unit-area kernels and linear f, so ∂g/∂c = f' and ∂g/∂s = 0
    let x = linspace(-2.0, 4.0, 60001);
    let hats = Space::hats(&x).unwrap();
    let f: ndarray::Array1<f64> = x.iter().map(|v| 3.0 * v - 1.0).collect();
    let kernels = Space::profiles(&[1.0], &[Profile::voigt(0.05, 0.02).unwrap()], 30.0).unwrap();
    for (p, expected) in [(Parameter::Center, 3.0), (Parameter::Scale, 0.0)] {
        let (_, dw) =
            measure_derivative(&kernels, &hats, Normalization::Integral, Weight::Uniform, p)
                .unwrap();
        let g = dw.apply(f.view());
        assert!((g[0] - expected).abs() < 1e-10, "{p:?}: {}", g[0]);
    }
}

#[test]
fn inapplicable_parameters_are_errors() {
    let hats = Space::hats(&linspace(0.0, 1.0, 11)).unwrap();
    let g = Space::profiles(&[0.5], &[Profile::gaussian(0.1).unwrap()], 3.0).unwrap();
    assert!(
        measure_derivative(
            &g,
            &hats,
            Normalization::Integral,
            Weight::Uniform,
            Parameter::Eta
        )
        .is_err()
    );
    let b = Space::new(vec![Element::boxcar(0.2, 0.4).unwrap()]);
    assert!(
        measure_derivative(
            &b,
            &hats,
            Normalization::Integral,
            Weight::Uniform,
            Parameter::Sigma
        )
        .is_err()
    );
}
