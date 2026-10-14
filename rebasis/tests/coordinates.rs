//! Elements used in another coordinate (wavelength ↔ wavenumber).
//!
//! Tolerances: a hat 0.0025 nm wide at 500 nm is evaluated from x − a with x rounded to
//! ε·500 nm, so its values carry relative errors ~ ε·500/0.0025 ≈ 4e-11 in any coordinate;
//! mapping adds one more rounding of x. Results are compared at that conditioning limit.

use ndarray::Array1;
use rebasis::{
    CoordinateMap, Normalization, Parameter, Profile, Space, Weight, inner_products, measure,
    measure_derivative,
};

const U: Weight = Weight::Uniform;
const NM: CoordinateMap = CoordinateMap::NM_CMINV;

fn linspace(a: f64, b: f64, n: usize) -> Vec<f64> {
    (0..n)
        .map(|i| a + (b - a) * i as f64 / (n - 1) as f64)
        .collect()
}

fn max_abs(a: &ndarray::Array2<f64>) -> f64 {
    a.iter().fold(0.0f64, |m, v| m.max(v.abs()))
}

/// Pairs (a, b) of spaces in wavelength (nm) near 500 nm.
fn pairs() -> Vec<(&'static str, Space, Space)> {
    let centers = [499.3712, 500.0217, 500.9031];
    let hats = Space::hats(&linspace(497.0, 503.0, 2401)).unwrap();
    let boxes = Space::boxes(&linspace(497.0, 503.0, 241)).unwrap();
    let splines = Space::bsplines_clamped(&linspace(497.0, 503.0, 61), 3).unwrap();
    vec![
        (
            "gaussian/hats",
            Space::profiles(&centers, &[Profile::gaussian(0.2).unwrap()], 6.0).unwrap(),
            hats.clone(),
        ),
        (
            "voigt/hats",
            Space::profiles(&centers, &[Profile::voigt(0.1, 0.05).unwrap()], 10.0).unwrap(),
            hats.clone(),
        ),
        (
            "lorentzian/boxes",
            Space::profiles(&centers, &[Profile::lorentzian(0.05).unwrap()], 20.0).unwrap(),
            boxes.clone(),
        ),
        (
            "super_gaussian/splines",
            Space::profiles(
                &centers,
                &[Profile::super_gaussian(0.2, 0.3, 3.0).unwrap()],
                5.0,
            )
            .unwrap(),
            splines.clone(),
        ),
        (
            "rectangles/hats",
            Space::rectangles(&centers, &[0.4]).unwrap(),
            hats,
        ),
        ("splines/boxes", splines, boxes),
    ]
}

#[test]
fn inner_products_are_invariant_under_a_change_of_coordinate() {
    // ∫ a(λ) b(λ) dλ = ∫ a(λ(ν)) |dλ/dν| b(λ(ν)) dν
    for (name, a, b) in pairs() {
        let direct = inner_products(&a, &b, U).to_dense();
        let mapped = inner_products(
            &a.mapped(NM, true).unwrap(),
            &b.mapped(NM, false).unwrap(),
            U,
        )
        .to_dense();
        let err = max_abs(&(&direct - &mapped));
        assert!(
            err < 2e-10 * max_abs(&direct),
            "{name}: {err:e} of {:e}",
            max_abs(&direct)
        );
    }
}

#[test]
fn measuring_a_wavenumber_spectrum_with_a_wavelength_line_shape() {
    // The practical case: model on a wavenumber grid, instrument response given in nm. The same
    // measurement computed in wavelength (basis mapped instead) must agree.
    let nu = linspace(19_880.0, 20_120.0, 24001);
    let hats_nu = Space::hats(&nu).unwrap();
    let f: Array1<f64> = nu.iter().map(|v| 1.0 + 0.3 * (v / 7.0).sin()).collect();
    let ils_nm =
        // ±1.8 nm supports, inside the grid's 497.02-503.02 nm
        Space::profiles(&[499.12, 500.0, 500.7], &[Profile::gaussian(0.3).unwrap()], 6.0).unwrap();
    let g_nu = measure(
        &ils_nm.mapped(NM, true).unwrap(),
        &hats_nu,
        Normalization::Integral,
        U,
    )
    .apply(f.view());
    let g_nm = measure(
        &ils_nm,
        &hats_nu.mapped(NM, false).unwrap(),
        Normalization::Integral,
        U,
    )
    .apply(f.view());
    for (a, b) in g_nu.iter().zip(g_nm.iter()) {
        assert!((a - b).abs() < 1e-10, "{a} vs {b}");
    }
    // and a constant stays a constant
    let ones = Array1::<f64>::ones(nu.len());
    let g1 = measure(
        &ils_nm.mapped(NM, true).unwrap(),
        &hats_nu,
        Normalization::Integral,
        U,
    )
    .apply(ones.view());
    let worst = g1.iter().fold(0.0f64, |m, v| m.max((v - 1.0).abs()));
    assert!(worst < 1e-12, "constant: {worst:e}");
}

#[test]
fn integrals_with_the_jacobian_are_preserved() {
    for (name, a, _) in pairs() {
        let i = a.integrals(U);
        let j = a.mapped(NM, true).unwrap().integrals(U);
        for (x, y) in i.iter().zip(&j) {
            assert!((x - y).abs() < 1e-11 * x.abs(), "{name}: {x} vs {y}");
        }
    }
}

#[test]
fn affine_maps_stay_exact_polynomials() {
    // nm -> μm: a hat stays a hat, integrals scale by 1e-3 without the jacobian
    let hats = Space::hats(&linspace(400.0, 401.0, 11)).unwrap();
    let um = CoordinateMap::Affine {
        offset: 0.0,
        scale: 1e-3,
    };
    let m = hats.mapped(um, false).unwrap();
    let a = hats.integrals(U);
    let b = m.integrals(U);
    for (x, y) in a.iter().zip(&b) {
        assert!((x * 1e-3 - y).abs() < 1e-18);
    }
    let g = inner_products(&m, &m, U).to_dense();
    let h = inner_products(&hats, &hats, U).to_dense() * 1e-3;
    let err = max_abs(&(&g - &h));
    // hats 0.1 nm wide at 400 nm: conditioning ε·400/0.1 ≈ 1e-12
    assert!(
        err < 1e-11 * max_abs(&h),
        "affine: {err:e} of {:e}",
        max_abs(&h)
    );
}

#[test]
fn derivatives_with_respect_to_wavelength_parameters_of_a_mapped_kernel() {
    let nu = linspace(19_800.0, 20_200.0, 8001);
    let hats_nu = Space::hats(&nu).unwrap();
    // A Gaussian, whose truncation edges carry ~1e-8 of the peak: the finite difference of a
    // kernel with heavier truncated tails picks up kinks of W(c) (derivatives.rs covers those
    // unmapped); this test is of the chain rule through the map.
    let build = |c: f64, s: f64| {
        Space::profiles(&[499.7123 + c], &[Profile::gaussian(0.2 * s).unwrap()], 6.0)
            .unwrap()
            .mapped(NM, true)
            .unwrap()
    };
    for (p, plus, minus) in [
        // the conditioning noise (~1e-11) over the step stays ~1e-6 of the derivative, and the
        // truncation error (h/σ)² ~ 3e-9
        (Parameter::Center, build(1e-5, 1.0), build(-1e-5, 1.0)),
        (
            Parameter::Scale,
            build(0.0, 1e-5f64.exp()),
            build(0.0, (-1e-5f64).exp()),
        ),
    ] {
        let (_, dw) =
            measure_derivative(&build(0.0, 1.0), &hats_nu, Normalization::Integral, U, p).unwrap();
        let fd = (measure(&plus, &hats_nu, Normalization::Integral, U).to_dense()
            - measure(&minus, &hats_nu, Normalization::Integral, U).to_dense())
            / 2e-5;
        let err = max_abs(&(&dw.to_dense() - &fd));
        assert!(
            err < 1e-5 * max_abs(&fd),
            "{p:?}: {err:e} of {:e}",
            max_abs(&fd)
        );
    }
}

#[test]
fn reciprocal_maps_reject_supports_through_zero() {
    let s = Space::boxes(&[-1.0, 1.0]).unwrap();
    assert!(s.mapped(NM, false).is_err());
}
