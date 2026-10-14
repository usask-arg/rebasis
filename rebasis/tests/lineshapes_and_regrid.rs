//! Line-shape profiles, B-splines, weighted and bounded projection, and fitting.

use ndarray::{Array1, Array2};
use rebasis::{
    Apodization, Bounds, Error, Fit, Normalization, Penalty, Profile, Projection, Space, Weight,
    evaluate, measure,
};

const U: Weight = Weight::Uniform;

fn linspace(a: f64, b: f64, n: usize) -> Vec<f64> {
    (0..n)
        .map(|i| a + (b - a) * i as f64 / (n - 1) as f64)
        .collect()
}

fn all_symmetric_profiles() -> Vec<(&'static str, Profile, f64)> {
    vec![
        ("gaussian", Profile::gaussian(0.1).unwrap(), 6.0),
        ("lorentzian", Profile::lorentzian(0.05).unwrap(), 40.0),
        ("voigt", Profile::voigt(0.08, 0.03).unwrap(), 40.0),
        (
            "pseudo_voigt",
            Profile::pseudo_voigt(0.1, 0.4).unwrap(),
            40.0,
        ),
        (
            "super_gaussian",
            Profile::super_gaussian(0.1, 0.1, 3.5).unwrap(),
            6.0,
        ),
        (
            "smoothed_box",
            Profile::smoothed_box(0.2, 0.05).unwrap(),
            6.0,
        ),
        (
            "fts_none",
            Profile::fts(10.0, Apodization::NONE).unwrap(),
            40.0,
        ),
        (
            "fts_triangle",
            Profile::fts(10.0, Apodization::Triangle).unwrap(),
            40.0,
        ),
        (
            "fts_hamming",
            Profile::fts(10.0, Apodization::HAMMING).unwrap(),
            40.0,
        ),
        (
            "fts_blackman_harris",
            Profile::fts(10.0, Apodization::BLACKMAN_HARRIS).unwrap(),
            40.0,
        ),
        (
            "fts_nb_strong",
            Profile::fts(10.0, Apodization::NORTON_BEER_STRONG).unwrap(),
            40.0,
        ),
    ]
}

#[test]
fn every_symmetric_profile_returns_the_center_value_of_a_linear_function() {
    // wide enough that no kernel is truncated by the grid ends
    let x = linspace(-5.0, 15.0, 40001);
    let hires = Space::hats(&x).unwrap();
    let f: Array1<f64> = x.iter().map(|v| 3.0 * v - 2.0).collect();
    let centers = [4.0, 5.123, 6.5];
    for (name, p, cutoff) in all_symmetric_profiles() {
        let ils = Space::profiles(&centers, &[p], cutoff).unwrap();
        let g = measure(&ils, &hires, Normalization::Integral, U).apply(f.view());
        for (gi, c) in g.iter().zip(centers) {
            assert!(
                (gi - (3.0 * c - 2.0)).abs() < 1e-11,
                "{name}: {gi} vs {}",
                3.0 * c - 2.0
            );
        }
    }
}

#[test]
fn clamped_bsplines_are_a_partition_of_unity_and_degree_one_is_hats() {
    let bp = [0.0, 0.3, 0.35, 1.0, 1.7, 2.0];
    let xq = linspace(0.0, 1.999_999, 301);
    for degree in 0..=3 {
        let s = Space::bsplines_clamped(&bp, degree).unwrap();
        assert_eq!(s.len(), bp.len() + degree - 1);
        let e = evaluate(&s, &xq).unwrap();
        let sums = e.row_sums();
        for (x, v) in xq.iter().zip(sums) {
            assert!((v - 1.0).abs() < 1e-14, "degree {degree} at {x}: {v}");
        }
    }
    let h = Space::hats(&bp).unwrap();
    let b1 = Space::bsplines_clamped(&bp, 1).unwrap();
    let a = rebasis::gram(&h, U).to_dense();
    let b = rebasis::gram(&b1, U).to_dense();
    assert!((&a - &b).iter().all(|d| d.abs() < 1e-15));
}

#[test]
fn projection_onto_cubic_splines_reproduces_cubics() {
    let fine_x = linspace(0.0, 1.0, 4001);
    let fine = Space::hats(&fine_x).unwrap();
    // cubic sampled on fine hats is only piecewise linear, so project a cubic B-spline space
    // onto a finer cubic B-spline space instead (nested spaces: exact)
    let coarse = Space::bsplines_clamped(&linspace(0.0, 1.0, 6), 3).unwrap();
    let finer = Space::bsplines_clamped(&linspace(0.0, 1.0, 11), 3).unwrap();
    // coefficients of p(x) = x³ − x on the coarse space by fitting point values
    let xs = linspace(0.0, 1.0, 50);
    let fit = Fit::new(
        &Space::deltas(&xs).unwrap(),
        &coarse,
        Normalization::None,
        U,
        0.0,
        Penalty::Derivative(2),
    )
    .unwrap();
    let data: Array1<f64> = xs.iter().map(|x| x * x * x - x).collect();
    let c = fit.apply(data.view());
    let g = Projection::new(&coarse, &finer, U).unwrap().apply(c.view());
    let xq = linspace(0.0, 0.999, 37);
    let vals = evaluate(&finer, &xq).unwrap().apply(g.view());
    for (x, v) in xq.iter().zip(vals.iter()) {
        assert!((v - (x * x * x - x)).abs() < 1e-12, "{x}: {v}");
    }
    let _ = fine;
}

#[test]
fn bounded_projection_does_not_overshoot_and_conserves() {
    let x = linspace(0.0, 1.0, 1001);
    let fine = Space::hats(&x).unwrap();
    let coarse = Space::hats(&linspace(0.0, 1.0, 21)).unwrap();
    let f: Array1<f64> = x
        .iter()
        .map(|&v| if (0.3..0.62).contains(&v) { 1.0 } else { 0.0 })
        .collect();
    let p = Projection::new(&fine, &coarse, U).unwrap();
    let f2 = f.clone().insert_axis(ndarray::Axis(0));
    let plain = p.apply_last_axis(f2.view());
    let bounded = p.apply_bounded_last_axis(f2.view(), Bounds::Local);
    assert!(
        plain.iter().any(|&v| !(-1e-12..=1.0 + 1e-12).contains(&v)),
        "plain should overshoot"
    );
    assert!(bounded.iter().all(|&v| (-1e-14..=1.0 + 1e-14).contains(&v)));
    // the bounded result is the best bounded L2 fit: no worse than clipping the plain one
    let fine_vals = |g: &Array2<f64>| {
        let e = evaluate(&coarse, &x[..x.len() - 1]).unwrap();
        e.apply(g.row(0))
    };
    let err = |v: Array1<f64>| {
        v.iter()
            .zip(f.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
    };
    let clipped = plain.mapv(|v| v.clamp(0.0, 1.0));
    assert!(err(fine_vals(&bounded)) <= err(fine_vals(&clipped)) * 1.0001);
    let mass = coarse.integrals(U);
    let total = |g: &Array2<f64>| g.row(0).iter().zip(&mass).map(|(a, b)| a * b).sum::<f64>();
    let truth: f64 = f.iter().zip(fine.integrals(U)).map(|(a, b)| a * b).sum();
    assert!((total(&plain) - truth).abs() < 1e-13);
    assert!((total(&bounded) - truth).abs() < 1e-13);
}

#[test]
fn area_weighted_latitude_regridding_conserves_the_area_integral() {
    let w = Weight::cos_latitude_degrees();
    let fine = Space::boxes(&linspace(-90.0, 90.0, 181)).unwrap();
    let coarse = Space::boxes(&[-90.0, -60.0, -15.0, 0.0, 30.0, 90.0]).unwrap();
    let f: Array1<f64> = fine
        .centers()
        .iter()
        .map(|lat| 1.0 + (lat / 20.0).sin())
        .collect();
    let g = Projection::new(&fine, &coarse, w).unwrap().apply(f.view());
    let a: f64 = f.iter().zip(fine.integrals(w)).map(|(x, m)| x * m).sum();
    let b: f64 = g.iter().zip(coarse.integrals(w)).map(|(x, m)| x * m).sum();
    assert!((a - b).abs() < 1e-12 * a.abs());
    // for nested boxes this is the area-weighted average of the fine boxes
    let fm = fine.integrals(w);
    let (num, den) = (0..30).fold((0.0, 0.0), |(n, d), k| (n + f[k] * fm[k], d + fm[k]));
    assert!((g[0] - num / den).abs() < 1e-13);
}

#[test]
fn histopolation_is_smooth_and_conservative() {
    // bin averages of a smooth function -> quadratic spline whose bin averages match
    let edges = linspace(0.0, 4.0, 17);
    let bins = Space::boxes(&edges).unwrap();
    let truth = |x: f64| (1.3 * x).sin() + 0.2 * x;
    let fine_x = linspace(0.0, 4.0, 40001);
    let fine = Space::hats(&fine_x).unwrap();
    let ft: Array1<f64> = fine_x.iter().map(|&x| truth(x)).collect();
    let data = measure(&bins, &fine, Normalization::Integral, U).apply(ft.view());

    let model = Space::bsplines_clamped(&edges, 2).unwrap();
    let fit = Fit::new(
        &bins,
        &model,
        Normalization::Integral,
        U,
        1e-8,
        Penalty::Derivative(2),
    )
    .unwrap();
    let c = fit.apply(data.view());
    let back = fit.design().apply(c.view());
    for (a, b) in back.iter().zip(data.iter()) {
        assert!(
            (a - b).abs() < 1e-6,
            "bin average not reproduced: {a} vs {b}"
        );
    }
    let xq = linspace(0.0, 3.999, 77);
    let vals = evaluate(&model, &xq).unwrap().apply(c.view());
    for (x, v) in xq.iter().zip(vals.iter()) {
        // The minimum-curvature fit has f'' ≈ 0 at the ends (like a natural spline) while the
        // truth has |f''| ≈ 1.5 there, so expect ~h² f'' / 8 ≈ 1e-2 at the end bins and
        // spline accuracy inside.
        let tol = if (0.5..=3.5).contains(x) { 2e-3 } else { 2e-2 };
        assert!((v - truth(*x)).abs() < tol, "{x}: {v} vs {}", truth(*x));
    }
}

#[test]
fn histopolation_reproduces_functions_the_penalty_does_not_see() {
    // a linear function has zero ∫ f''², so its bin averages must give it back exactly
    let edges = [0.0, 0.2, 0.5, 0.6, 1.1, 1.5, 2.0];
    let bins = Space::boxes(&edges).unwrap();
    let model = Space::bsplines_clamped(&edges, 2).unwrap();
    let centers = bins.centers();
    let data: Array1<f64> = centers.iter().map(|c| 2.0 * c - 1.0).collect(); // box averages
    let fit = Fit::new(
        &bins,
        &model,
        Normalization::Integral,
        U,
        1e-3,
        Penalty::Derivative(2),
    )
    .unwrap();
    let c = fit.apply(data.view());
    let xq = linspace(0.0, 1.999, 41);
    let vals = evaluate(&model, &xq).unwrap().apply(c.view());
    for (x, v) in xq.iter().zip(vals.iter()) {
        assert!((v - (2.0 * x - 1.0)).abs() < 1e-11, "{x}: {v}");
    }
}

#[test]
fn underdetermined_fit_without_smoothing_is_singular() {
    let bins = Space::boxes(&linspace(0.0, 1.0, 5)).unwrap();
    let model = Space::bsplines_clamped(&linspace(0.0, 1.0, 5), 2).unwrap();
    let err = Fit::new(
        &bins,
        &model,
        Normalization::Integral,
        U,
        0.0,
        Penalty::Derivative(2),
    )
    .unwrap_err();
    assert!(matches!(err, Error::Singular(m) if m.contains("smoothing")));
}

#[test]
fn tabulated_pchip_and_linear_agree_on_linear_data() {
    use rebasis::Interpolation;
    let xs = vec![vec![-1.0, -0.2, 0.0, 0.5, 1.0]];
    let ys = vec![vec![0.0, 0.8, 1.0, 0.5, 0.0]; 2];
    let lin = Space::tabulated(&[2.0, 3.0], &xs, &ys, Interpolation::Linear).unwrap();
    let pch = Space::tabulated(&[2.0, 3.0], &xs, &ys, Interpolation::Pchip).unwrap();
    let li = lin.integrals(U);
    let pi = pch.integrals(U);
    // same nodes; integrals close but not equal (pchip is curved)
    for (a, b) in li.iter().zip(&pi) {
        assert!((a - b).abs() < 0.05 && (a - b).abs() > 0.0);
    }
}

#[test]
fn harmonics_values_and_box_averages() {
    let h = Space::harmonics(2, 360.0, 0.0).unwrap();
    assert_eq!(h.len(), 4);
    let xs = [0.0, 37.0, 181.5, 359.0];
    let e = evaluate(&h, &xs).unwrap().to_dense();
    for (r, &x) in xs.iter().enumerate() {
        let a = x.to_radians();
        let expected = [a.cos(), a.sin(), (2.0 * a).cos(), (2.0 * a).sin()];
        for c in 0..4 {
            assert!((e[[r, c]] - expected[c]).abs() < 1e-14, "{x} {c}");
        }
    }
    // the full-circle average of each wave is zero; a partial box has a closed form
    let boxes = Space::boxes(&[0.0, 360.0]).unwrap();
    let full = measure(&boxes, &h, Normalization::Integral, U).to_dense();
    assert!(full.iter().all(|v| v.abs() < 1e-14));
    let part = Space::boxes(&[20.0, 130.0]).unwrap();
    let w = measure(&part, &h, Normalization::Integral, U).to_dense();
    let (a, b) = (20f64.to_radians(), 130f64.to_radians());
    let width = b - a;
    let expected = [
        (b.sin() - a.sin()) / width,
        (a.cos() - b.cos()) / width,
        ((2.0 * b).sin() - (2.0 * a).sin()) / (2.0 * width),
        ((2.0 * a).cos() - (2.0 * b).cos()) / (2.0 * width),
    ];
    for c in 0..4 {
        assert!((w[[0, c]] - expected[c]).abs() < 1e-14, "{c}");
    }
}

#[test]
fn periodic_bsplines_wrap_and_partition_unity() {
    let bp = [0.0, 40.0, 100.0, 170.0, 240.0, 300.0, 360.0];
    for degree in 0..=3 {
        let s = Space::bsplines_periodic(&bp, degree).unwrap();
        assert_eq!(s.len(), 6);
        let xs: Vec<f64> = (0..720).map(|k| k as f64 * 0.5).collect();
        let e = evaluate(&s, &xs).unwrap();
        for (x, v) in xs.iter().zip(e.row_sums()) {
            assert!((v - 1.0).abs() < 1e-13, "degree {degree} at {x}: {v}");
        }
        let tp = std::f64::consts::TAU;
        let odd: Vec<f64> = (0..=7).map(|k| 0.1 + tp * k as f64 / 7.0).collect();
        let so = Space::bsplines_periodic(&odd, degree).unwrap();
        let e = evaluate(&so, &[0.1, 1.0, 3.3, 6.3]).unwrap();
        assert!(e.row_sums().iter().all(|v| (v - 1.0).abs() < 1e-12));
        if degree >= 1 {
            // continuous across the seam
            let a = evaluate(&s, &[1e-9]).unwrap().to_dense();
            let b = evaluate(&s, &[360.0 - 1e-9]).unwrap().to_dense();
            assert!((&a - &b).iter().all(|d| d.abs() < 1e-8));
        }
    }
}
