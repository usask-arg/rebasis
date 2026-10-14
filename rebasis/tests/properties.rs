//! Properties every rebasis map should have, checked on small examples.

use ndarray::{Array1, Array2, ArrayView1};
use rebasis::{Element, Normalization, Projection, Shape, Space, Weight, evaluate, measure};

const U: Weight = Weight::Uniform;

fn linspace(a: f64, b: f64, n: usize) -> Vec<f64> {
    (0..n)
        .map(|i| a + (b - a) * i as f64 / (n - 1) as f64)
        .collect()
}

fn close(a: &[f64], b: &[f64], tol: f64) {
    assert_eq!(a.len(), b.len());
    for (i, (x, y)) in a.iter().zip(b).enumerate() {
        assert!((x - y).abs() <= tol, "index {i}: {x} vs {y}");
    }
}

#[test]
fn gaussian_kernel_reproduces_linear_function_on_hats() {
    // A linear f is exact in the hat basis, and a symmetric unit-area kernel returns f(center).
    let nodes = linspace(300.0, 310.0, 2001);
    let hires = Space::hats(&nodes).unwrap();
    let centers = linspace(302.0, 308.0, 13);
    let ils = Space::gaussians(&centers, &[0.25], 6.0).unwrap();
    let w = measure(&ils, &hires, Normalization::Integral, U);
    let f: Array1<f64> = nodes.iter().map(|x| 2.0 * x - 1.0).collect();
    let g = w.apply(f.view());
    let expected: Vec<f64> = centers.iter().map(|x| 2.0 * x - 1.0).collect();
    close(g.as_slice().unwrap(), &expected, 1e-10);
}

#[test]
fn rowsum_renormalizes_kernels_truncated_by_the_grid_edge() {
    let nodes = linspace(0.0, 1.0, 101);
    let hires = Space::hats(&nodes).unwrap();
    let ils = Space::gaussians(&[0.0, 0.5, 1.0], &[0.05], 5.0).unwrap();
    let ones = Array1::<f64>::ones(nodes.len());
    let g = measure(&ils, &hires, Normalization::RowSum, U).apply(ones.view());
    close(g.as_slice().unwrap(), &[1.0, 1.0, 1.0], 1e-14);
    let g = measure(&ils, &hires, Normalization::Integral, U).apply(ones.view());
    // half of each edge kernel hangs off the grid
    close(g.as_slice().unwrap(), &[0.5, 1.0, 0.5], 1e-12);
}

#[test]
fn tabulated_line_shape_matches_box_kernel() {
    let nodes = linspace(0.0, 10.0, 1001);
    let hires = Space::hats(&nodes).unwrap();
    // a (nearly) unit-width box as a tabulated shape, stretched to width 2
    let shape = Shape::piecewise_linear(
        &[-0.5, -0.5 + 1e-12, 0.5 - 1e-12, 0.5],
        &[0.0, 1.0, 1.0, 0.0],
    )
    .unwrap();
    let tab = Space::shifted(shape, &[3.0, 5.0], &[2.0]).unwrap();
    let boxes = Space::new(vec![
        Element::boxcar(2.0, 4.0).unwrap(),
        Element::boxcar(4.0, 6.0).unwrap(),
    ]);
    let f: Array1<f64> = nodes.iter().map(|x| x.sin()).collect();
    let a = measure(&tab, &hires, Normalization::Integral, U).apply(f.view());
    let b = measure(&boxes, &hires, Normalization::Integral, U).apply(f.view());
    close(a.as_slice().unwrap(), b.as_slice().unwrap(), 1e-9);
}

#[test]
fn evaluate_on_hats_is_linear_interpolation() {
    let nodes = [0.0, 1.0, 3.0, 4.0];
    let f = [1.0, 2.0, 0.0, 5.0];
    let e = evaluate(&Space::hats(&nodes).unwrap(), &[0.0, 0.5, 2.0, 3.5, 4.0]).unwrap();
    let g = e.apply(ArrayView1::from(&f));
    close(g.as_slice().unwrap(), &[1.0, 1.5, 1.0, 2.5, 5.0], 1e-15);
}

#[test]
fn projection_onto_the_same_space_is_the_identity() {
    for space in [
        Space::hats(&linspace(0.0, 1.0, 17)).unwrap(),
        Space::boxes(&linspace(0.0, 1.0, 17)).unwrap(),
    ] {
        let p = Projection::new(&space, &space, U).unwrap().to_dense();
        let eye = Array2::<f64>::eye(space.len());
        close(p.as_slice().unwrap(), eye.as_slice().unwrap(), 1e-12);
    }
}

#[test]
fn projection_to_coarse_boxes_is_conservative_averaging() {
    let fine = Space::boxes(&linspace(0.0, 1.0, 13)).unwrap();
    let coarse = Space::boxes(&linspace(0.0, 1.0, 4)).unwrap();
    let f: Array1<f64> = (0..12).map(|i| i as f64).collect();
    let g = Projection::new(&fine, &coarse, U).unwrap().apply(f.view());
    close(g.as_slice().unwrap(), &[1.5, 5.5, 9.5], 1e-13);
}

#[test]
fn projection_conserves_the_integral() {
    let fine_nodes = linspace(0.0, 2.0, 401);
    let fine = Space::hats(&fine_nodes).unwrap();
    let f: Array1<f64> = fine_nodes.iter().map(|x| (3.0 * x).cos() + x * x).collect();
    let total: f64 = f.iter().zip(fine.integrals(U)).map(|(a, b)| a * b).sum();
    // uneven coarse grids, each a partition of unity on [0, 2]
    for coarse in [
        Space::hats(&[0.0, 0.3, 0.35, 1.1, 2.0]).unwrap(),
        Space::boxes(&[0.0, 0.13, 0.9, 1.0, 2.0]).unwrap(),
    ] {
        let g = Projection::new(&fine, &coarse, U).unwrap().apply(f.view());
        let projected: f64 = g.iter().zip(coarse.integrals(U)).map(|(a, b)| a * b).sum();
        assert!(
            (projected - total).abs() < 1e-12 * total.abs(),
            "{projected} vs {total}"
        );
    }
}

#[test]
fn projection_reproduces_functions_in_the_target_span() {
    // A linear function on fine hats projects exactly onto any coarser hat grid.
    let fine_nodes = linspace(0.0, 1.0, 101);
    let fine = Space::hats(&fine_nodes).unwrap();
    let coarse_nodes = [0.0, 0.2, 0.25, 0.7, 1.0];
    let coarse = Space::hats(&coarse_nodes).unwrap();
    let f: Array1<f64> = fine_nodes.iter().map(|x| 4.0 * x - 2.0).collect();
    let g = Projection::new(&fine, &coarse, U).unwrap().apply(f.view());
    let expected: Vec<f64> = coarse_nodes.iter().map(|x| 4.0 * x - 2.0).collect();
    close(g.as_slice().unwrap(), &expected, 1e-12);
}

#[test]
fn projection_onto_deltas_is_interpolation() {
    let nodes = linspace(0.0, 1.0, 11);
    let hats = Space::hats(&nodes).unwrap();
    let x = [0.05, 0.5, 0.93];
    let f: Array1<f64> = nodes.iter().map(|x| x * x).collect();
    let g = Projection::new(&hats, &Space::deltas(&x).unwrap(), U)
        .unwrap()
        .apply(f.view());
    let e = evaluate(&hats, &x).unwrap().apply(f.view());
    close(g.as_slice().unwrap(), e.as_slice().unwrap(), 1e-15);
}

#[test]
fn duplicate_target_elements_are_singular() {
    let src = Space::hats(&linspace(0.0, 1.0, 5)).unwrap();
    let dst = Space::new(vec![Element::boxcar(0.0, 1.0).unwrap(); 2]);
    assert!(matches!(
        Projection::new(&src, &dst, U),
        Err(rebasis::Error::Singular(_))
    ));
}

#[test]
fn apply_last_axis_matches_apply_and_dense() {
    let fine = Space::hats(&linspace(0.0, 1.0, 50)).unwrap();
    let coarse = Space::hats(&linspace(0.0, 1.0, 7)).unwrap();
    let p = Projection::new(&fine, &coarse, U).unwrap();
    let f = Array2::from_shape_fn((3, 50), |(k, j)| (k as f64 + 1.0) * (j as f64).sqrt());
    let g = p.apply_last_axis(f.view());
    for k in 0..3 {
        let gk = p.apply(f.row(k));
        close(g.row(k).as_slice().unwrap(), gk.as_slice().unwrap(), 1e-13);
    }
    let gd = f.dot(&p.to_dense().t());
    close(g.as_slice().unwrap(), gd.as_slice().unwrap(), 1e-12);
}
