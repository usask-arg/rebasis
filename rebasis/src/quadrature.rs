//! Gauss–Legendre rules on [-1, 1], computed once by Newton iteration.

use std::sync::OnceLock;

pub(crate) struct Rule {
    pub nodes: Vec<f64>,
    pub weights: Vec<f64>,
}

fn gauss_legendre(n: usize) -> Rule {
    let mut nodes = vec![0.0; n];
    let mut weights = vec![0.0; n];
    for i in 0..n.div_ceil(2) {
        // Tricomi's initial guess for the i-th root, then Newton on P_n.
        let mut x = (std::f64::consts::PI * (i as f64 + 0.75) / (n as f64 + 0.5)).cos();
        let mut dp = 0.0;
        for _ in 0..100 {
            let (mut p0, mut p1) = (1.0, x);
            for k in 2..=n {
                let p2 = ((2 * k - 1) as f64 * x * p1 - (k - 1) as f64 * p0) / k as f64;
                p0 = p1;
                p1 = p2;
            }
            dp = n as f64 * (x * p1 - p0) / (x * x - 1.0);
            let dx = p1 / dp;
            x -= dx;
            if dx.abs() < 1e-16 {
                break;
            }
        }
        let w = 2.0 / ((1.0 - x * x) * dp * dp);
        nodes[i] = -x;
        nodes[n - 1 - i] = x;
        weights[i] = w;
        weights[n - 1 - i] = w;
    }
    Rule { nodes, weights }
}

/// 4 points: exact for polynomials of degree <= 7, so for products of two cubics.
pub(crate) fn gl4() -> &'static Rule {
    static RULE: OnceLock<Rule> = OnceLock::new();
    RULE.get_or_init(|| gauss_legendre(4))
}

/// 8 points: used on sub-intervals no wider than one standard deviation of a Gaussian, where it
/// is accurate to rounding.
pub(crate) fn gl8() -> &'static Rule {
    static RULE: OnceLock<Rule> = OnceLock::new();
    RULE.get_or_init(|| gauss_legendre(8))
}

/// Integrates `f` over [a, b] with `rule`.
#[inline]
pub(crate) fn integrate(rule: &Rule, a: f64, b: f64, mut f: impl FnMut(f64) -> f64) -> f64 {
    let half = 0.5 * (b - a);
    let mid = 0.5 * (a + b);
    rule.nodes
        .iter()
        .zip(&rule.weights)
        .map(|(x, w)| w * f(mid + half * x))
        .sum::<f64>()
        * half
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn weights_sum_to_two_and_rules_are_exact() {
        for rule in [gl4(), gl8()] {
            let n = rule.nodes.len();
            assert!((rule.weights.iter().sum::<f64>() - 2.0).abs() < 1e-14);
            // exact for x^(2n-1) and x^(2n-2) on [0, 1]
            for deg in [2 * n - 2, 2 * n - 1] {
                let got = integrate(rule, 0.0, 1.0, |x| x.powi(deg as i32));
                assert!(
                    (got - 1.0 / (deg as f64 + 1.0)).abs() < 1e-14,
                    "n={n} deg={deg}"
                );
            }
        }
    }
}
