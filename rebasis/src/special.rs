//! Special functions for the line-shape profiles.

use std::sync::OnceLock;

/// Order of Weideman's rational approximation of the Faddeeva function. N = 40 is accurate to
/// about 3e-15 of w(iy) for all real x and y >= 0.
const WEIDEMAN_N: usize = 40;

struct Weideman {
    l: f64,
    /// p(Z) = Σ a[n] Z^n
    a: Vec<f64>,
}

fn weideman() -> &'static Weideman {
    static W: OnceLock<Weideman> = OnceLock::new();
    W.get_or_init(|| {
        let n = WEIDEMAN_N;
        let m = 2 * n;
        let l = (n as f64 / std::f64::consts::SQRT_2).sqrt();
        let pi = std::f64::consts::PI;
        let f: Vec<(f64, f64)> = (-(m as i64) + 1..m as i64)
            .map(|k| {
                let t = l * (k as f64 * pi / (2 * m) as f64).tan();
                (k as f64, (-t * t).exp() * (l * l + t * t))
            })
            .collect();
        let a = (1..=n)
            .map(|j| {
                f.iter()
                    .map(|(k, fk)| fk * (pi * k * j as f64 / m as f64).cos())
                    .sum::<f64>()
                    / (2 * m) as f64
            })
            .collect();
        Weideman { l, a }
    })
}

/// Re w(x + iy) for y >= 0, w the Faddeeva function (Weideman 1994, SIAM J. Numer. Anal. 31).
///
/// The Voigt profile is Re w((t + iγ) / (σ√2)) / (σ√(2π)).
pub fn faddeeva_re(x: f64, y: f64) -> f64 {
    let w = weideman();
    // z = x + iy; L − iz = (L + y) − ix; L + iz = (L − y) + ix
    let (dr, di) = (w.l + y, -x);
    let (nr, ni) = (w.l - y, x);
    let den = dr * dr + di * di;
    // Z = (L + iz) / (L − iz)
    let zr = (nr * dr + ni * di) / den;
    let zi = (ni * dr - nr * di) / den;
    // Horner for p(Z)
    let (mut pr, mut pi) = (0.0, 0.0);
    for &c in w.a.iter().rev() {
        let r = pr * zr - pi * zi + c;
        pi = pr * zi + pi * zr;
        pr = r;
    }
    // 1 / (L − iz) and its square
    let (ir, ii) = (dr / den, -di / den);
    let (sr, si) = (ir * ir - ii * ii, 2.0 * ir * ii);
    2.0 * (pr * sr - pi * si) + ir / std::f64::consts::PI.sqrt()
}

/// erf(b) − erf(a) for a <= b, without cancellation when both are in the same tail.
pub fn erf_diff(a: f64, b: f64) -> f64 {
    if a >= 0.0 {
        libm::erfc(a) - libm::erfc(b)
    } else if b <= 0.0 {
        libm::erfc(-b) - libm::erfc(-a)
    } else {
        libm::erf(b) - libm::erf(a)
    }
}

/// sin(x) / x.
#[inline]
pub fn sinc(x: f64) -> f64 {
    if x.abs() < 1e-4 {
        1.0 - x * x / 6.0
    } else {
        x.sin() / x
    }
}

/// j_n(x) / x^n for n = 0..=4, j_n the spherical Bessel functions. Finite at x = 0, where it is
/// 1 / (2n + 1)!!.
pub fn spherical_bessel_over_power(x: f64) -> [f64; 5] {
    if x.abs() < 4.0 {
        bessel_series(x)
    } else {
        bessel_recurrence(x)
    }
}

/// Σ_k (−x²/2)^k / (k! (2n + 2k + 1)!!)
fn bessel_series(x: f64) -> [f64; 5] {
    let mut out = [0.0; 5];
    let h = -0.5 * x * x;
    let mut dfact = 1.0; // (2n + 1)!!
    for (n, o) in out.iter_mut().enumerate() {
        dfact *= (2 * n + 1) as f64;
        let mut term = 1.0 / dfact;
        let mut sum = term;
        for k in 1..60 {
            term *= h / (k as f64 * (2 * n + 2 * k + 1) as f64);
            sum += term;
            if term.abs() < 1e-17 * sum.abs() {
                break;
            }
        }
        *o = sum;
    }
    out
}

/// Upward recurrence from j_0 and j_1, stable for n < x.
fn bessel_recurrence(x: f64) -> [f64; 5] {
    let mut out = [0.0; 5];
    let (s, c) = x.sin_cos();
    let mut jm1 = s / x;
    let mut j = s / (x * x) - c / x;
    out[0] = jm1;
    out[1] = j / x;
    for n in 1..4 {
        let jp1 = (2 * n + 1) as f64 / x * j - jm1;
        jm1 = j;
        j = jp1;
        out[n + 1] = j / x.powi(n as i32 + 1);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn faddeeva_special_values() {
        // w(iy) = exp(y²) erfc(y); w(x) = exp(−x²) + i·(…) on the real axis
        for y in [0.0, 0.1, 1.0, 5.0] {
            let expected = libm::exp(y * y) * libm::erfc(y);
            let got = faddeeva_re(0.0, y);
            assert!((got - expected).abs() < 1e-14 * expected.max(1e-3), "y={y}");
        }
        for x in [0.3, 1.0, 2.5, 6.0] {
            let got = faddeeva_re(x, 0.0);
            assert!((got - (-x * x).exp()).abs() < 1e-14, "x={x}");
        }
        // asymptotic w(iy) = (1 − 1/(2y²) + 3/(4y⁴) − …) / (√π y)
        let y = 50.0f64;
        let expected = (1.0 - 0.5 / (y * y) + 0.75 / y.powi(4) - 1.875 / y.powi(6)
            + 6.5625 / y.powi(8))
            / (std::f64::consts::PI.sqrt() * y);
        assert!((faddeeva_re(0.0, y) / expected - 1.0).abs() < 1e-13);
        // Lorentzian limit for large |z|: Re w ≈ y / (√π (x² + y²))
        let (x, y) = (300.0, 2.0);
        let expected = y / (std::f64::consts::PI.sqrt() * (x * x + y * y));
        assert!((faddeeva_re(x, y) / expected - 1.0).abs() < 1e-4);
    }

    #[test]
    fn spherical_bessel_series_and_recurrence_agree() {
        for x in [3.0, 4.0, 5.5] {
            let a = bessel_series(x);
            let b = bessel_recurrence(x);
            for n in 0..5 {
                assert!(
                    (a[n] - b[n]).abs() < 1e-13,
                    "x={x} n={n}: {} {}",
                    a[n],
                    b[n]
                );
            }
        }
        let z = spherical_bessel_over_power(0.0);
        assert_eq!(z, [1.0, 1.0 / 3.0, 1.0 / 15.0, 1.0 / 105.0, 1.0 / 945.0]);
        // j_1(x) = sin x / x² − cos x / x at x = 7
        let x = 7.0f64;
        let j1 = x.sin() / (x * x) - x.cos() / x;
        assert!((spherical_bessel_over_power(x)[1] - j1 / x).abs() < 1e-15);
    }

    #[test]
    fn erf_diff_tails() {
        let d = erf_diff(8.0, 9.0);
        let expected = libm::erfc(8.0) - libm::erfc(9.0);
        assert!((d - expected).abs() < 1e-15 * expected);
        assert!((erf_diff(-0.5, 0.5) - 2.0 * libm::erf(0.5)).abs() < 1e-16);
    }
}
