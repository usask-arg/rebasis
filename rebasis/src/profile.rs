//! Analytic line-shape profiles: smooth (or piecewise smooth) functions of the offset t from
//! the line center, each normalized to 1 at t = 0.

use crate::error::{Result, invalid};
use crate::special::{
    erf_diff, faddeeva, faddeeva_derivative, faddeeva_re, sinc, sinc_derivative,
    spherical_bessel_over_power,
};

const LN2: f64 = std::f64::consts::LN_2;

/// FWHM / σ of a Gaussian.
pub const FWHM_PER_SIGMA: f64 = 2.354_820_045_030_949_3;

/// The apodization A(u), u = δ / L on [−1, 1], of a Fourier-transform spectrometer with maximum
/// optical path difference L. The instrument line shape is ∫ A(u) cos(xu) du, x = 2πL(ν − ν₀).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Apodization {
    /// A(u) = Σ_m a_m cos(mπu): boxcar (none), Hann, Hamming, Blackman, Blackman–Harris.
    CosineSum([f64; 4]),
    /// A(u) = 1 − |u|, giving a sinc² line shape.
    Triangle,
    /// A(u) = Σ_i c_i (1 − u²)^i: the Norton–Beer family.
    Polynomial([f64; 5]),
}

impl Apodization {
    pub const NONE: Self = Apodization::CosineSum([1.0, 0.0, 0.0, 0.0]);
    pub const HANN: Self = Apodization::CosineSum([0.5, 0.5, 0.0, 0.0]);
    /// Also called Happ–Genzel.
    pub const HAMMING: Self = Apodization::CosineSum([0.54, 0.46, 0.0, 0.0]);
    pub const BLACKMAN: Self = Apodization::CosineSum([0.42, 0.5, 0.08, 0.0]);
    /// The 4-term Blackman–Harris window (92 dB).
    pub const BLACKMAN_HARRIS: Self = Apodization::CosineSum([0.35875, 0.48829, 0.14128, 0.01168]);
    /// Norton & Beer (1976, 1977) weak, medium and strong apodizations.
    pub const NORTON_BEER_WEAK: Self =
        Apodization::Polynomial([0.384093, -0.087577, 0.703484, 0.0, 0.0]);
    pub const NORTON_BEER_MEDIUM: Self =
        Apodization::Polynomial([0.152442, -0.136176, 0.983734, 0.0, 0.0]);
    pub const NORTON_BEER_STRONG: Self =
        Apodization::Polynomial([0.045335, 0.0, 0.554883, 0.0, 0.399782]);

    pub fn by_name(name: &str) -> Result<Self> {
        Ok(match name {
            "none" | "boxcar" => Self::NONE,
            "triangle" => Self::Triangle,
            "hann" | "hanning" => Self::HANN,
            "hamming" | "happ_genzel" => Self::HAMMING,
            "blackman" => Self::BLACKMAN,
            "blackman_harris" => Self::BLACKMAN_HARRIS,
            "norton_beer_weak" => Self::NORTON_BEER_WEAK,
            "norton_beer_medium" => Self::NORTON_BEER_MEDIUM,
            "norton_beer_strong" => Self::NORTON_BEER_STRONG,
            other => return invalid(format!("unknown apodization {other:?}")),
        })
    }

    /// ∫_{−1}^{1} A(u) cos(xu) du.
    pub fn line_shape(&self, x: f64) -> f64 {
        let pi = std::f64::consts::PI;
        match self {
            Apodization::CosineSum(a) => {
                let mut s = 2.0 * a[0] * sinc(x);
                for (m, am) in a.iter().enumerate().skip(1) {
                    if *am != 0.0 {
                        let k = m as f64 * pi;
                        s += am * (sinc(x - k) + sinc(x + k));
                    }
                }
                s
            }
            Apodization::Triangle => {
                let s = sinc(0.5 * x);
                s * s
            }
            Apodization::Polynomial(c) => {
                // ∫ (1 − u²)^i cos(xu) du = i! 2^(i+1) j_i(x) / x^i
                let g = spherical_bessel_over_power(x);
                let mut fact = 1.0;
                let mut s = 0.0;
                for i in 0..5 {
                    if i > 0 {
                        fact *= i as f64;
                    }
                    s += c[i] * fact * 2f64.powi(i as i32 + 1) * g[i];
                }
                s
            }
        }
    }
}

impl Apodization {
    /// d/dx of [`Apodization::line_shape`].
    pub fn line_shape_derivative(&self, x: f64) -> f64 {
        let pi = std::f64::consts::PI;
        match self {
            Apodization::CosineSum(a) => {
                let mut s = 2.0 * a[0] * sinc_derivative(x);
                for (m, am) in a.iter().enumerate().skip(1) {
                    if *am != 0.0 {
                        let k = m as f64 * pi;
                        s += am * (sinc_derivative(x - k) + sinc_derivative(x + k));
                    }
                }
                s
            }
            Apodization::Triangle => sinc(0.5 * x) * sinc_derivative(0.5 * x),
            Apodization::Polynomial(c) => {
                // d/dx (j_i / x^i) = −x j_(i+1) / x^(i+1)
                let g = spherical_bessel_over_power(x);
                let mut fact = 1.0;
                let mut s = 0.0;
                for i in 0..5 {
                    if i > 0 {
                        fact *= i as f64;
                    }
                    s -= c[i] * fact * 2f64.powi(i as i32 + 1) * x * g[i + 1];
                }
                s
            }
        }
    }
}

/// A parameter of a measurement kernel to differentiate with respect to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Parameter {
    /// The center (a spectral shift), for every kernel.
    Center,
    /// The logarithm of a common scale of all widths (squeeze/stretch): width · ∂/∂width, for
    /// every kernel but deltas.
    Scale,
    /// σ of a Gaussian, Voigt or smoothed box.
    Sigma,
    /// γ (half width) of a Lorentzian or Voigt.
    Gamma,
    /// η of a pseudo-Voigt.
    Eta,
    /// The power of a super-Gaussian.
    Power,
    /// The asymmetry a of a super-Gaussian (half widths w(1 − a) and w(1 + a)).
    Asymmetry,
    /// The box half width of a smoothed box.
    HalfWidth,
}

impl Parameter {
    pub fn by_name(name: &str) -> Result<Self> {
        Ok(match name {
            "center" => Parameter::Center,
            "scale" => Parameter::Scale,
            "sigma" => Parameter::Sigma,
            "gamma" => Parameter::Gamma,
            "eta" => Parameter::Eta,
            "power" => Parameter::Power,
            "asymmetry" => Parameter::Asymmetry,
            "half_width" => Parameter::HalfWidth,
            other => return invalid(format!("unknown parameter {other:?}")),
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Profile {
    /// exp(−t² / 2σ²)
    Gaussian { sigma: f64 },
    /// 1 / (1 + (t/γ)²), γ the half width at half maximum.
    Lorentzian { hwhm: f64 },
    /// Gaussian (σ) convolved with Lorentzian (half width γ).
    Voigt {
        sigma: f64,
        gamma: f64,
        inv_peak: f64,
    },
    /// η · Lorentzian + (1 − η) · Gaussian, both with half width `hwhm`.
    PseudoVoigt { hwhm: f64, eta: f64 },
    /// exp(−ln2 · |t / w|^p), w = `hwhm_left` for t < 0 and `hwhm_right` for t >= 0.
    SuperGaussian {
        hwhm_left: f64,
        hwhm_right: f64,
        power: f64,
    },
    /// A box of half width `half_width` convolved with a Gaussian (σ): a slit image blurred by
    /// optics.
    SmoothedBox {
        half_width: f64,
        sigma: f64,
        inv_peak: f64,
    },
    /// Fourier-transform spectrometer line shape for maximum optical path difference `max_opd`.
    Fts {
        max_opd: f64,
        apodization: Apodization,
        inv_peak: f64,
    },
}

fn positive(name: &str, v: f64) -> Result<()> {
    if v.is_finite() && v > 0.0 {
        Ok(())
    } else {
        invalid(format!("{name} must be finite and > 0, got {v}"))
    }
}

impl Profile {
    pub fn gaussian(sigma: f64) -> Result<Self> {
        positive("sigma", sigma)?;
        Ok(Profile::Gaussian { sigma })
    }

    pub fn lorentzian(hwhm: f64) -> Result<Self> {
        positive("hwhm", hwhm)?;
        Ok(Profile::Lorentzian { hwhm })
    }

    /// `sigma` of the Gaussian part and `gamma` (half width) of the Lorentzian part; either may
    /// be zero but not both.
    pub fn voigt(sigma: f64, gamma: f64) -> Result<Self> {
        if !(sigma.is_finite() && gamma.is_finite() && sigma >= 0.0 && gamma >= 0.0) {
            return invalid("voigt needs finite sigma >= 0 and gamma >= 0");
        }
        if sigma == 0.0 && gamma == 0.0 {
            return invalid("voigt needs sigma > 0 or gamma > 0");
        }
        if sigma == 0.0 {
            return Self::lorentzian(gamma);
        }
        if gamma == 0.0 {
            return Self::gaussian(sigma);
        }
        let y = gamma / (sigma * std::f64::consts::SQRT_2);
        Ok(Profile::Voigt {
            sigma,
            gamma,
            inv_peak: 1.0 / faddeeva_re(0.0, y),
        })
    }

    pub fn pseudo_voigt(hwhm: f64, eta: f64) -> Result<Self> {
        positive("hwhm", hwhm)?;
        if !(0.0..=1.0).contains(&eta) {
            return invalid(format!("pseudo-Voigt eta must be in [0, 1], got {eta}"));
        }
        Ok(Profile::PseudoVoigt { hwhm, eta })
    }

    pub fn super_gaussian(hwhm_left: f64, hwhm_right: f64, power: f64) -> Result<Self> {
        positive("hwhm_left", hwhm_left)?;
        positive("hwhm_right", hwhm_right)?;
        if !(power.is_finite() && power >= 1.0) {
            return invalid(format!("super-Gaussian power must be >= 1, got {power}"));
        }
        Ok(Profile::SuperGaussian {
            hwhm_left,
            hwhm_right,
            power,
        })
    }

    pub fn smoothed_box(half_width: f64, sigma: f64) -> Result<Self> {
        positive("half_width", half_width)?;
        positive("sigma", sigma)?;
        let s = sigma * std::f64::consts::SQRT_2;
        Ok(Profile::SmoothedBox {
            half_width,
            sigma,
            inv_peak: 1.0 / erf_diff(-half_width / s, half_width / s),
        })
    }

    pub fn fts(max_opd: f64, apodization: Apodization) -> Result<Self> {
        positive("max_opd", max_opd)?;
        let peak = apodization.line_shape(0.0);
        if !(peak.is_finite() && peak > 0.0) {
            return invalid("apodization must have a positive integral");
        }
        Ok(Profile::Fts {
            max_opd,
            apodization,
            inv_peak: 1.0 / peak,
        })
    }

    /// Value at offset t from the center; 1 at t = 0.
    pub fn value(&self, t: f64) -> f64 {
        match *self {
            Profile::Gaussian { sigma } => {
                let u = t / sigma;
                (-0.5 * u * u).exp()
            }
            Profile::Lorentzian { hwhm } => {
                let u = t / hwhm;
                1.0 / (1.0 + u * u)
            }
            Profile::Voigt {
                sigma,
                gamma,
                inv_peak,
            } => {
                let s = sigma * std::f64::consts::SQRT_2;
                faddeeva_re(t / s, gamma / s) * inv_peak
            }
            Profile::PseudoVoigt { hwhm, eta } => {
                let u = t / hwhm;
                eta / (1.0 + u * u) + (1.0 - eta) * (-LN2 * u * u).exp()
            }
            Profile::SuperGaussian {
                hwhm_left,
                hwhm_right,
                power,
            } => {
                let w = if t < 0.0 { hwhm_left } else { hwhm_right };
                (-LN2 * (t.abs() / w).powf(power)).exp()
            }
            Profile::SmoothedBox {
                half_width,
                sigma,
                inv_peak,
            } => {
                let s = sigma * std::f64::consts::SQRT_2;
                erf_diff((t - half_width) / s, (t + half_width) / s) * inv_peak
            }
            Profile::Fts {
                max_opd,
                apodization,
                inv_peak,
            } => apodization.line_shape(2.0 * std::f64::consts::PI * max_opd * t) * inv_peak,
        }
    }

    /// d/dt of [`Profile::value`].
    pub fn d_dt(&self, t: f64) -> f64 {
        let pi = std::f64::consts::PI;
        match *self {
            Profile::Gaussian { sigma } => -t / (sigma * sigma) * self.value(t),
            Profile::Lorentzian { hwhm } => {
                let u = t / hwhm;
                -2.0 * u / (hwhm * (1.0 + u * u).powi(2))
            }
            Profile::Voigt {
                sigma,
                gamma,
                inv_peak,
            } => {
                let s = sigma * std::f64::consts::SQRT_2;
                faddeeva_derivative(t / s, gamma / s).0 / s * inv_peak
            }
            Profile::PseudoVoigt { hwhm, eta } => {
                let u = t / hwhm;
                eta * (-2.0 * u / (hwhm * (1.0 + u * u).powi(2)))
                    + (1.0 - eta) * (-2.0 * LN2 * u / hwhm) * (-LN2 * u * u).exp()
            }
            Profile::SuperGaussian {
                hwhm_left,
                hwhm_right,
                power,
            } => {
                if t == 0.0 {
                    return 0.0;
                }
                let w = if t < 0.0 { hwhm_left } else { hwhm_right };
                let u = t.abs() / w;
                -LN2 * power * u.powf(power - 1.0) * t.signum() / w * self.value(t)
            }
            Profile::SmoothedBox {
                half_width,
                sigma,
                inv_peak,
            } => {
                let s = sigma * std::f64::consts::SQRT_2;
                let (a, b) = ((t + half_width) / s, (t - half_width) / s);
                2.0 / pi.sqrt() / s * ((-a * a).exp() - (-b * b).exp()) * inv_peak
            }
            Profile::Fts {
                max_opd,
                apodization,
                inv_peak,
            } => {
                let k = 2.0 * pi * max_opd;
                k * apodization.line_shape_derivative(k * t) * inv_peak
            }
        }
    }

    /// ∂/∂p of [`Profile::value`] at fixed offset t, for a profile-specific parameter.
    pub fn d_param(&self, t: f64, p: Parameter) -> Result<f64> {
        let pi = std::f64::consts::PI;
        Ok(match (*self, p) {
            (Profile::Gaussian { sigma }, Parameter::Sigma) => {
                t * t / (sigma * sigma * sigma) * self.value(t)
            }
            (Profile::Lorentzian { hwhm }, Parameter::Gamma) => {
                let u = t / hwhm;
                2.0 * u * u / (hwhm * (1.0 + u * u).powi(2))
            }
            (
                Profile::Voigt {
                    sigma,
                    gamma,
                    inv_peak,
                },
                Parameter::Sigma | Parameter::Gamma,
            ) => {
                // K = Re w(z) / Re w(z0), z = (t + iγ)/s, z0 = iγ/s, s = σ√2
                let s = sigma * std::f64::consts::SQRT_2;
                let (x, y) = (t / s, gamma / s);
                // ∂z/∂σ = −z/σ; ∂z/∂γ = i/s (same for z0)
                let dz = |zx: f64, zy: f64| match p {
                    Parameter::Sigma => (-zx / sigma, -zy / sigma),
                    _ => (0.0, 1.0 / s),
                };
                let re_mul = |(ar, ai): (f64, f64), (br, bi): (f64, f64)| ar * br - ai * bi;
                let dk = re_mul(faddeeva_derivative(x, y), dz(x, y));
                let dk0 = re_mul(faddeeva_derivative(0.0, y), dz(0.0, y));
                let k = faddeeva(x, y).0;
                dk * inv_peak - k * dk0 * inv_peak * inv_peak
            }
            (Profile::PseudoVoigt { hwhm, .. }, Parameter::Eta) => {
                let u = t / hwhm;
                1.0 / (1.0 + u * u) - (-LN2 * u * u).exp()
            }
            (
                Profile::SuperGaussian {
                    hwhm_left,
                    hwhm_right,
                    power,
                },
                Parameter::Power | Parameter::Asymmetry,
            ) => {
                if t == 0.0 {
                    return Ok(0.0);
                }
                let w = if t < 0.0 { hwhm_left } else { hwhm_right };
                let u = t.abs() / w;
                let up = u.powf(power);
                let k = self.value(t);
                if p == Parameter::Power {
                    -LN2 * up * u.ln() * k
                } else {
                    // ∂K/∂w · ∂w/∂a with w = h(1 ∓ a), h the mean half width
                    let h = 0.5 * (hwhm_left + hwhm_right);
                    LN2 * power * up / w * k * if t < 0.0 { -h } else { h }
                }
            }
            (
                Profile::SmoothedBox {
                    half_width,
                    sigma,
                    inv_peak,
                },
                Parameter::HalfWidth | Parameter::Sigma,
            ) => {
                let s = sigma * std::f64::consts::SQRT_2;
                let c = 2.0 / pi.sqrt();
                let d_e = |t: f64| {
                    let (a, b) = ((t + half_width) / s, (t - half_width) / s);
                    if p == Parameter::HalfWidth {
                        c / s * ((-a * a).exp() + (-b * b).exp())
                    } else {
                        -c / sigma * (a * (-a * a).exp() - b * (-b * b).exp())
                    }
                };
                let e = erf_diff((t - half_width) / s, (t + half_width) / s);
                d_e(t) * inv_peak - e * d_e(0.0) * inv_peak * inv_peak
            }
            (profile, p) => {
                return invalid(format!("{profile:?} has no parameter {p:?}"));
            }
        })
    }

    /// ∂/∂p of the support extents (left, right) returned by [`Profile::support`].
    pub fn d_support(&self, cutoff: f64, p: Parameter) -> Result<(f64, f64)> {
        Ok(match (*self, p) {
            (Profile::Gaussian { .. }, Parameter::Sigma)
            | (Profile::Lorentzian { .. }, Parameter::Gamma)
            | (Profile::SmoothedBox { .. }, Parameter::Sigma) => (cutoff, cutoff),
            (Profile::SmoothedBox { .. }, Parameter::HalfWidth) => (1.0, 1.0),
            (Profile::Voigt { sigma, gamma, .. }, Parameter::Sigma | Parameter::Gamma) => {
                let (fg, fl) = (FWHM_PER_SIGMA * sigma, 2.0 * gamma);
                let root = (0.2166 * fl * fl + fg * fg).sqrt();
                let d = if p == Parameter::Sigma {
                    0.5 * fg * FWHM_PER_SIGMA / root
                } else {
                    0.5 * (0.5346 * 2.0 + 0.2166 * fl * 2.0 / root)
                };
                (cutoff * d, cutoff * d)
            }
            (Profile::PseudoVoigt { .. }, Parameter::Eta)
            | (Profile::SuperGaussian { .. }, Parameter::Power) => (0.0, 0.0),
            (
                Profile::SuperGaussian {
                    hwhm_left,
                    hwhm_right,
                    ..
                },
                Parameter::Asymmetry,
            ) => {
                let h = 0.5 * (hwhm_left + hwhm_right);
                (-cutoff * h, cutoff * h)
            }
            (profile, p) => return invalid(format!("{profile:?} has no parameter {p:?}")),
        })
    }

    /// Half width at half maximum (approximate for the Voigt profile, Olivero & Longbothum 1977).
    pub fn hwhm(&self) -> f64 {
        match *self {
            Profile::Gaussian { sigma } => 0.5 * FWHM_PER_SIGMA * sigma,
            Profile::Lorentzian { hwhm } | Profile::PseudoVoigt { hwhm, .. } => hwhm,
            Profile::Voigt { sigma, gamma, .. } => voigt_hwhm(sigma, gamma),
            Profile::SuperGaussian {
                hwhm_left,
                hwhm_right,
                ..
            } => 0.5 * (hwhm_left + hwhm_right),
            Profile::SmoothedBox {
                half_width, sigma, ..
            } => half_width.max(0.5 * FWHM_PER_SIGMA * sigma),
            // first zero of the unapodized sinc
            Profile::Fts { max_opd, .. } => 0.5 / max_opd,
        }
    }

    /// The support [−left, right] of the profile truncated at `cutoff`, in the units documented
    /// on the corresponding `Space` constructor.
    pub fn support(&self, cutoff: f64) -> Result<(f64, f64)> {
        positive("cutoff", cutoff)?;
        let r = match *self {
            Profile::Gaussian { sigma } => (cutoff * sigma, cutoff * sigma),
            Profile::SuperGaussian {
                hwhm_left,
                hwhm_right,
                ..
            } => (cutoff * hwhm_left, cutoff * hwhm_right),
            Profile::SmoothedBox {
                half_width, sigma, ..
            } => (half_width + cutoff * sigma, half_width + cutoff * sigma),
            _ => (cutoff * self.hwhm(), cutoff * self.hwhm()),
        };
        Ok(r)
    }

    /// The largest Gauss–Legendre (8-point) sub-interval allowed at distance d >= 0 from the
    /// center; nondecreasing in d.
    pub(crate) fn step(&self, d: f64) -> f64 {
        match *self {
            Profile::Gaussian { sigma } => sigma,
            // poles at ±iγ: keep each sub-interval well inside the analyticity region
            Profile::Lorentzian { hwhm } => 0.3 * (hwhm + d),
            Profile::Voigt { sigma, gamma, .. } => {
                (0.5 * sigma).max(0.3 * gamma + 0.3 * (d - 6.0 * sigma).max(0.0))
            }
            Profile::PseudoVoigt { hwhm, .. } => {
                let sigma = hwhm / (2.0 * LN2).sqrt();
                (0.5 * sigma).max(0.3 * hwhm + 0.3 * (d - 6.0 * sigma).max(0.0))
            }
            Profile::SuperGaussian {
                hwhm_left,
                hwhm_right,
                power,
            } => {
                // |t|^p (and the |t|^p ln|t| of its power derivative) is not smooth at 0
                // unless p is an even integer: grade the sub-intervals geometrically into it
                let w = hwhm_left.min(hwhm_right);
                (w / power.max(2.0)).min(1e-8 * w + 0.3 * d)
            }
            Profile::SmoothedBox { sigma, .. } => sigma,
            // band-limited to |δ| <= L: a quarter period of the fastest oscillation
            Profile::Fts { max_opd, .. } => 0.25 / max_opd,
        }
    }
}

/// Olivero & Longbothum (1977) approximation to the Voigt HWHM, accurate to 0.02 %.
pub fn voigt_hwhm(sigma: f64, gamma: f64) -> f64 {
    let fg = FWHM_PER_SIGMA * sigma;
    let fl = 2.0 * gamma;
    0.5 * (0.5346 * fl + (0.2166 * fl * fl + fg * fg).sqrt())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn peaks_are_one_and_half_maxima_are_at_hwhm() {
        let profiles = [
            Profile::gaussian(0.7).unwrap(),
            Profile::lorentzian(0.3).unwrap(),
            Profile::pseudo_voigt(0.4, 0.3).unwrap(),
            Profile::super_gaussian(0.5, 0.5, 4.0).unwrap(),
        ];
        for p in profiles {
            assert!((p.value(0.0) - 1.0).abs() < 1e-15, "{p:?}");
            assert!((p.value(p.hwhm()) - 0.5).abs() < 1e-12, "{p:?}");
            assert!((p.value(-p.hwhm()) - 0.5).abs() < 1e-12, "{p:?}");
        }
        for p in [
            Profile::voigt(0.3, 0.2).unwrap(),
            Profile::smoothed_box(1.0, 0.2).unwrap(),
            Profile::fts(2.0, Apodization::NORTON_BEER_MEDIUM).unwrap(),
            Profile::fts(2.0, Apodization::BLACKMAN_HARRIS).unwrap(),
            Profile::fts(2.0, Apodization::Triangle).unwrap(),
        ] {
            assert!((p.value(0.0) - 1.0).abs() < 1e-14, "{p:?}");
        }
        let v = Profile::voigt(0.3, 0.2).unwrap();
        assert!((v.value(v.hwhm()) - 0.5).abs() < 1e-3);
    }

    #[test]
    fn asymmetric_super_gaussian() {
        let p = Profile::super_gaussian(0.2, 0.6, 3.0).unwrap();
        assert!((p.value(-0.2) - 0.5).abs() < 1e-15);
        assert!((p.value(0.6) - 0.5).abs() < 1e-15);
    }

    #[test]
    fn apodizations_are_normalized_windows() {
        // A(0) = 1 for every named window: the Norton–Beer coefficients sum to 1, and the
        // cosine-sum ones too.
        for a in [
            Apodization::NORTON_BEER_WEAK,
            Apodization::NORTON_BEER_MEDIUM,
            Apodization::NORTON_BEER_STRONG,
        ] {
            let Apodization::Polynomial(c) = a else {
                unreachable!()
            };
            assert!((c.iter().sum::<f64>() - 1.0).abs() < 1e-6);
        }
        for a in [
            Apodization::HANN,
            Apodization::HAMMING,
            Apodization::BLACKMAN_HARRIS,
        ] {
            let Apodization::CosineSum(c) = a else {
                unreachable!()
            };
            assert!((c.iter().sum::<f64>() - 1.0).abs() < 1e-12);
        }
    }

    #[test]
    fn fts_line_shapes_match_direct_integration_of_the_window() {
        type Window = (Apodization, fn(f64) -> f64);
        let windows: [Window; 4] = [
            (Apodization::NONE, |_| 1.0),
            (Apodization::Triangle, |u: f64| 1.0 - u.abs()),
            (Apodization::HAMMING, |u: f64| {
                0.54 + 0.46 * (std::f64::consts::PI * u).cos()
            }),
            (Apodization::NORTON_BEER_STRONG, |u: f64| {
                let v = 1.0 - u * u;
                0.045335 + 0.554883 * v * v + 0.399782 * v.powi(4)
            }),
        ];
        for (apod, a) in windows {
            for x in [0.0, 0.7, 3.0, 9.5, 25.0] {
                // composite midpoint rule, plenty for this smooth integrand
                let n = 200_000;
                let direct: f64 = (0..n)
                    .map(|k| {
                        let u = -1.0 + (k as f64 + 0.5) * 2.0 / n as f64;
                        a(u) * (x * u).cos()
                    })
                    .sum::<f64>()
                    * 2.0
                    / n as f64;
                assert!((apod.line_shape(x) - direct).abs() < 1e-8, "{apod:?} x={x}");
            }
        }
    }
}
