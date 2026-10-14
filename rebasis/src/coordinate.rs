//! Changes of coordinate: an element defined in u used in x = forward(u).

use crate::error::{Result, invalid};

/// A monotone map x = forward(u).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CoordinateMap {
    /// x = offset + scale · u (unit conversion, Doppler shift).
    Affine { offset: f64, scale: f64 },
    /// x = k / u, its own inverse (wavelength in nm ↔ wavenumber in cm⁻¹ with k = 10⁷). Only for
    /// supports on one side of 0.
    Reciprocal { k: f64 },
}

impl CoordinateMap {
    /// Wavelength in nm ↔ wavenumber in cm⁻¹ (either direction).
    pub const NM_CMINV: Self = CoordinateMap::Reciprocal { k: 1e7 };

    pub fn validate(&self) -> Result<()> {
        match *self {
            CoordinateMap::Affine { offset, scale } => {
                if !(offset.is_finite() && scale.is_finite() && scale != 0.0) {
                    return invalid("affine map needs finite offset and nonzero finite scale");
                }
            }
            CoordinateMap::Reciprocal { k } => {
                if !(k.is_finite() && k != 0.0) {
                    return invalid("reciprocal map needs a finite nonzero k");
                }
            }
        }
        Ok(())
    }

    #[inline]
    pub fn forward(&self, u: f64) -> f64 {
        match *self {
            CoordinateMap::Affine { offset, scale } => offset + scale * u,
            CoordinateMap::Reciprocal { k } => k / u,
        }
    }

    #[inline]
    pub fn inverse(&self, x: f64) -> f64 {
        match *self {
            CoordinateMap::Affine { offset, scale } => (x - offset) / scale,
            CoordinateMap::Reciprocal { k } => k / x,
        }
    }

    /// du/dx.
    #[inline]
    pub fn d_inverse(&self, x: f64) -> f64 {
        match *self {
            CoordinateMap::Affine { scale, .. } => 1.0 / scale,
            CoordinateMap::Reciprocal { k } => -k / (x * x),
        }
    }

    /// d²u/dx².
    #[inline]
    pub fn d2_inverse(&self, x: f64) -> f64 {
        match *self {
            CoordinateMap::Affine { .. } => 0.0,
            CoordinateMap::Reciprocal { k } => 2.0 * k / (x * x * x),
        }
    }

    pub fn is_affine(&self) -> bool {
        matches!(self, CoordinateMap::Affine { .. })
    }

    /// The largest quadrature sub-interval at x for which the map is smooth enough (the
    /// reciprocal has a pole at 0); infinite for affine maps.
    pub(crate) fn step(&self, x: f64) -> f64 {
        match self {
            CoordinateMap::Affine { .. } => f64::INFINITY,
            CoordinateMap::Reciprocal { .. } => 0.2 * x.abs(),
        }
    }

    /// The image [lo, hi] of [ulo, uhi]; errors if the map is singular on it.
    pub fn image(&self, ulo: f64, uhi: f64) -> Result<(f64, f64)> {
        if let CoordinateMap::Reciprocal { .. } = self {
            if !(ulo > 0.0 || uhi < 0.0) {
                return invalid(format!(
                    "a reciprocal map needs supports on one side of 0, got [{ulo}, {uhi}]"
                ));
            }
        }
        let (a, b) = (self.forward(ulo), self.forward(uhi));
        Ok((a.min(b), a.max(b)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reciprocal_is_its_own_inverse() {
        let m = CoordinateMap::NM_CMINV;
        assert!((m.forward(500.0) - 20_000.0).abs() < 1e-9);
        assert!((m.inverse(m.forward(437.3)) - 437.3).abs() < 1e-12);
        let h = 1e-4;
        let fd = (m.inverse(500.0 + h) - m.inverse(500.0 - h)) / (2.0 * h);
        assert!((m.d_inverse(500.0) - fd).abs() < 1e-9 * fd.abs());
        assert!(m.image(-1.0, 1.0).is_err());
        assert_eq!(m.image(400.0, 500.0).unwrap(), (20_000.0, 25_000.0));
    }
}
