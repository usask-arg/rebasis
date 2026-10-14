---
file_format: mystnb
kernelspec:
  name: python3
---

# Derivatives with respect to line-shape parameters

{func}`~rebasis.measure_derivative` returns the measurement matrix $W$ and its derivative
$\partial W / \partial \theta$, row $i$ with respect to the parameter of kernel $i$. For a
spectrum $f$, `dW @ f` is the Jacobian of the measurements with respect to per-channel
parameters (it is diagonal), or with respect to one parameter shared by all channels, such as a
common spectral shift.

| parameter | kernels |
|---|---|
| `center` | all (spectral shift) |
| `scale` | all but deltas: width·∂/∂width, every width scaled together (squeeze or stretch) |
| `sigma` | Gaussian, Voigt, smoothed box |
| `gamma` | Lorentzian, Voigt |
| `eta` | pseudo-Voigt |
| `power`, `asymmetry` | super-Gaussian |
| `half_width` | smoothed box |

The derivatives are exact derivatives of what {func}`~rebasis.measure` computes. The kernel
derivative is treated as a distribution: a smooth part $\partial K / \partial\theta$, integrated
with the same quadrature as $W$, plus point terms where the kernel jumps (box edges, and the
truncation edges of profiles, which move with the parameter). The normalization is
differentiated too: $\partial W = (\partial A - W\, \partial N) / N$.

```{code-cell} ipython3
import numpy as np

import rebasis as rb

x = np.linspace(758, 772, 28001)
hires = rb.Space.hats(x)
f = 1 - 0.8 * np.exp(-(((x - 764.3) / 0.05) ** 2))  # one absorption line

centers = np.arange(760.0, 769.0, 0.2)
build = lambda shift: rb.Space.super_gaussians(centers + shift, fwhm=0.4, power=3.0)

W, dW = rb.measure_derivative(build(0.0), hires, "center")
jacobian = dW @ f  # ∂g_i / ∂(center shift)

h = 1e-6
finite_difference = (rb.measure(build(h), hires) @ f - rb.measure(build(-h), hires) @ f) / (2 * h)
print(f"largest relative difference: {np.abs(jacobian - finite_difference).max() / np.abs(jacobian).max():.1e}")
```

Computing $W$ and $\partial W$ together costs about twice {func}`~rebasis.measure` alone.
