# Concepts

## Spaces and elements

A {class}`~rebasis.Space` is an ordered set of **elements** (functions of one variable). The
same space can serve as a **basis**, in which coefficients $f_j$ represent
$f(x) = \sum_j f_j B_j(x)$, or as a set of **kernels** $K_i$ that measure such a function.

Elements are *shapes*, not densities: boxes, hats, B-splines and every line-shape profile peak
at 1. Normalization belongs to the operation, not to the element. A consequence worth
remembering: a grid of hats, boxes or clamped B-splines is a **partition of unity**, so the
coefficients are function values (samples for hats, bin averages for boxes).

| basis | coefficients | represents |
|---|---|---|
| {meth}`~rebasis.Space.hats` | values at the nodes | the linear interpolant: use for samples |
| {meth}`~rebasis.Space.boxes` | bin averages | a piecewise-constant function |
| {meth}`~rebasis.Space.bsplines_clamped` | spline coefficients | a smooth spline (degree 0-3) |
| {meth}`~rebasis.Space.deltas` | point masses | a sum of Dirac deltas |

## The operations

With an inner product $\langle a, b \rangle = \int a(x)\, b(x)\, w(x)\, dx$ (the weight $w$ is 1
unless area weighting is requested):

**measure**: $W_{ij} = \langle K_i, B_j \rangle / \langle K_i, 1 \rangle$, so $g = W f$ are the
responses of unit-area kernels. Exact and sparse; no inversion. `normalization="rowsum"`
instead makes each row sum to one, which renormalizes kernels that extend past the end of the
basis.

**evaluate**: $E_{kj} = B_j(x_k)$, the point values.

**project**: the coefficients $g$ minimizing $\| f - \sum_i g_i G_i \|$, i.e. the solution of
$\langle G, G \rangle g = \langle G, B \rangle f$. The Gram matrix $\langle G, G \rangle$ is
banded for local bases and is factored once. Projection conserves $\int f w$ whenever the
target can represent a constant, is the identity onto the same space, and is interpolation
onto deltas. With `bounded`, the best approximation within bounds that still conserves the
integral is found exactly (a small banded quadratic program).

**fit**: the coefficients $c$ of a model space minimizing
$\| W c - d \|^2 + \lambda P(c)$ for measurements $d$, with a roughness penalty
$P = \int (f^{(r)})^2$ (a smoothing spline) or differences of coefficients (P-splines).

## Accuracy

Inner products are split at every knot of both elements. Between knots, products of
polynomials (degree at most 6) are integrated exactly by 4-point Gauss–Legendre; anything
involving an analytic profile, a weight or a non-affine change of coordinate is integrated in
sub-intervals sized to its local smoothness with 8 points, which is accurate to rounding. There
are no tolerances to choose.

"Rounding" is relative to the problem: a hat 0.0025 nm wide at 500 nm is evaluated from
$x - a$ with $x$ known to $\epsilon \cdot 500$ nm, so its values carry relative errors of order
$\epsilon \cdot 500 / 0.0025 \approx 4 \times 10^{-11}$ in any implementation.

## Truncation

Line-shape profiles are truncated at a `cutoff` (in the profile's natural width unit, see
{doc}`lineshapes`). Within the support every integral is exact; the normalization of
{func}`~rebasis.measure` divides by the truncated area, so a truncated kernel is still a
weighted average. Heavy-tailed profiles (Lorentzian, Voigt) default to wide supports.
