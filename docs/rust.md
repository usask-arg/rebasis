# Rust crate

The computational core is the [`rebasis`](https://crates.io/crates/rebasis) crate, which has no
Python dependency (the Python bindings live in the separate `rebasis-py` crate). The Python API
is a thin layer over it, and the names correspond:

| Python | Rust |
|---|---|
| `rb.Space.hats(x)`, `rb.Space.voigts(...)` | `Space::hats(&x)`, `Space::profiles(&centers, &[Profile::voigt(σ, γ)?], cutoff)` |
| `rb.measure(k, b)` | `measure(&k, &b, Normalization::Integral, Weight::Uniform)` → `Csr` |
| `rb.measure_derivative(k, b, "center")` | `measure_derivative(&k, &b, norm, weight, Parameter::Center)` |
| `rb.Projection(src, dst)` | `Projection::new(&src, &dst, weight)` |
| `space.to_wavenumber(jacobian=True)` | `space.mapped(CoordinateMap::NM_CMINV, true)` |

Results are compressed sparse row matrices (`Csr`) with `apply` and `apply_last_axis` for
`ndarray` inputs. The crate's API reference is on [docs.rs](https://docs.rs/rebasis).
