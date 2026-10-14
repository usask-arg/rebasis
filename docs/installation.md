# Installation

## Python

```bash
pip install pyrebasis
```

The distribution is called `pyrebasis` (the name `rebasis` on PyPI belongs to an unrelated
project); the package is imported as `rebasis`. Wheels are built for Linux, macOS and Windows on
x86-64 and ARM, for Python 3.11 and later. The optional `xarray` extra adds
{mod}`rebasis.xarray`:

```bash
pip install "pyrebasis[xarray]"
```

## Rust

The core is the `rebasis` crate, with no Python dependency:

```bash
cargo add rebasis
```

## From source

With [uv](https://docs.astral.sh/uv/) and a Rust toolchain (1.85 or later):

```bash
git clone https://github.com/usask-arg/rebasis
cd rebasis
uv sync              # builds the extension
uv run pytest        # Python tests
cargo test -p rebasis
```
