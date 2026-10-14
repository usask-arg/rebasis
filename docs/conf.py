"""Sphinx configuration for the rebasis documentation.

The pages are executed (myst-nb) against the installed extension; ``uv run`` rebuilds it when the
Rust sources change::

    uv run --extra docs sphinx-build -W --keep-going -b html docs docs/_build/html

Intersphinx downloads the Python, numpy, scipy and xarray inventories; set
``REBASIS_DOCS_OFFLINE=1`` to skip that without network access.
"""

import os
from importlib.metadata import version as _version

project = "rebasis"
author = "rebasis contributors"
copyright = "2026, rebasis contributors"
release = _version("pyrebasis")
version = release

extensions = [
    "myst_nb",
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "numpydoc",
]

exclude_patterns = ["_build", "jupyter_execute", "**.ipynb_checkpoints"]
source_suffix = {".md": "myst-nb", ".rst": "restructuredtext"}

myst_enable_extensions = ["colon_fence", "dollarmath", "amsmath"]
# Every page runs on every build and fails it if a cell raises or writes to stderr.
nb_execution_mode = "force"
nb_execution_raise_on_error = True
nb_execution_timeout = 300
nb_output_stderr = "warn"

html_theme = "furo"
html_title = "rebasis"
html_theme_options = {
    "source_repository": "https://github.com/usask-arg/rebasis/",
    "source_branch": "main",
    "source_directory": "docs/",
}

autodoc_default_options = {"members": True, "undoc-members": False}
autodoc_member_order = "bysource"
autodoc_typehints = "none"
numpydoc_show_class_members = False
numpydoc_xref_param_type = False

OFFLINE = os.environ.get("REBASIS_DOCS_OFFLINE", "") not in ("", "0")
intersphinx_mapping = (
    {}
    if OFFLINE
    else {
        "python": ("https://docs.python.org/3", None),
        "numpy": ("https://numpy.org/doc/stable", None),
        "scipy": ("https://docs.scipy.org/doc/scipy", None),
        "xarray": ("https://docs.xarray.dev/en/stable", None),
    }
)
intersphinx_timeout = 30
