import numpy as np

# np.trapezoid is numpy >= 2; the oldest supported numpy (1.26) has np.trapz.
if not hasattr(np, "trapezoid"):
    np.trapezoid = np.trapz  # noqa: NPY201
