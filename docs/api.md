# Python API

```{eval-rst}
.. module:: rebasis

Spaces
------

.. autoclass:: rebasis.Space

Operations
----------

.. autofunction:: rebasis.measure
.. autofunction:: rebasis.measure_derivative
.. autofunction:: rebasis.evaluate
.. autofunction:: rebasis.inner_products
.. autofunction:: rebasis.gram
.. autofunction:: rebasis.apply_along

.. autoclass:: rebasis.Projection
.. autofunction:: rebasis.project

.. autoclass:: rebasis.Fit
.. autofunction:: rebasis.fit

Regridding
----------

.. autofunction:: rebasis.regrid
.. autofunction:: rebasis.regrid_nd

.. automodule:: rebasis.xarray
   :members: regrid

Estimation
----------

.. autofunction:: rebasis.optimal_estimate
.. autoclass:: rebasis.Estimate
   :members: apply, weight_histogram
.. autoclass:: rebasis.Covariance
   :members: replace
.. autoclass:: rebasis.CovarianceDim
.. autoclass:: rebasis.CovarianceSum
.. autoclass:: rebasis.VarianceProfile
.. autoclass:: rebasis.MeanTerm
.. autofunction:: rebasis.log_likelihood
.. autofunction:: rebasis.fit_covariance
.. autoclass:: rebasis.CovarianceFit
   :members: estimate

Errors and constants
--------------------

.. autoexception:: rebasis.SingularError
.. autodata:: rebasis.FWHM_PER_SIGMA
   :no-value:
.. autofunction:: rebasis.faddeeva_re
```
