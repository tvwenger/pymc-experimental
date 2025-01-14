API Reference
***************

This reference provides detailed documentation for all modules, classes, and
methods in the current release of PyMC experimental.

.. currentmodule:: pymc_experimental
.. autosummary::
   :toctree: generated/

   marginal_model.MarginalModel
   model_builder.ModelBuilder

Inference
=========

.. currentmodule:: pymc_experimental.inference
.. autosummary::
   :toctree: generated/

   fit


Distributions
=============

.. currentmodule:: pymc_experimental.distributions
.. autosummary::
   :toctree: generated/

   GenExtreme
   GeneralizedPoisson
   DiscreteMarkovChain
   R2D2M2CP
   histogram_approximation


Utils
=====

.. currentmodule:: pymc_experimental.utils
.. autosummary::
   :toctree: generated/

   spline.bspline_interpolation
   prior.prior_from_idata


Statespace Models
=================
.. automodule:: pymc_experimental.statespace
.. toctree::
   :maxdepth: 1

   statespace/core
   statespace/filters
   statespace/models
