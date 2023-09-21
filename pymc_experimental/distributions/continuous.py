#   Copyright 2022 The PyMC Developers
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

# coding: utf-8
"""
Experimental probability distributions for stochastic nodes in PyMC.

The imports from pymc are not fully replicated here: add imports as necessary.
"""

from typing import List, Optional, Tuple, Union

import numpy as np
import pytensor.tensor as pt
from pymc.distributions.continuous import PositiveContinuous
from pymc.distributions.dist_math import check_parameters
from pymc.distributions.distribution import Continuous
from pymc.distributions.shape_utils import rv_size_is_none
from pymc.pytensorf import floatX
from pytensor.tensor.random.op import RandomVariable
from pytensor.tensor.variable import TensorVariable
from scipy import stats


class GenExtremeRV(RandomVariable):
    name: str = "Generalized Extreme Value"
    ndim_supp: int = 0
    ndims_params: List[int] = [0, 0, 0]
    dtype: str = "floatX"
    _print_name: Tuple[str, str] = ("Generalized Extreme Value", "\\operatorname{GEV}")

    def __call__(self, mu=0.0, sigma=1.0, xi=0.0, size=None, **kwargs) -> TensorVariable:
        return super().__call__(mu, sigma, xi, size=size, **kwargs)

    @classmethod
    def rng_fn(
        cls,
        rng: Union[np.random.RandomState, np.random.Generator],
        mu: np.ndarray,
        sigma: np.ndarray,
        xi: np.ndarray,
        size: Tuple[int, ...],
    ) -> np.ndarray:
        # Notice negative here, since remainder of GenExtreme is based on Coles parametrization
        return stats.genextreme.rvs(c=-xi, loc=mu, scale=sigma, random_state=rng, size=size)


gev = GenExtremeRV()


class GenExtreme(Continuous):
    r"""
    Univariate Generalized Extreme Value log-likelihood

    The cdf of this distribution is

    .. math::

       G(x \mid \mu, \sigma, \xi) = \exp\left[ -\left(1 + \xi z\right)^{-\frac{1}{\xi}} \right]

    where

    .. math::

        z = \frac{x - \mu}{\sigma}

    and is defined on the set:

    .. math::

        \left\{x: 1 + \xi\left(\frac{x-\mu}{\sigma}\right) > 0 \right\}.

    Note that this parametrization is per Coles (2001), and differs from that of
    Scipy in the sign of the shape parameter, :math:`\xi`.

    .. plot::

        import matplotlib.pyplot as plt
        import numpy as np
        import scipy.stats as st
        import arviz as az
        plt.style.use('arviz-darkgrid')
        x = np.linspace(-10, 20, 200)
        mus = [0., 4., -1.]
        sigmas = [2., 2., 4.]
        xis = [-0.3, 0.0, 0.3]
        for mu, sigma, xi in zip(mus, sigmas, xis):
            pdf = st.genextreme.pdf(x, c=-xi, loc=mu, scale=sigma)
            plt.plot(x, pdf, label=rf'$\mu$ = {mu}, $\sigma$ = {sigma}, $\xi$={xi}')
        plt.xlabel('x', fontsize=12)
        plt.ylabel('f(x)', fontsize=12)
        plt.legend(loc=1)
        plt.show()


    ========  =========================================================================
    Support   * :math:`x \in [\mu - \sigma/\xi, +\infty]`, when :math:`\xi > 0`
              * :math:`x \in \mathbb{R}` when :math:`\xi = 0`
              * :math:`x \in [-\infty, \mu - \sigma/\xi]`, when :math:`\xi < 0`
    Mean      * :math:`\mu + \sigma(g_1 - 1)/\xi`, when :math:`\xi \neq 0, \xi < 1`
              * :math:`\mu + \sigma \gamma`, when :math:`\xi = 0`
              * :math:`\infty`, when :math:`\xi \geq 1`
                where :math:`\gamma` is the Euler-Mascheroni constant, and
                :math:`g_k = \Gamma (1-k\xi)`
    Variance  * :math:`\sigma^2 (g_2 - g_1^2)/\xi^2`, when :math:`\xi \neq 0, \xi < 0.5`
              * :math:`\frac{\pi^2}{6} \sigma^2`, when :math:`\xi = 0`
              * :math:`\infty`, when :math:`\xi \geq 0.5`
    ========  =========================================================================

    Parameters
    ----------
    mu : float
        Location parameter.
    sigma : float
        Scale parameter (sigma > 0).
    xi : float
        Shape parameter
    scipy : bool
        Whether or not to use the Scipy interpretation of the shape parameter
        (defaults to `False`).

    References
    ----------
    .. [Coles2001] Coles, S.G. (2001).
        An Introduction to the Statistical Modeling of Extreme Values
        Springer-Verlag, London

    """

    rv_op = gev

    @classmethod
    def dist(cls, mu=0, sigma=1, xi=0, scipy=False, **kwargs):
        # If SciPy, use its parametrization, otherwise convert to standard
        if scipy:
            xi = -xi
        mu = pt.as_tensor_variable(floatX(mu))
        sigma = pt.as_tensor_variable(floatX(sigma))
        xi = pt.as_tensor_variable(floatX(xi))

        return super().dist([mu, sigma, xi], **kwargs)

    def logp(value, mu, sigma, xi):
        """
        Calculate log-probability of Generalized Extreme Value distribution
        at specified value.

        Parameters
        ----------
        value: numeric
            Value(s) for which log-probability is calculated. If the log probabilities for multiple
            values are desired the values must be provided in a numpy array or Pytensor tensor

        Returns
        -------
        TensorVariable
        """
        scaled = (value - mu) / sigma

        logp_expression = pt.switch(
            pt.isclose(xi, 0),
            -pt.log(sigma) - scaled - pt.exp(-scaled),
            -pt.log(sigma)
            - ((xi + 1) / xi) * pt.log1p(xi * scaled)
            - pt.pow(1 + xi * scaled, -1 / xi),
        )

        logp = pt.switch(pt.gt(1 + xi * scaled, 0.0), logp_expression, -np.inf)

        return check_parameters(
            logp, sigma > 0, pt.and_(xi > -1, xi < 1), msg="sigma > 0 or -1 < xi < 1"
        )

    def logcdf(value, mu, sigma, xi):
        """
        Compute the log of the cumulative distribution function for Generalized Extreme Value
        distribution at the specified value.

        Parameters
        ----------
        value: numeric or np.ndarray or `TensorVariable`
            Value(s) for which log CDF is calculated. If the log CDF for
            multiple values are desired the values must be provided in a numpy
            array or `TensorVariable`.

        Returns
        -------
        TensorVariable
        """
        scaled = (value - mu) / sigma
        logc_expression = pt.switch(
            pt.isclose(xi, 0), -pt.exp(-scaled), -pt.pow(1 + xi * scaled, -1 / xi)
        )

        logc = pt.switch(1 + xi * (value - mu) / sigma > 0, logc_expression, -np.inf)

        return check_parameters(
            logc, sigma > 0, pt.and_(xi > -1, xi < 1), msg="sigma > 0 or -1 < xi < 1"
        )

    def moment(rv, size, mu, sigma, xi):
        r"""
        Using the mode, as the mean can be infinite when :math:`\xi > 1`
        """
        mode = pt.switch(pt.isclose(xi, 0), mu, mu + sigma * (pt.pow(1 + xi, -xi) - 1) / xi)
        if not rv_size_is_none(size):
            mode = pt.full(size, mode)
        return mode


class ChiRV(RandomVariable):
    r"""
    A chi continuous random variable.

    The probability density function for `chi` in terms of its parameters
    :math:`\nu`, :math:`\mu`, and :math:`\sigma` is:

    .. math::
        f(x; \nu, \sigma) = \frac{x^{k-1}e^{-x^2/(2\sigma^2)}}{2^{k/2-1}\sigma^{k}\Gamma(k/2)}

    for :math:`x \geq 0`, :math:`\nu > 0`, and :math:`\sigma > 0`
    """
    name = "chi"
    ndim_supp = 0
    ndims_params = [0, 0]
    dtype = "floatX"
    _print_name = ("Chi", "\\operatorname{Chi}")

    def __call__(self, df, scale, size=None, **kwargs):
        r"""
        Draw samples from a Chi distribution.

        Signature
        ---------
        `(), () -> ()`

        Parameters
        ----------
        df
            Degrees of freedom :math:`\nu`. Must be positive.
        scale
            Scale parameter :math:`\sigma` of the distribution. Must be
            positive.
        size
            Sample shape. If the given size is, e.g. `(m, n, k)` then `m * n * k`
            independent, identically distributed random variables are
            returned. Default is `None` in which case a single random variable
            is returned.
        """
        return super().__call__(df, scale, size=size, **kwargs)

    @classmethod
    def rng_fn(
        cls,
        rng: Union[np.random.Generator, np.random.RandomState],
        df: Union[np.ndarray, float],
        scale: Union[np.ndarray, float],
        size: Optional[Union[List[int], int]],
    ) -> np.ndarray:
        return stats.chi.rvs(df, scale=scale, size=size, random_state=rng)


chi = ChiRV()


class Chi(PositiveContinuous):
    r"""
    Univariate chi log-likelihood.

    The pdf of this distribution is

    .. math::
        f(x; \nu, \sigma) = \frac{x^{k-1}e^{-x^2/(2\sigma^2)}}{2^{k/2-1}\sigma^{k}\Gamma(k/2)}

    .. plot::
        :context: close-figs

        import matplotlib.pyplot as plt
        import numpy as np
        import scipy.stats as st
        import arviz as az
        plt.style.use('arviz-darkgrid')
        x = np.linspace(0, 10, 1000)
        dfs = [1., 2., 3., 3.]
        sigmas = [1., 1., 1., 3.]
        for df, sigma in zip(dfs, sigmas):
            pdf = st.chi.pdf(x, df=df, scale=sigma)
            plt.plot(x, pdf, label=r'$\nu$ = {}, $\sigma$ = {}'.format(df, sigma))
        plt.xlabel('x', fontsize=12)
        plt.ylabel('f(x)', fontsize=12)
        plt.legend(loc=1)
        plt.show()

    ========  =========================================================================
    Support   :math:`x \in [0, \infty)`
    Mean      :math:`\sqrt{2}\sigma\Gamma((\nu + 1)/2)/\Gammma(\nu/2)`
    Variance  :math:`\sigma^2[\nu - 2\sigma^2(\Gamma(\nu+1/2)/\Gamma(\nu/2)^2)]`
    ========  =========================================================================

    Parameters
    ----------
    df : tensor_like of float
        Degrees of freedom. (df > 0).
    sigma : tensor_like of float
        Scale parameter. (sigma > 0).

    Examples
    --------
    .. code-block:: python
        with pm.Model():
            x = pm.Chi('x', 3, mu=2, sigma=5)
    """
    rv_op = chi

    @classmethod
    def dist(cls, df, sigma, **kwargs):
        df = pt.as_tensor_variable(df)
        sigma = pt.as_tensor_variable(sigma)
        return super().dist([df, sigma], **kwargs)

    def moment(rv, size, df, sigma):
        df, sigma = pt.broadcast_arrays(df, sigma)
        moment = pt.sqrt(2.0) * sigma * pt.gamma((df + 1.0) / 2.0) / pt.gamma(df / 2.0)
        if not rv_size_is_none(size):
            moment = pt.full(size, moment)
        return moment

    def logp(value, df, sigma):
        df, sigma = pt.broadcast_arrays(df, sigma)
        res = (
            (df - 1.0) * pt.log(value)
            - pt.pow((value) / sigma, 2) / 2.0
            - (df / 2.0 - 1.0) * pt.log(2.0)
            - pt.gammaln(df / 2.0)
            - df * pt.log(sigma)
        )
        res = pt.switch(pt.gt(value, 0.0), res, -np.inf)
        return check_parameters(
            res,
            [df > 0, sigma > 0],
            msg="df > 0, sigma > 0",
        )

    def logcdf(value, df, sigma):
        df, sigma = pt.broadcast_arrays(df, sigma)
        res = pt.log(pt.gammainc(df / 2.0, pt.pow(value / sigma, 2) / 2.0))
        res = pt.switch(pt.gt(value, 0.0), res, -np.inf)
        return check_parameters(
            res,
            [df > 0, sigma > 0],
            msg="df > 0, sigma > 0",
        )
