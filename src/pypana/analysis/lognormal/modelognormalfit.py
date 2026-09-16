
import numpy as np
from scipy.stats import lognorm

from pypana.analysis.lognormal.fit import Fit


class ModeLognormalFit(Fit):
    """A lognormal fit. """

    _n: np.floating
    _sigma: np.floating
    _mu: np.floating

    def __init__(self, n: np.floating, sigma: np.floating, mu: np.floating) -> None:
        """A single lognormal mode.

        Args:
            n: Area scaling. For one single mode the :meth:`pypana.data.size_distribution.SizeDistribution.total`.
            sigma: Width of the mode.
            mu: Log mean.
        """
        self._n = n
        self._sigma = sigma
        self._mu = mu

    @property
    def total(self) -> np.floating:
        """Number concentration of this mode."""
        return self._n

    @property
    def sigma(self) -> np.floating:
        """Width of this mode."""
        return self._sigma

    @property
    def mu(self) -> np.floating:
        """Log mean of this mode."""
        return self._mu

    def cdf(self, x: np.floating) -> np.floating:
        r"""The scaled lognormal cumulative distribution function evaluated at x.

        .. math::
            F(x) = n \frac{1}{2} \left[ 1 + erf \left( \frac{\log(x) - \mu}{\sigma \sqrt{2}} \right) \right]

        ``n`` is the area scaling. In the case of a single LogNormalFit, it is ``dN`` and in general the
        number concentration of the fitted mode.

        Args:
            x: :math:`d_p`

        Returns:
            F(x)
        """
        return self._n * lognorm.cdf(x, s=self._sigma, loc=0, scale=np.exp(self._mu))

    def pdf(self, x: np.floating) -> np.floating:
        r"""The lognormal probability density function evaluated at x.

        .. math::
            f(x) = n \frac{1}{x \sigma \sqrt{2 \pi}} \exp\left( -\frac{(\log(x) - \mu)^2}{2 \sigma^2} \right)

        for :math:`x > 0`

        ``n`` is the area scaling. In the case of a single LogNormalFit, it is  ``dN`` and in general the
        number concentration of the fitted mode.

        Args:
            x: :math:`d_p`

        Returns:
            f(x)
        """
        return self._n * lognorm.pdf(x, s=self._sigma, loc=0, scale=np.exp(self._mu))

    @property
    def peak(self) -> tuple[np.floating, np.floating]:
        r"""d_p and delta_dlogdp of this mode's maximum"""
        d_p = np.exp(self._mu)

        return d_p, np.log(10) * d_p * self.pdf(d_p)

