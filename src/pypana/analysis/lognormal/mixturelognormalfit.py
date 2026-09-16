import numpy as np

from pypana.analysis.lognormal.fit import Fit
from pypana.analysis.lognormal.modelognormalfit import ModeLognormalFit


class MixtureLognormalFit(Fit):
    """A multi-lognormal fit."""

    _fits: list[ModeLognormalFit]

    def __init__(self, fits: list[ModeLognormalFit]):
        """A multi-lognormal fit consisting of individual lognormal fits.

        Args:
            fits (list[ModeLognormalFit]): The lognormal fits.
        """
        self._fits = fits

    def cdf(self, x: np.floating) -> np.floating:
        r"""The lognormal cumulative distribution function evaluated at x.

        .. math::
            F(x) = \sum F_i(x)

        Args:
            x: :math:`d_p`

        Returns:
            F(x)
        """
        return np.sum([lnf.cdf(x) for lnf in self._fits], axis=0)

    def pdf(self, x: np.floating) -> np.floating:
        r"""The lognormal probability density function evaluated at x.

        .. math::
            f(x) = \sum f_i(x)

        for :math:`x > 0`

        Args:
            x: :math:`d_p`

        Returns:
            f(x)
        """
        return np.sum([lnf.pdf(x) for lnf in self._fits], axis=0)