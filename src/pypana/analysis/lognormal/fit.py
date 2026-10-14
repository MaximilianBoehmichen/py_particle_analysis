"""Generic Fit class for distribution fitting."""

from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import ArrayLike, NDArray


class Fit(ABC):
    """Abstract base class for pypana fits."""

    @abstractmethod
    def cdf(self, x: ArrayLike) -> NDArray[np.float64]:
        r"""The cumulative distribution function of the fit.

        :math:`F(d_p) = P(D_p \leq d_p)`

        Args:
            x:  :math:`d_p`

        Returns:
            The probability of :math:`D_p \leq d_p`
        """
        ...

    @abstractmethod
    def pdf(self, x: ArrayLike) -> NDArray[np.float64]:
        r"""The probability density function of the fit.

        :math:`f(d_p) = \frac{d}{dd_p} F(d_p)`

        Args:
            x:  :math:`d_p`

        Returns:
            The derivative of :math:`F(d_p)`.
        """
        ...
