from abc import ABC, abstractmethod

import numpy as np


class Fit(ABC):
    """Abstract base class for pypana fits."""

    @abstractmethod
    def cdf(self, x: np.floating) -> np.floating:
        r"""The cumulative distribution function of the fit.

        :math:`F(d_p) = P(D_p \leq d_p)`

        Args:
            x:  :math:`d_p`

        Returns:
            The probability of :math:`D_p \leq d_p`
        """
        ...

    @abstractmethod
    def pdf(self, x: np.floating) -> np.floating:
        r"""The probability density function of the fit.

        :math:`f(d_p) = \frac{d}{dd_p} F(d_p)`

        Args:
            x:  :math:`d_p`

        Returns:
            The derivative of :math:`F(d_p)`.
        """
        ...
