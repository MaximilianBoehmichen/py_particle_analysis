"""The SizeDistribution, it holds the data of the chosen :class:`~pypana.data.defs.quantity.Quantity`."""

from collections.abc import Callable
from functools import cached_property
from typing import ClassVar, Literal, Self

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, model_validator
from scipy.optimize import curve_fit
from scipy.stats import lognorm

from pypana.analysis.lognormal import (
    LogNormalFit,
    LogNormalFitType,
    MixtureLognormalFit,
    ModeLognormalFit,
)
from pypana.console import console
from pypana.data.bin_axis import BinAxis
from pypana.data.defs import DataType, DataTypeLike, FloatArray, Normalization, Quantity
from pypana.pana_error import ParticleAnalysisError
from pypana.utils.debug import Debuggable


class SizeDistribution(BaseModel, Debuggable):
    """One :class:`~pypana.data.defs.quantity.Quantity`'s value arrays."""

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        ignored_types=(cached_property,),
        populate_by_name=True,
        validate_assignment=True,
    )

    quantity: Quantity = Field(
        frozen=True,
        description="The interpretation of the stored values.",
    )
    axis: BinAxis = Field(
        description="The axis for this quantity. Cannot be changed after instantiation.",
        frozen=True,
    )
    raw_delta: FloatArray | None = Field(
        default=None,
        alias="delta",
        description="Δ per bin in the quantity's canonical unit (e.g. ΔN [1/cm³]).",
    )
    raw_delta_dlogdp: FloatArray | None = Field(
        default=None,
        alias="delta_dlogdp",
        description="Normalized dX/dlog₁₀(d_p); same unit as `delta`.",
    )
    provenance: Literal["measured", "derived"] = Field(
        default="measured",
        frozen=True,
        description="Whether this distribution was reported by the instrument or derived "
        "from another quantity (e.g. dV from dN). Normalization conversion "
        "(Δ ↔ Δ/dlogdp) does not count as derivation.",
    )  # unsupported now, for when conversions are implemented.
    distribution_fit: LogNormalFit | None = None

    _action_log: list[str] = PrivateAttr(default_factory=list)

    _PAIRS: ClassVar[dict[str, str]] = {
        "raw_delta": "raw_delta_dlogdp",
        "raw_delta_dlogdp": "raw_delta",
    }

    # public names that map onto a raw_* field. Assigning to
    # one of these writes the source-of-truth field instead of the read-only cached_property.
    _WRITABLE_ALIASES: ClassVar[dict[str, str]] = {
        "delta": "raw_delta",
        "delta_dlogdp": "raw_delta_dlogdp",
    }

    MAX_AUTO_FIT_MODES: ClassVar[int] = 5
    """Maximum number of :class:`pypana.analysis.lognormal.modelognormalfit.ModeLognormalFit` to fit."""

    MAX_SIGMA: ClassVar[float] = np.log(3)
    """Maximum sigma for each individual :class:`pypana.analysis.lognormal.modelognormalfit.ModeLognormalFit` to fit.
    Translates to a geometric standard deviation of 3.
    """

    BIC_MARGIN_COEFFICIENT: ClassVar[float] = 2.2
    r"""Scales the minimum BIC improvement required per added mode.
    
    The BIC difference between adjacent mode counts must exceed :math:`BIC_MARGIN_COEFFICIENT * \sqrt{n_\text{bins}}`.
    This magic number was found empirically and may change in the future.
    """

    def __setattr__(self, name: str, value: object) -> None:
        name = self._WRITABLE_ALIASES.get(name, name)
        super().__setattr__(name, value)
        self._action_log.append(f"set {name} to {value}")

        paired = self._PAIRS.get(name, None)
        if paired is not None and value is not None:
            super().__setattr__(paired, None)

        self._invalidate_caches()

    def _invalidate_caches(self) -> None:
        """Invalidates values for cached properties."""
        for key in list(self.__dict__):
            if key not in type(self).model_fields:
                self.__dict__.pop(key)
                self._action_log.append(f"invalidated {key}")

    @model_validator(mode="after")
    def _check_values(self) -> Self:
        """Checks that at least one representation is present and fits the axis."""
        if self.raw_delta is None and self.raw_delta_dlogdp is None:
            raise ValueError(
                "At least one of `delta` or `delta_dlogdp` must be supplied."
            )

        for values in (self.raw_delta, self.raw_delta_dlogdp):
            if values is None:
                continue

            if values.ndim != 1:
                raise ValueError("Values must be 1D.")

            if values.size != self.axis.n_bins:
                raise ValueError(
                    f"Got {values.size} values but the axis has {self.axis.n_bins} bins."
                )

        return self

    def __getitem__(self, key: DataTypeLike) -> FloatArray:
        """Read this distribution's values in the requested representation.

        Args:
            key: A data type whose quantity matches this distribution. ``"dN"`` (per-bin)
                or ``"dN/dlogdp"`` (normalized). A bare quantity defaults to the per-bin form.

        Returns:
            The per-bin (``delta``) or normalized (``delta_dlogdp``) values.

        Raises:
            KeyError: If the requested quantity differs from this distribution's quantity.
        """
        requested = DataType.parse(key)

        if requested.quantity is not self.quantity:
            raise KeyError(
                f"This is a {self.quantity.full_name} distribution; "
                f"cannot read {requested.quantity.full_name} values from it."
            )

        if requested.normalization is Normalization.DLOG_DP:
            return self.delta_dlogdp

        return self.delta

    def _mixture_seeds(self, modes: int) -> FloatArray:
        """Starting parameters for a mixture fit.

        Args:
            modes: Quantity of seeds.

        Returns:
            Diameters [m], ascending.
        """
        cumulative = np.nancumsum(self.delta) / self.total
        quantiles = (2 * np.arange(modes) + 1) / (2 * modes)

        return self.axis.d_p[np.searchsorted(cumulative, quantiles)]

    def _fit_mixture(self, modes: int) -> tuple[MixtureLognormalFit, FloatArray]:
        """Fits count ``modes`` lognormal modes to the distribution.

        Args:
            modes: Number of modes.

        Returns:
            The fit and its residuals.
        """
        # NaN bins are ignored
        measured = np.isfinite(self.delta)
        d_lower = self.axis.d_lower[measured]
        d_upper = self.axis.d_upper[measured]
        values = self.delta[measured]

        def binned(_: FloatArray | None, *params: np.floating) -> FloatArray:
            summed = np.zeros(d_lower.size)

            for i in range(modes):
                n, sigma, mu = params[3 * i : 3 * i + 3]
                scale = np.exp(mu)
                summed += n * (
                    lognorm.cdf(d_upper, s=sigma, loc=0, scale=scale)
                    -lognorm.cdf(d_lower, s=sigma, loc=0, scale=scale)
                )

            return np.asarray(summed, dtype=float)

        p0: list[float] = []
        lower: list[float] = []
        upper: list[float] = []

        for geo_mean in self._mixture_seeds(modes):
            p0 += [
                self.total / modes,
                np.log(1.25),  # assume monodisperse particles per mode
                np.log(geo_mean),
            ]
            lower += [0.0, 1e-6, -np.inf]
            upper += [np.inf, self.MAX_SIGMA, np.inf]

        popt, _ = curve_fit(
            binned,
            self.axis.d_p[measured],
            values,
            p0=p0,
            bounds=(lower, upper),
            maxfev=50_000,
        )

        _fit = MixtureLognormalFit([
            ModeLognormalFit(n=popt[3 * i], sigma=popt[3 * i + 1], mu=popt[3 * i + 2])
            for i in range(modes)]
        )

        return _fit, values - binned(None, *popt)

    @staticmethod
    def _bic(residuals: FloatArray, modes: int) -> float:
        """BIC for the least squares fit."""
        n = residuals.size
        chi_squared = float(np.sum(residuals**2))
        log_likelihood = -n / 2 * (np.log(2 * np.pi * chi_squared / n) + 1)

        return (3 * modes + 1) * np.log(n) - 2 * log_likelihood

    @cached_property
    def delta(self) -> FloatArray:
        """Concentration per bin, lazily converted if only the normalized form was supplied."""
        if self.raw_delta is not None:
            return self.raw_delta

        assert self.raw_delta_dlogdp is not None
        return (self.raw_delta_dlogdp * self.axis.delta_log_d_p).astype(float)

    @cached_property
    def delta_dlogdp(self) -> FloatArray:
        """Normalized distribution, lazily converted if only the per-bin form was supplied."""
        if self.raw_delta_dlogdp is not None:
            return self.raw_delta_dlogdp

        assert self.raw_delta is not None
        return (self.raw_delta / self.axis.delta_log_d_p).astype(float)

    @cached_property
    def total(self) -> float:
        """Total concentration, integrated over all bins, in the quantity's canonical unit."""
        return float(np.nansum(self.delta))

    @cached_property
    def geo_mean(self) -> float:
        """Geometric mean diameter, weighted by this quantity [m]."""
        if self.total == 0:
            return 0.0

        return float(
            10 ** (np.nansum(np.log10(self.axis.d_p) * self.delta) / self.total)
        )

    @cached_property
    def geo_std_dev(self) -> float:
        """Geometric standard deviation, weighted by this quantity."""
        if self.total == 0:
            return 1.0

        log_dg = np.log10(self.geo_mean)
        var = (
            np.nansum(self.delta * (np.log10(self.axis.d_p) - log_dg) ** 2) / self.total
        )

        return float(10 ** np.sqrt(var))

    @cached_property
    def mean(self) -> float:
        """Mean diameter, weighted by this quantity [m]."""
        if self.total == 0:
            return 0.0

        return float(np.nansum(self.axis.d_p * self.delta) / self.total)

    @cached_property
    def median(self) -> float:
        """Median diameter, weighted by this quantity [m]."""
        if self.total == 0:
            return 0.0

        cum = np.cumsum(self.delta)
        return float(np.interp(0.5 * self.total, cum, self.axis.d_p))

    @cached_property
    def mode(self) -> float:
        """Diameter of the maximum of the normalized distribution [m]."""
        return float(self.axis.d_p[int(np.argmax(self.delta_dlogdp))])

    def apply(self, func: Callable[[FloatArray], FloatArray]) -> Self:
        """Applies a function to the stored values, in place.

        The function is applied to the source-of-truth representation (whichever
        raw array is present); the other representation is re-derived from the result.

        Args:
            func: Maps the current values to new values of the same shape.

        Returns:
            Itself, after applying the changes.
        """
        if self.raw_delta is not None:
            self.raw_delta = func(self.raw_delta)
        elif self.raw_delta_dlogdp is not None:  # pragma: no branch
            self.raw_delta_dlogdp = func(self.raw_delta_dlogdp)

        return self

    def cut(self, d: tuple[float, float]) -> Self:
        """Sets bins with midpoint diameter outside the lower and higher bound to zero, in place.

        Args:
            d: The boundaries as (lower, higher) [m].

        Returns:
            Itself, after applying the changes.
        """
        d_lo, d_hi = d

        if d_lo >= d_hi:
            raise ValueError(f"d_lo ({d_lo}) must be less than d_hi ({d_hi})")

        outside = (self.axis.d_p < d_lo) | (self.axis.d_p > d_hi)

        def zero_outside(values: FloatArray) -> FloatArray:
            new = values.copy()
            new[outside] = 0.0
            return new

        return self.apply(zero_outside)

    def fit(
        self,
        *,
        fit_type: LogNormalFitType = "mode",
        modes: int | None = 1,
        loss: Literal["linear", "soft_l1", "huber", "cauchy", "arctan"] = "linear",
        outlier_scale: float = 0.05,
    ) -> LogNormalFit:
        """Fits the specified function to the SizeDistribution.

        The model is fitted against the contents of each bin.
        Bins holding ``NaN`` are treated as missing and are ignored.

        Args:
            fit_type: ``"mode"`` for a single lognormal mode, ``"mixture"`` for multiple modes.
            modes: Number of modes to fit or autodetect with BIC. Only used for ``"mixture"``.
            loss: The residual loss function used for fitting ``fit_type="mode"``.
                If ``"linear"`` becomes unstable, try ``"soft_l1"``.
            outlier_scale: Residual relative magnitude at which a bin is treated as outlier.
                Ignored for ``loss="linear"``.

        Returns:
            The fit.
        """
        if self.distribution_fit:
            console.print("Overriding previous fit!")

        if fit_type == "mode":
            measured = np.isfinite(self.delta)
            d_lower = self.axis.d_lower[measured]
            d_upper = self.axis.d_upper[measured]

            def binned(
                _: FloatArray,
                n: np.floating,
                sigma: np.floating,
                mu: np.floating,
            ) -> FloatArray:
                scale = np.exp(mu)

                return np.asarray(
                    n * (
                        lognorm.cdf(d_upper, s=sigma, loc=0, scale=scale)
                        - lognorm.cdf(d_lower, s=sigma, loc=0, scale=scale)
                    ),
                    dtype=float,
                )

            popt, _ = curve_fit(
                binned,
                self.axis.d_p[measured],
                self.delta[measured],
                p0=[
                    self.total,
                    max(np.log(self.geo_std_dev), 0.005),  # completely empty SizeDistribution: geo_std_dev = 0.0
                    np.log(self.mode)
                ],
                bounds=([0.0, 1e-6, -np.inf], [np.inf, np.inf, np.inf]),
                loss=loss,
                f_scale=outlier_scale * float(np.nanmax(self.delta))
            )
            fit = ModeLognormalFit(n=popt[0], sigma=popt[1], mu=popt[2])
            self.distribution_fit = fit

            return fit

        if fit_type == "mixture":
            if modes is not None:
                fit, _ = self._fit_mixture(modes)
                self.distribution_fit = fit

                return fit

            fits: dict[int, tuple[MixtureLognormalFit, float]] = {}

            for candidate in range(1, self.MAX_AUTO_FIT_MODES + 1):
                try:
                    fit, residuals = self._fit_mixture(candidate)
                except RuntimeError:
                    continue

                fits[candidate] = (fit, self._bic(residuals, candidate))

            if not fits:
                raise ParticleAnalysisError("No mixture fit converged.")

            margin = self.BIC_MARGIN_COEFFICIENT * np.sqrt(
                np.count_nonzero(
                    np.isfinite(self.delta)
                )
            )
            chosen = min(fits)

            for candidate in sorted(fits):
                if fits[candidate][1] < fits[chosen][1] - margin:
                    chosen = candidate

            self.distribution_fit = fits[chosen][0]

            return fits[chosen][0]

        raise ParticleAnalysisError(f"Unknown fit type {fit_type!r}.")

    def summary(self) -> dict[str, object]:
        """Summary of this distribution's key derived quantities.

        Returns:
            A dict overview.
        """
        return {
            "quantity": self.quantity.full_name,
            "provenance": self.provenance,
            "n_bins": self.axis.n_bins,
            "d_p_min": float(self.axis.d_p.min()),
            "d_p_max": float(self.axis.d_p.max()),
            "total": self.total,
            "geo_mean": self.geo_mean,
            "geo_std_dev": self.geo_std_dev,
            "mean": self.mean,
            "median": self.median,
            "mode": self.mode,
        }
