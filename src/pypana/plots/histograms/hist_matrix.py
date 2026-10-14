"""Methods for plotting histograms of measurements in a matrix format."""

from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from itertools import cycle
from typing import Any, Literal

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import ticker
from matplotlib.axes import Axes
from matplotlib.container import BarContainer
from matplotlib.patches import StepPatch
from matplotlib.ticker import Formatter

from pypana.analysis.lognormal import MixtureLognormalFit, ModeLognormalFit
from pypana.analysis.lognormal.fit import Fit
from pypana.config import settings
from pypana.data.bin_axis import BinAxis
from pypana.data.defs import DataType, DataTypeLike, FloatArray, Normalization
from pypana.data.measurement import Measurement
from pypana.pana_error import ParticleAnalysisError
from pypana.plots.themes import BaseTheme
from pypana.plots.utils import coerce_formatter, linear_sci_formatter, split_kwargs

STANDARD_HIST_SINGLE_KWARGS: dict[str, Any] = {
    "legend": "lower right",
    "bar_edgecolor": "black",
    "bar_label": lambda m: f"Measurement {m.scan_nr}",
    "bar_linewidth": 0.25,
    "secondary_alpha": 0.7,
    "secondary_color": "black",
    "legend_frameon": False,
}
"""Standard kwargs for an out-of-the-box configuration of a single histogram.

These parameters may change with newer versions of pypana.
"""

STANDARD_HIST_MATRIX_KWARGS: dict[str, Any] = {
    "legend": "column",
    "bar_label": lambda m: f"Measurement {m.scan_nr}",
    "bar_linewidth": 0,
    "secondary_alpha": 0.7,
    "secondary_color": "black",
    "legend_frameon": False,
}
"""Standard kwargs for an out-of-the-box configuration of a matrix of histograms.

These parameters may change with newer versions of pypana.
"""


def plot_hist_matrix(  # pragma: no cover # noqa: PLR0912, PLR0915
    m: list[list[tuple[Measurement, ...]]],
    data_type: DataTypeLike,
    *,
    theme: BaseTheme | None = None,
    hist_type: Literal["bar", "stairs", "both"] = "bar",
    secondary: Literal["cdf", "fit_cdf", "fit_pdf"]
    | Callable[[FloatArray], FloatArray]
    | Fit
    | None = None,
    save_as: str | None = None,
    legend: Literal[
        "best",
        "upper right",
        "upper left",
        "lower left",
        "lower right",
        "right",
        "center left",
        "center right",
        "lower center",
        "upper center",
        "center",
        "row",
        "column",
    ]
    | None = "best",
    transforms: Callable[[FloatArray], FloatArray] = lambda x: x,
    pmf: bool = False,
    spines_invisible: list[Literal["left", "right", "top", "bottom"]] | None = None,
    title: str | None = "",
    xlabel: str | None = None,
    xlim: tuple[float, float] = (-np.inf, np.inf),
    xmajor_formatter: Formatter | str | None = None,
    xmajor_locations: tuple[float, ...] | None = None,
    ylabel: str | None = None,
    ylim: tuple[float, float] | None = None,
    ymajor_formatter: Formatter | str | None = None,
    yscale: Literal["linear", "log"] = "linear",
    **kwargs,
) -> None:
    """Plots the histogram of the specified measurement.

    Note:
        Not all possible matplotlib kwargs are specified in the Keyword Args section.
        Additional kwargs can be passed to matplotlib with their respective name prepended
        by the following prefixes that indicates the target:

        - ``bar_`` for the main histogram,
        - ``grid_`` for the background grid (only visible, if grid_visible=True in theme),
        - ``legend_`` for the legend,
        - ``secondary_`` for the secondary line plot,
        - ``stairs_`` for the stairs plot.

        For matplotlib kwargs, please consult the matplotlib documentation: https://matplotlib.org/stable/ .

    Args:
        m (list[list[Measurement]]): The measurement grid. The measurements are given row-major like numpy.
            Has to be a full rectangular matrix.
        data_type (DataTypeLike): The data type to plot, e.g. ``dN/dlogdp`` or ``dN``.
        theme (BaseTheme): The theme for the plot. Defaults to ``settings.THEME``.
        hist_type (str): What histogram type to display. "bar" plots a standard bar histogram,
            "stairs" plots the outlines of the histogram, and "both" plots both together.
            Defaults to ``"bar"``.
        secondary (str | Callable[[FloatArray], FloatArray] | Fit): The additional function to plot.
            ``"cdf"`` plots the cumulative sum on a secondary axis. ``"fit_pdf"`` and ``"fit_cdf"`` plot the fit
            of the first measurement of each subplot; a mixture also shows its individual modes dashed.
            A :class:`pypana.analysis.lognormal.fit.Fit` is plotted as outline in every subplot.
            A callable computes custom secondary data. Defaults to ``None``.
        save_as (str | None): The path where to save the figure. Defaults to ``None`` which does not save.
        legend (str | None): Whether to show the legend and where.
        pmf (bool): Whether to plot the measurement as probability mass function. Defaults to ``False``.
        transforms (Callable[[FloatArray], FloatArray]): A transform to apply to the plotted data of m.
            Defaults to the identity function.
        spines_invisible (list): The spines not to show. Defaults to ``None``, in which case all are plotted.
        title (str | None): The title of the plot. Defaults to ``None`` and uses an adaptive title.
        xlabel (str | None): The x-axis label of the plot. Defaults to ``None`` and uses an adaptive title.
        xlim (tuple): The x-axis lower and upper bound.
        xmajor_formatter (Formatter | str): The matplotlib ticker.Formatter for the x-axis.
        xmajor_locations (tuple): The float values between 1.0 and 10.0 where major ticks are plotted.
        ylabel (str | None): The y-axis label of the plot. Defaults to ``None`` and uses an adaptive title.
        ylim (tuple): The y-axis lower and upper bound. Can be used to give specific y-ranges on the axis.
        ymajor_formatter (Formatter | str): The matplotlib ticker.Formatter for the y-axis.
        yscale (str): The type of scaling the y-axis uses. Defaults to ``"linear"``.
        kwargs: The additional kwargs for matplotlib. See the Keyword Args section for more information.
    """
    _rows = len(m)
    _cols = len(m[0])
    _theme = theme or settings.THEME
    requested = DataType.parse(data_type)

    (
        _bar_kwargs,
        _grid_kwargs,
        _legend_kwargs,
        _secondary_kwargs,
        _stairs_kwargs,
        _save_kwargs,
    ) = split_kwargs(
        "bar_", "grid_", "legend_", "secondary_", "stairs_", "save_", **kwargs
    )

    _colors = (
        list(_theme.color_cycle.values())
        if _theme.color_cycle
        else plt.rcParams["axes.prop_cycle"].by_key()["color"]
    )

    _secondary_meta_keys = ("color", "label", "ylabel", "ylim", "yformatter")
    _secondary_color = str(_secondary_kwargs.get("color") or "black")
    _secondary_line_kwargs = {
        k: v for k, v in _secondary_kwargs.items() if k not in _secondary_meta_keys
    }
    _drawn = 0

    _handles: list = []
    _labels: list = []
    _secondary_handles: list = []
    _secondary_labels: list = []

    _bf = _bar_kwargs.pop("facecolor", None)
    _sf = _stairs_kwargs.pop("facecolor", None)
    _bar_colors = cycle([_bf]) if isinstance(_bf, str) else cycle(_colors)
    _stairs_colors = cycle([_sf]) if isinstance(_sf, str) else cycle(_colors)

    with matplotlib.rc_context(_theme.to_rcparams()):
        fig, axs = plt.subplots(
            _rows, _cols, sharex=True, sharey=True, squeeze=False, layout="constrained"
        )

        if title is not None:
            _title = title or _get_default_title()
            fig.suptitle(_title)

        for (r, c), ax in np.ndenumerate(axs):
            ax.grid()
            _m_tuple: tuple[Measurement, ...] = m[r][c]

            for _m in _m_tuple:
                _data = transforms(_m[requested].copy())
                _bar_kwargs["label"] = _resolve_label(
                    kwargs.get("bar_label"), _m, f"Measurement {_m.scan_nr}"
                )
                _stairs_kwargs["label"] = (
                    _resolve_label(
                        kwargs.get("stairs_label"), _m, f"Measurement {_m.scan_nr}"
                    )
                    if hist_type != "both"
                    else None
                )

                if pmf:
                    _data /= sum(_data)

                if hist_type in ["bar", "both"]:
                    bar = _bar_plot(_bar_kwargs, _bar_colors, _data, _m, ax)

                    _handles.append(bar)
                    _labels.append(_bar_kwargs.get("label", ""))

                if hist_type in ["stairs", "both"]:
                    stairs = _stairs_plot(_stairs_kwargs, _stairs_colors, _data, _m, ax)

                    _handles.append(stairs) if hist_type != "both" else None
                    _labels.append(
                        _stairs_kwargs.get("label", "")
                    ) if hist_type != "both" else None

                _drawn += 1

            ax2: Axes | None = None

            if secondary is not None:
                curves, on_twin, secondary_ylabel = _secondary_curves(
                    secondary,
                    _m_tuple[0],
                    requested,
                    transforms=transforms,
                    pmf=pmf,
                    color=_secondary_color,
                    colors=_colors,
                    color_offset=_drawn,
                    label_spec=kwargs.get("secondary_label"),
                    ylabel=str(_secondary_kwargs.get("ylabel") or ""),
                )

                twin = ax.twinx() if on_twin else None
                target = twin or ax

                for curve in curves:
                    (line,) = target.plot(
                        curve.x,
                        curve.y,
                        label=curve.label,
                        **{**_secondary_line_kwargs, **curve.style},
                    )
                    _secondary_handles.append(line)
                    _secondary_labels.append(curve.label)

                # configure the twin axis only after plotting, so the autoscaled upper limit is kept
                if twin is not None:
                    ax2 = twin

                    twin.yaxis.set_major_formatter(
                        coerce_formatter(_secondary_kwargs.get("yformatter"))
                        or (
                            linear_sci_formatter()
                            if yscale == "linear"
                            else ticker.LogFormatterSciNotation()
                        )
                    )
                    twin.tick_params(axis="y", direction="in")

                    _secondary_ylim = _secondary_kwargs.get("ylim")

                    if _secondary_ylim:
                        twin.set_ylim(_secondary_ylim)
                    else:
                        # override line default, since it is likely a transform of main data
                        twin.set_ylim(bottom=0)

                    if c != _cols - 1:
                        twin.yaxis.set_visible(False)
                    else:
                        twin.set_ylabel(secondary_ylabel)

            _format_ax(
                ax,
                requested,
                pmf,
                xlabel,
                xlim,
                xmajor_formatter,
                ylabel,
                ylim,
                ymajor_formatter,
                yscale,
                xmajor_locations=xmajor_locations,
            )

            h1, l1 = ax.get_legend_handles_labels()
            h2, l2 = ax2.get_legend_handles_labels() if ax2 is not None else ([], [])
            ax.legend(
                h1 + h2, l1 + l2, loc=legend, **_legend_kwargs
            ) if legend and legend not in ["row", "column"] else None

            for spine in spines_invisible or []:
                ax.spines[spine].set_visible(False)

        _secondary_handles, _secondary_labels = _deduplicate_handles_labels(
            _secondary_handles, _secondary_labels
        )

        if legend and legend == "row":
            if "ncol" not in _legend_kwargs:
                _legend_kwargs["ncol"] = len(_labels)

            fig.legend(
                _handles + _secondary_handles,
                _labels + _secondary_labels,
                loc="outside lower center",
                **_legend_kwargs,
            )

        if legend and legend == "column":
            fig.legend(
                _handles + _secondary_handles,
                _labels + _secondary_labels,
                loc="outside right upper",
                **_legend_kwargs,
            )

        if save_as is not None:
            if "transparent" not in _save_kwargs:
                _save_kwargs["transparent"] = True

            if "facecolor" not in _save_kwargs:
                _save_kwargs["facecolor"] = "white"

            fig.savefig(save_as, bbox_inches="tight", **_save_kwargs)

        plt.show()


def _format_ax(  # pragma: no cover
    ax: Axes,
    data_type: DataType,
    pmf: bool,
    xlabel: str | None,
    xlim: tuple[float, float],
    xmajor_formatter: Formatter | str | None,
    ylabel: str | None,
    ylim: tuple[float, float] | None,
    ymajor_formatter: Formatter | str | None,
    yscale: Literal["linear", "log"],
    xmajor_locations: tuple[float, ...] | None = None,
) -> None:
    """Sets the parameters for ax."""
    xmajor_locations = xmajor_locations or (1.0, 2.0, 5.0)
    xminor_locations = tuple(
        x
        for x in (1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0)
        if x not in xmajor_locations
    )

    ax.set_xscale("log")
    ax.set_yscale(yscale)

    ax.xaxis.set_major_locator(
        ticker.LogLocator(base=10.0, subs=xmajor_locations, numticks=15)
    )
    ax.xaxis.set_minor_locator(ticker.LogLocator(base=10.0, subs=xminor_locations))

    _xformatter = coerce_formatter(xmajor_formatter) or ticker.EngFormatter(unit="m")
    ax.xaxis.set_major_formatter(_xformatter)
    ax.xaxis.set_minor_formatter(ticker.NullFormatter())

    _xformatter = coerce_formatter(xmajor_formatter) or ticker.EngFormatter(unit="m")
    _yformatter = coerce_formatter(ymajor_formatter) or (
        linear_sci_formatter()
        if yscale == "linear"
        else ticker.LogFormatterSciNotation()
    )
    ax.xaxis.set_major_formatter(_xformatter)
    ax.yaxis.set_major_formatter(_yformatter)
    ax.tick_params(axis="both", direction="in", which="both")

    ax.set_xlim(*xlim) if xlim != (-np.inf, np.inf) else None
    ax.set_ylim(*ylim) if ylim else None

    _xlabel = xlabel or _get_default_xlabel()
    _ylabel = ylabel or _get_default_ylabel(data_type, pmf)

    ax.set_xlabel(_xlabel)
    ax.set_ylabel(_ylabel)
    ax.label_outer()


def _stairs_plot(  # pragma: no cover
    _stairs_kwargs: dict[str, object],
    _color_generator: Iterator[str],
    _data: FloatArray,
    _m: Measurement,
    ax: Axes,
) -> StepPatch:
    """Plots the stairs plot part."""
    _color_kwarg = {"color": next(_color_generator)}

    stairs = ax.stairs(
        values=_data,
        edges=_m.bin_boundaries,
        **_color_kwarg,
        **_stairs_kwargs,
    )
    return stairs


def _bar_plot(  # pragma: no cover
    _bar_kwargs: dict[str, object],
    _color_generator: Iterator[str],
    _data: FloatArray,
    _m: Measurement,
    ax: Axes,
) -> BarContainer:
    """Plots the bar plot part."""
    _color_kwarg = {"color": next(_color_generator)}

    bar = ax.bar(
        _m.d_p,
        _data,
        width=_m.delta_d_p,
        align="center",
        **_color_kwarg,
        **_bar_kwargs,
    )
    return bar


FIT_RESOLUTION = 500
"""Number of points a fit curve is evaluated at."""


@dataclass(frozen=True)
class _Curve:
    """One line of the secondary plot."""

    x: FloatArray
    y: FloatArray
    label: str
    style: dict[str, object] = field(default_factory=dict)


def _fit_values(
    fit: Fit, x: FloatArray, axis: BinAxis, normalization: Normalization
) -> FloatArray:
    """Evaluates a fit continuously in the representation of the histogram.

    For ``dX/dlogdp`` this is the density per decade. For ``dX`` it is the content a bin of the
    local bin width centered at ``x`` would hold, which equals the bar height at each bin center.
    """
    if normalization is Normalization.DLOG_DP:
        return np.asarray(np.log(10) * x * fit.pdf(x), dtype=float)

    widths = np.interp(np.log10(x), np.log10(axis.d_p), axis.delta_log_d_p)
    half = 10 ** (widths / 2)

    return np.asarray(fit.cdf(x * half) - fit.cdf(x / half), dtype=float)


def _secondary_curves(  # pragma: no cover # noqa: PLR0913
    secondary: Literal["cdf", "fit_cdf", "fit_pdf"] | Callable[[FloatArray], FloatArray] | Fit,
    m: Measurement,
    requested: DataType,
    *,
    transforms: Callable[[FloatArray], FloatArray],
    pmf: bool,
    color: str,
    colors: list[str],
    color_offset: int,
    label_spec: str | Callable[[Measurement], str] | None,
    ylabel: str,
) -> tuple[list[_Curve], bool, str]:
    """Computes the secondary lines of one subplot.

    Returns:
        The curves, whether they belong on a twin axis, and the twin axis label.
    """
    axis = m.axis

    if axis is None:
        raise ParticleAnalysisError("Secondary functions require a size distribution.")

    data = transforms(m[requested].copy())
    data_total = float(np.nansum(data))
    x = np.geomspace(axis.d_lower[0], axis.d_upper[-1], FIT_RESOLUTION)

    if isinstance(secondary, Fit):
        if pmf:
            raise NotImplementedError  # peaks normalized per measurement share no scale

        y = _fit_values(secondary, x, axis, requested.normalization)
        label = _resolve_label(label_spec, m, "Outline")

        return [_Curve(x, y, label, {"color": color})], False, ""

    if callable(secondary):
        if pmf:
            data /= data_total

        label = _resolve_label(label_spec, m, "function")

        return [_Curve(m.d_p, secondary(data), label, {"color": color})], True, ylabel

    if secondary == "cdf":
        y = np.cumsum(data / data_total) if data_total > 0 else np.zeros_like(data)
        label = _resolve_label(label_spec, m, "CDF")

        return (
            [_Curve(m.d_p, np.asarray(y, dtype=float), label, {"color": color})],
            True,
            "CDF",
        )

    if secondary in ["fit_cdf", "fit_pdf"]:
        distribution = m.distributions.get(requested.quantity)

        if distribution is None or distribution.distribution_fit is None:
            raise ParticleAnalysisError(
                f"Measurement {m.scan_nr} has no fit. Call fit() before plotting it."
            )

        fit = distribution.distribution_fit
        fit_total = float(fit.cdf(np.inf))  # normalize to the fit, since it extends beyond the measured range

        def evaluate(f: Fit) -> FloatArray:
            if secondary == "fit_cdf":
                return np.asarray(f.cdf(x) / fit_total, dtype=float)

            y = _fit_values(f, x, axis, requested.normalization)
            return y / data_total if pmf else y

        on_twin = secondary == "fit_cdf"
        _ylabel = "CDF" if on_twin else ""
        summed = _Curve(
            x, evaluate(fit), _resolve_label(label_spec, m, "Fit"), {"color": color}
        )

        match fit:
            case ModeLognormalFit():
                return [summed], on_twin, _ylabel
            case MixtureLognormalFit():
                parts = [
                    _Curve(
                        x,
                        evaluate(mode),
                        f"Mode {i + 1}",
                        {"color": colors[(color_offset + i) % len(colors)], "linestyle": "--"},
                    )
                    for i, mode in enumerate(fit.modes)
                ]
                return [summed, *parts], on_twin, _ylabel

    raise ParticleAnalysisError(f"Unknown secondary {secondary!r}.")


def _get_default_title() -> str:  # pragma: no cover
    """Gets the default title."""
    return "Histogram"


def _get_default_xlabel() -> str:  # pragma: no cover
    """Gets the default xlabel."""
    return "Particle Diameter"


def _get_default_ylabel(data_type: DataType, pmf: bool) -> str:  # pragma: no cover
    """Gets the default ylabel."""
    suffix = ", PMF" if pmf else f", in {data_type.base_unit}"
    symbol = data_type.quantity.value

    if data_type.normalization is Normalization.DLOG_DP:
        return f"Normalized Concentration\nΔ{symbol}/ΔlogDₚ{suffix}"

    return f"{data_type.quantity.full_name} Concentration\nΔ{symbol}{suffix}"


def _deduplicate_handles_labels(handles: list, labels: list) -> tuple[list, list]:
    """Deduplicated handles and labels.

    Args:
        handles (list): handles.
        labels (list): labels.

    Returns:
        The secondary functions deduplicated legend input.
    """
    seen = set()
    deduped_handles = []
    deduped_labels = []

    for handle, label in zip(reversed(handles), reversed(labels), strict=True):
        if label not in seen:
            seen.add(label)
            deduped_handles.append(handle)
            deduped_labels.append(label)

    return list(reversed(deduped_handles)), list(reversed(deduped_labels))


def _resolve_label(
    label: str | Callable[[Measurement], str] | None,
    m: Measurement,
    default: str,
) -> str:
    """Resolves the label."""
    if label is None:
        return default

    return label(m) if callable(label) else label
