"""Matplotlib plots for decline curve analysis."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .fitting import FitResult  # noqa: E402
from .forecast import ForecastSettings, _dmin, forecast_end_time  # noqa: E402

# Fixed model -> colour mapping so a model looks the same on every chart.
MODEL_COLORS = {
    "exponential": "#2a78d6",
    "hyperbolic": "#eb6834",
    "harmonic": "#1baf7a",
}
DATA_COLOR = "#52514e"
REJECTED_COLOR = "#b9b8b2"
GRID_COLOR = "#e6e5e0"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"


def _style_axes(ax):
    ax.grid(True, which="major", color=GRID_COLOR, linewidth=0.8)
    ax.grid(True, which="minor", color=GRID_COLOR, linewidth=0.4, alpha=0.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(TEXT_SECONDARY)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=9)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))


def _curve(fit: FitResult, settings: ForecastSettings, t_from: float, t_to: float, n=400):
    t = np.linspace(t_from, t_to, n)
    return t, fit.rate(t, _dmin(fit, settings)), fit.cumulative(t, _dmin(fit, settings))


def plot_decline(prepared: pd.DataFrame, fits: dict[str, FitResult], *,
                 settings: ForecastSettings | None = None,
                 best: str | None = None, title: str = "", unit: str = "bbl/d",
                 forecast_years: float = 15.0, path=None):
    """Two-panel figure: rate vs time (semilog) and cumulative vs time.

    Fitted history is drawn solid; the forecast is dashed. Points excluded by
    cleaning (outliers, partial months, pre-peak) are shown hollow in gray.
    """
    settings = settings or ForecastSettings()
    start = prepared.loc[prepared["t"] == 0, "date"].iloc[0]
    to_date = lambda t: start + pd.to_timedelta(t, unit="D")  # noqa: E731

    used = prepared[prepared["used"]]
    rejected = prepared[~prepared["used"] & (prepared["rate"] > 0)]
    t_last = float(prepared["t"].iloc[-1])
    cum_last = float(prepared["cum"].iloc[-1])

    fig, (ax_q, ax_np) = plt.subplots(2, 1, figsize=(10, 8.5), sharex=True,
                                      gridspec_kw={"height_ratios": [3, 2], "hspace": 0.08})

    ax_q.scatter(rejected["date"], rejected["rate"], s=22, facecolors="none",
                 edgecolors=REJECTED_COLOR, linewidths=1, label="excluded", zorder=2)
    ax_q.scatter(used["date"], used["rate"], s=22, color=DATA_COLOR,
                 edgecolors="white", linewidths=0.6, label="monthly rate", zorder=3)
    ax_np.plot(prepared["date"], prepared["cum"], color=DATA_COLOR, lw=2, label="actual")

    for name, fit in fits.items():
        color = MODEL_COLORS.get(name, TEXT_SECONDARY)
        is_best = name == best
        lw = 2.4 if is_best else 1.6
        t_end, _ = forecast_end_time(fit, t_last, settings)
        t_end = min(t_end, t_last + forecast_years * 365.25)

        t_h, q_h, _ = _curve(fit, settings, 0.0, t_last)
        t_f, q_f, np_f = _curve(fit, settings, t_last, t_end)
        np_f = cum_last + (np_f - np_f[0])

        label = f"{name} (b={fit.b:.2f})" + ("  ← best" if is_best else "")
        ax_q.plot(to_date(t_h), q_h, color=color, lw=lw, label=label, zorder=4)
        ax_q.plot(to_date(t_f), q_f, color=color, lw=lw, ls="--", zorder=4)
        ax_np.plot(to_date(t_f), np_f, color=color, lw=lw, ls="--", label=name)
        if is_best:
            ax_np.annotate(f"{np_f[-1] / 1e3:,.0f}k", (to_date(t_f[-1:])[0], np_f[-1]),
                           xytext=(4, 0), textcoords="offset points", va="center",
                           fontsize=8, color=TEXT_SECONDARY)

    ax_q.axvline(prepared["date"].iloc[-1], color=TEXT_SECONDARY, lw=0.8, ls=":")
    ax_q.text(prepared["date"].iloc[-1], 1.0, " forecast →", transform=ax_q.get_xaxis_transform(),
              va="top", fontsize=8, color=TEXT_SECONDARY)
    if settings.econ_limit > 0:
        ax_q.axhline(settings.econ_limit, color=TEXT_SECONDARY, lw=0.8, ls=(0, (1, 3)))
        ax_q.text(prepared["date"].iloc[0], settings.econ_limit, " economic limit",
                  va="bottom", fontsize=8, color=TEXT_SECONDARY)

    ax_q.set_yscale("log")
    ax_q.set_ylabel(f"Rate ({unit})", color=TEXT_PRIMARY)
    ax_q.legend(frameon=False, fontsize=9, loc="upper right")
    ax_np.set_ylabel(f"Cumulative ({unit.split('/')[0]})", color=TEXT_PRIMARY)
    ax_np.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v / 1e3:,.0f}k"))
    ax_np.legend(frameon=False, fontsize=9, loc="lower right")
    for ax in (ax_q, ax_np):
        _style_axes(ax)

    if title:
        fig.suptitle(title, x=0.125, ha="left", fontsize=13, color=TEXT_PRIMARY, y=0.95)
    if path:
        fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig)
    return fig


def plot_residuals(prepared: pd.DataFrame, fits: dict[str, FitResult], path=None):
    """ln(actual / fitted) vs time for each model - patterns mean a bad model."""
    used = prepared[prepared["used"]]
    t, q = used["t"].to_numpy(float), used["rate"].to_numpy(float)
    fig, axes = plt.subplots(len(fits), 1, figsize=(10, 2.2 * len(fits)), sharex=True,
                             sharey=True, squeeze=False)
    for ax, (name, fit) in zip(axes[:, 0], fits.items()):
        resid = np.log(q / fit.rate(t))
        ax.axhline(0, color=TEXT_SECONDARY, lw=0.8)
        ax.vlines(used["date"], 0, resid, color=MODEL_COLORS.get(name, DATA_COLOR), lw=2)
        ax.set_ylabel("ln(q/q̂)", fontsize=9, color=TEXT_PRIMARY)
        ax.set_title(f"{name}  RMSE={fit.rmse_log:.3f}", loc="left", fontsize=10,
                     color=TEXT_PRIMARY)
        _style_axes(ax)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig)
    return fig


def plot_eur_distribution(summary: pd.DataFrame, column: str = "eur", path=None, unit="bbl"):
    """Histogram of EUR across a batch of wells, with P10/P50/P90 marked."""
    vals = summary[column].dropna().to_numpy() / 1e3
    # Clip the axis at the 99th percentile so one monster well doesn't squash
    # the distribution; clipped wells are counted in a note instead.
    p99 = np.percentile(vals, 99)
    x_max = vals.max() if vals.max() <= 1.5 * p99 else p99 * 1.15
    shown = vals[vals <= x_max]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.set_axisbelow(True)
    counts, _, _ = ax.hist(shown, bins=min(30, max(5, len(shown) // 10)),
                           color=MODEL_COLORS["exponential"], edgecolor="white", linewidth=1.5)
    ax.set_ylim(0, counts.max() * 1.3)  # headroom for the percentile labels
    # Petroleum convention: P90 is the low case (90% chance of exceeding).
    for i, (label, pct) in enumerate((("P90", 10), ("P50", 50), ("P10", 90))):
        v = np.percentile(vals, pct)
        ax.axvline(v, color=TEXT_PRIMARY, lw=1, ls="--")
        ax.text(v, 0.99 - 0.07 * (i % 2), f" {label} {v:,.0f}k",
                transform=ax.get_xaxis_transform(), va="top", fontsize=8, color=TEXT_PRIMARY)
    if len(shown) < len(vals):
        ax.text(1.0, 0.6, f"{len(vals) - len(shown)} well(s) > {x_max:,.0f}k not shown",
                transform=ax.transAxes, ha="right", fontsize=8, color=TEXT_SECONDARY)
    ax.set_title(f"EUR distribution, {len(vals)} wells", loc="left", fontsize=11, color=TEXT_PRIMARY)
    ax.set_xlabel(f"EUR (thousand {unit})", color=TEXT_PRIMARY)
    ax.set_ylabel("Wells", color=TEXT_PRIMARY)
    ax.grid(True, axis="y", color=GRID_COLOR)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig)
    return fig
