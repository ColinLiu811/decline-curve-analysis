"""Fit Arps decline models to rate-time data with scipy.

Fits are done on log(rate). Production data noise is roughly multiplicative,
and fitting in linear space lets the high early rates dominate the objective
so the tail - which drives EUR - gets ignored.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import curve_fit

from . import models

MODEL_NAMES = ("exponential", "hyperbolic", "harmonic")


@dataclass
class FitResult:
    model: str
    qi: float
    di: float          # nominal initial decline, 1/day
    b: float
    rmse_log: float    # RMSE of ln(q) residuals
    r2: float          # R^2 on ln(q)
    aic: float
    n_points: int
    param_std: dict = field(default_factory=dict)

    @property
    def di_annual_nominal(self) -> float:
        return self.di * models.DAYS_PER_YEAR

    @property
    def di_annual_effective(self) -> float:
        return float(models.nominal_to_effective(self.di, self.b))

    def rate(self, t, dmin: float | None = None):
        if dmin is not None:
            return models.modified_hyperbolic_rate(t, self.qi, self.di, self.b, dmin)
        return models.hyperbolic_rate(t, self.qi, self.di, self.b)

    def cumulative(self, t, dmin: float | None = None):
        if dmin is not None:
            return models.modified_hyperbolic_cumulative(t, self.qi, self.di, self.b, dmin)
        return models.hyperbolic_cumulative(t, self.qi, self.di, self.b)

    def summary(self) -> str:
        return (f"{self.model:<11} qi={self.qi:8.1f}/d  "
                f"Di={self.di_annual_effective:6.1%}/yr eff  b={self.b:4.2f}  "
                f"RMSE(ln q)={self.rmse_log:.3f}  R2={self.r2:.3f}  AIC={self.aic:7.1f}")


def _initial_guess(t: np.ndarray, q: np.ndarray) -> tuple[float, float]:
    """qi and di from a straight-line fit of ln(q) vs t (an exponential)."""
    slope, intercept = np.polyfit(t, np.log(q), 1)
    di = max(-slope, 1e-5)
    qi = float(np.exp(intercept))
    return max(qi, q.max()), di


def _log_model(model: str):
    if model == "exponential":
        return lambda t, qi, di: np.log(models.exponential_rate(t, qi, di))
    if model == "harmonic":
        return lambda t, qi, di: np.log(models.harmonic_rate(t, qi, di))
    if model == "hyperbolic":
        return lambda t, qi, di, b: np.log(models.hyperbolic_rate(t, qi, di, b))
    raise ValueError(f"unknown model {model!r}; expected one of {MODEL_NAMES}")


def fit_model(t, q, model: str = "hyperbolic", *, b_bounds=(0.0, 2.0),
              weights=None) -> FitResult:
    """Least-squares fit of a single Arps model.

    Parameters
    ----------
    t, q:
        Elapsed days since decline start, and rates (> 0).
    model:
        "exponential", "hyperbolic" or "harmonic".
    b_bounds:
        Allowed range of b for the hyperbolic. Values above 1 are physically
        transient flow (common in tight/shale wells); cap at 1 for a
        conventional, boundary-dominated interpretation.
    weights:
        Optional per-point weights (e.g. emphasise recent months).
    """
    t = np.asarray(t, dtype=float)
    q = np.asarray(q, dtype=float)
    mask = np.isfinite(t) & np.isfinite(q) & (q > 0)
    t, q = t[mask], q[mask]
    if len(t) < 3:
        raise ValueError("need at least 3 positive rate points to fit")

    qi0, di0 = _initial_guess(t, q)
    qmax = q.max()
    qi_bounds = (0.5 * qmax, 5.0 * qmax)
    di_bounds = (1e-6, 1.0)

    f = _log_model(model)
    if model == "hyperbolic":
        b0 = float(np.clip(0.8, *b_bounds))
        # A hyperbolic's early decline is steeper than the average exponential.
        p0 = [np.clip(qi0, *qi_bounds), np.clip(di0 * 2, *di_bounds), b0]
        bounds = ([qi_bounds[0], di_bounds[0], b_bounds[0]],
                  [qi_bounds[1], di_bounds[1], b_bounds[1]])
        names = ("qi", "di", "b")
    else:
        p0 = [np.clip(qi0, *qi_bounds), np.clip(di0, *di_bounds)]
        bounds = ([qi_bounds[0], di_bounds[0]], [qi_bounds[1], di_bounds[1]])
        names = ("qi", "di")

    sigma = None if weights is None else 1.0 / np.sqrt(np.asarray(weights, float)[mask])
    popt, pcov = curve_fit(f, t, np.log(q), p0=p0, bounds=bounds, sigma=sigma,
                           x_scale=np.abs(p0), max_nfev=20000)

    resid = np.log(q) - f(t, *popt)
    n, k = len(t), len(popt)
    sse = float(np.sum(resid**2))
    sst = float(np.sum((np.log(q) - np.log(q).mean()) ** 2))
    with np.errstate(invalid="ignore"):
        perr = np.sqrt(np.diag(pcov))

    b = {"exponential": 0.0, "harmonic": 1.0}.get(model, popt[2] if k == 3 else 0.0)
    return FitResult(
        model=model,
        qi=float(popt[0]),
        di=float(popt[1]),
        b=float(b),
        rmse_log=float(np.sqrt(sse / n)),
        r2=1.0 - sse / sst if sst > 0 else float("nan"),
        aic=n * np.log(max(sse, 1e-300) / n) + 2 * k,
        n_points=n,
        param_std=dict(zip(names, map(float, perr))),
    )


def fit_all(t, q, models_to_fit=MODEL_NAMES, **kwargs) -> dict[str, FitResult]:
    """Fit every requested model; models that fail to converge are skipped."""
    results = {}
    for name in models_to_fit:
        try:
            results[name] = fit_model(t, q, name, **kwargs)
        except (RuntimeError, ValueError) as exc:
            print(f"warning: {name} fit failed: {exc}")
    return results


def best_fit(results: dict[str, FitResult]) -> FitResult:
    """Pick the model with the lowest AIC (penalises the extra b parameter)."""
    if not results:
        raise ValueError("no successful fits")
    return min(results.values(), key=lambda r: r.aic)


def recency_weights(t, half_life_days: float = 365.0) -> np.ndarray:
    """Exponentially increasing weights so recent behaviour counts more."""
    t = np.asarray(t, dtype=float)
    return np.power(2.0, (t - t.max()) / half_life_days)
