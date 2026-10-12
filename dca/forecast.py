"""Forecast future production and estimate EUR from a fitted decline.

EUR (estimated ultimate recovery) is split into the volume already produced
and the remaining reserves forecast from the last data point until the well
either hits its economic limit or the forecast horizon runs out:

    EUR = Np(actual, to date) + [Np_model(t_end) - Np_model(t_last)]

Anchoring the remaining volume on the *model* cumulative between t_last and
t_end (rather than the model's cumulative from t = 0) means early-time misfit
doesn't leak into the reserve estimate.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import models
from .fitting import FitResult

# Effective annual terminal decline, typical for shale wells. Also used as a
# floor on decline for every model so a flat fit can't forecast forever.
DEFAULT_TERMINAL_DECLINE = 0.08
DEFAULT_ECON_LIMIT = 5.0          # volume/day
DEFAULT_HORIZON_YEARS = 50.0


@dataclass
class ForecastSettings:
    econ_limit: float = DEFAULT_ECON_LIMIT
    horizon_years: float = DEFAULT_HORIZON_YEARS
    terminal_decline: float | None = DEFAULT_TERMINAL_DECLINE  # effective/yr; None disables


@dataclass
class EURResult:
    model: str
    cum_to_date: float
    remaining: float
    eur: float
    t_last: float           # days since decline start
    t_end: float            # days since decline start when forecast stops
    end_reason: str         # "economic limit" or "horizon"
    rate_at_last: float

    @property
    def years_remaining(self) -> float:
        return (self.t_end - self.t_last) / models.DAYS_PER_YEAR


def _dmin(fit: FitResult, settings: ForecastSettings) -> float | None:
    if settings.terminal_decline is None:
        return None
    return float(models.effective_to_nominal(settings.terminal_decline))


def forecast_end_time(fit: FitResult, t_last: float, settings: ForecastSettings) -> tuple[float, str]:
    """When the forecast stops: economic limit or horizon, whichever is first."""
    dmin = _dmin(fit, settings)
    horizon = t_last + settings.horizon_years * models.DAYS_PER_YEAR
    t_econ = models.time_to_rate(settings.econ_limit, fit.qi, fit.di, fit.b, dmin)
    if t_econ <= t_last:
        return t_last, "economic limit"
    if t_econ < horizon:
        return float(t_econ), "economic limit"
    return float(horizon), "horizon"


def estimate_eur(fit: FitResult, t_last: float, cum_to_date: float,
                 settings: ForecastSettings | None = None) -> EURResult:
    settings = settings or ForecastSettings()
    dmin = _dmin(fit, settings)
    t_end, reason = forecast_end_time(fit, t_last, settings)
    remaining = float(fit.cumulative(t_end, dmin) - fit.cumulative(t_last, dmin))
    return EURResult(
        model=fit.model,
        cum_to_date=float(cum_to_date),
        remaining=max(remaining, 0.0),
        eur=float(cum_to_date) + max(remaining, 0.0),
        t_last=float(t_last),
        t_end=t_end,
        end_reason=reason,
        rate_at_last=float(fit.rate(t_last, dmin)),
    )


def monthly_forecast(fit: FitResult, start_date, t_last: float, cum_to_date: float,
                     settings: ForecastSettings | None = None) -> pd.DataFrame:
    """Month-by-month forecast table from the last actual month onward.

    ``start_date`` is the calendar date corresponding to t = 0 (decline start).
    Monthly volumes are exact integrals of the decline curve over each month.
    """
    settings = settings or ForecastSettings()
    dmin = _dmin(fit, settings)
    t_end, _ = forecast_end_time(fit, t_last, settings)

    start_date = pd.Timestamp(start_date)
    last_month = start_date + pd.Timedelta(days=t_last)
    end_date = start_date + pd.Timedelta(days=t_end)
    edges = pd.date_range(last_month + pd.offsets.MonthBegin(1), end_date + pd.offsets.MonthBegin(1),
                          freq="MS")
    if len(edges) < 2:
        return pd.DataFrame(columns=["date", "t", "rate", "volume", "cum"])

    t_edges = np.minimum((edges - start_date).days.to_numpy(float), t_end)
    cum_model = fit.cumulative(t_edges, dmin)
    volume = np.diff(cum_model)
    days = np.diff(t_edges)
    with np.errstate(divide="ignore", invalid="ignore"):
        rate = np.where(days > 0, volume / days, 0.0)

    out = pd.DataFrame({
        "date": edges[:-1],
        "t": t_edges[:-1],
        "rate": rate,
        "volume": volume,
    })
    out = out[out["volume"] > 0].reset_index(drop=True)
    out["cum"] = cum_to_date + (fit.cumulative(t_edges[0], dmin) - fit.cumulative(t_last, dmin)) \
        + out["volume"].cumsum()
    return out


def eur_table(fits: dict[str, FitResult], t_last: float, cum_to_date: float,
              settings: ForecastSettings | None = None) -> pd.DataFrame:
    """Side-by-side EUR comparison across fitted models."""
    rows = []
    for name, fit in fits.items():
        e = estimate_eur(fit, t_last, cum_to_date, settings)
        rows.append({
            "model": name,
            "qi": fit.qi,
            "di_eff_annual": fit.di_annual_effective,
            "b": fit.b,
            "rmse_log": fit.rmse_log,
            "aic": fit.aic,
            "cum_to_date": e.cum_to_date,
            "remaining": e.remaining,
            "eur": e.eur,
            "years_remaining": e.years_remaining,
            "end_reason": e.end_reason,
        })
    return pd.DataFrame(rows).set_index("model")
