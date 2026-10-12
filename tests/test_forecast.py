import numpy as np
import pytest

from dca import models
from dca.fitting import FitResult
from dca.forecast import ForecastSettings, estimate_eur, eur_table, monthly_forecast


def make_fit(model="hyperbolic", qi=600.0, di=0.006, b=1.2):
    return FitResult(model=model, qi=qi, di=di, b=b, rmse_log=0.0, r2=1.0, aic=0.0, n_points=10)


def test_exponential_eur_matches_closed_form():
    fit = make_fit("exponential", qi=400, di=0.002, b=0.0)
    s = ForecastSettings(econ_limit=10, horizon_years=100, terminal_decline=None)
    t_last = 365.0
    q_last = 400 * np.exp(-0.002 * t_last)
    e = estimate_eur(fit, t_last, cum_to_date=50_000, settings=s)
    # remaining = (q_last - q_econ) / D
    assert e.remaining == pytest.approx((q_last - 10) / 0.002, rel=1e-9)
    assert e.end_reason == "economic limit"
    assert e.eur == pytest.approx(50_000 + e.remaining)


def test_terminal_decline_makes_b_above_one_finite():
    fit = make_fit(b=1.5)
    no_limit = ForecastSettings(econ_limit=0.0, horizon_years=200, terminal_decline=None)
    with_dmin = ForecastSettings(econ_limit=0.0, horizon_years=200, terminal_decline=0.08)
    e_pure = estimate_eur(fit, 0, 0, no_limit)
    e_mod = estimate_eur(fit, 0, 0, with_dmin)
    assert e_mod.remaining < e_pure.remaining


def test_horizon_caps_forecast():
    fit = make_fit("harmonic", qi=500, di=0.001, b=1.0)
    s = ForecastSettings(econ_limit=0.01, horizon_years=5, terminal_decline=None)
    e = estimate_eur(fit, 100, 0, s)
    assert e.end_reason == "horizon"
    assert e.years_remaining == pytest.approx(5)


def test_monthly_forecast_volumes_sum_to_remaining():
    fit = make_fit()
    s = ForecastSettings(econ_limit=8)
    t_last = 900.0
    e = estimate_eur(fit, t_last, 120_000, s)
    fc = monthly_forecast(fit, "2021-03-01", t_last, 120_000, s)
    # The first partial month (t_last -> next month start) isn't in the table.
    dmin = models.effective_to_nominal(0.08)
    gap = fit.cumulative(fc["t"].iloc[0], dmin) - fit.cumulative(t_last, dmin)
    assert fc["volume"].sum() + gap == pytest.approx(e.remaining, rel=1e-9)
    assert fc["cum"].iloc[-1] == pytest.approx(e.eur, rel=1e-9)
    assert (np.diff(fc["rate"]) < 0).all()


def test_eur_table_has_row_per_model():
    fits = {"exponential": make_fit("exponential", b=0), "hyperbolic": make_fit()}
    tbl = eur_table(fits, 500, 80_000)
    assert list(tbl.index) == ["exponential", "hyperbolic"]
    assert (tbl["eur"] >= 80_000).all()
    assert tbl.loc["hyperbolic", "eur"] > tbl.loc["exponential", "eur"]


def test_flat_fit_is_floored_by_terminal_decline():
    flat = make_fit("exponential", qi=100, di=1e-7, b=0.0)
    e = estimate_eur(flat, 0, 0, ForecastSettings(econ_limit=5, terminal_decline=0.08))
    assert e.end_reason == "economic limit"
    assert e.years_remaining < 40
