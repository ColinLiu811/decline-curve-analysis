import numpy as np
import pandas as pd
import pytest

from dca.cleaning import (CleaningConfig, find_restarts, flag_outliers,
                          prepare_decline_data, to_daily_rates)
from dca.models import hyperbolic_rate


def synthetic_well(n=48, qi=900.0, di=0.006, b=0.9, ramp=2, noise=0.05, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-01", periods=n, freq="MS")
    t = np.arange(n) * 30.4
    q = hyperbolic_rate(np.maximum(t - ramp * 30.4, 0), qi, di, b)
    q[:ramp] = qi * np.linspace(0.3, 0.8, ramp)
    q *= np.exp(rng.normal(0, noise, n))
    days = np.full(n, 30)
    return pd.DataFrame({"date": dates, "oil_bbl": q * days, "days_on": days})


def test_to_daily_rates_divides_by_producing_days():
    df = pd.DataFrame({"date": pd.date_range("2021-01-01", periods=3, freq="MS"),
                       "oil_bbl": [3000, 0, 1500], "days_on": [30, 0, 15]})
    out = to_daily_rates(df)
    assert out["rate"].tolist() == [100.0, 0.0, 100.0]
    assert out["cum"].tolist() == [3000, 3000, 4500]


def test_flag_outliers_catches_spike():
    q = pd.Series(np.geomspace(500, 200, 30))
    q.iloc[12] *= 6
    flags = flag_outliers(q)
    assert flags.iloc[12]
    assert flags.sum() == 1


def test_find_restarts_detects_refrac():
    q = pd.Series(np.r_[np.geomspace(800, 100, 24), np.geomspace(600, 300, 12)])
    assert find_restarts(q) == [24]


def test_prepare_starts_at_peak_and_drops_shutins():
    df = synthetic_well(noise=0.01)
    df.loc[20, ["oil_bbl", "days_on"]] = 0          # shut-in month
    df.loc[30, "days_on"] = 3                         # partial month
    prepared = prepare_decline_data(df)
    used = prepared[prepared["used"]]
    assert used["t"].iloc[0] == 0
    assert used.index[0] == 2                         # peak right after the ramp
    assert not prepared.loc[20, "used"] and not prepared.loc[30, "used"]
    # cumulative still includes every reported barrel
    assert prepared["cum"].iloc[-1] == pytest.approx(df["oil_bbl"].sum())


def test_prepare_uses_last_segment_after_refrac():
    df = synthetic_well(n=60)
    df.loc[40:, "oil_bbl"] *= 5
    prepared = prepare_decline_data(df, CleaningConfig(use_last_segment=True))
    assert prepared[prepared["used"]]["date"].iloc[0] >= df["date"].iloc[40]
