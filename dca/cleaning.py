"""Turn raw monthly production volumes into a decline-ready rate series.

Monthly reports give *volumes* per calendar month, plus the number of days the
well actually produced. Real data is messy: wells are shut in, choked back,
worked over, or report partial months. The steps here are the usual DCA prep:

1. convert monthly volumes to producing-day rates (volume / days_on)
2. drop shut-in and partial months that would bias the rate
3. reject outliers relative to a rolling median in log-rate space
4. start the decline at the peak rate (the build-up before peak is not decline)
5. optionally keep only the last segment after a refrac/restart
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

AVG_DAYS_PER_MONTH = 365.25 / 12.0

PHASE_COLUMNS = {"oil": "oil_bbl", "gas": "gas_mcf", "water": "water_bbl"}


@dataclass
class CleaningConfig:
    phase: str = "oil"
    min_days_on: int = 10          # months with fewer producing days are dropped
    outlier_window: int = 5        # rolling-median window (months)
    outlier_threshold: float = 3.5  # robust z-score cutoff in log-rate space
    peak_search_months: int = 12   # look for the peak within this many months of first prod
    restart_ratio: float = 2.5     # rate jump (vs trailing median) treated as a restart
    min_segment_months: int = 12   # a post-restart segment needs this much history to be used
    use_last_segment: bool = True


def to_daily_rates(df: pd.DataFrame, phase: str = "oil") -> pd.DataFrame:
    """Add ``rate`` (volume per producing day) and running cumulative columns.

    Expects columns ``date``, ``days_on`` and the phase volume column. The
    cumulative is computed on *all* reported volumes before anything is
    filtered, so it stays the true produced-to-date volume.
    """
    vol_col = PHASE_COLUMNS[phase]
    out = df.sort_values("date").copy()
    out["date"] = pd.to_datetime(out["date"])
    out[vol_col] = pd.to_numeric(out[vol_col], errors="coerce").fillna(0.0).clip(lower=0)
    out["days_on"] = pd.to_numeric(out["days_on"], errors="coerce").fillna(0).clip(0, 31)
    out["volume"] = out[vol_col]
    out["cum"] = out["volume"].cumsum()
    with np.errstate(divide="ignore", invalid="ignore"):
        out["rate"] = np.where(out["days_on"] > 0, out["volume"] / out["days_on"], 0.0)
    return out.reset_index(drop=True)


def flag_outliers(rate: pd.Series, window: int = 5, threshold: float = 3.5,
                  min_spread: float = 0.05) -> pd.Series:
    """Boolean mask of points far from a centred rolling median (log space).

    Uses a robust z-score (median absolute deviation) so a few bad months
    don't inflate the spread estimate. ``min_spread`` floors the MAD (in log
    units, ~5%) so very smooth series don't flag ordinary month-to-month noise.
    """
    log_q = np.log(rate.astype(float))
    med = log_q.rolling(window, center=True, min_periods=1).median()
    resid = log_q - med
    mad = max(np.median(np.abs(resid - np.median(resid))), min_spread)
    z = 0.6745 * resid / mad
    return z.abs() > threshold


def find_peak(df: pd.DataFrame, search_months: int = 12) -> int:
    """Positional index of the peak-rate month within the early life of the well."""
    head = df.iloc[:max(search_months, 1)]
    return int(np.argmax(head["rate"].to_numpy()))


def find_restarts(rate: pd.Series, ratio: float = 2.5, lookback: int = 4) -> list[int]:
    """Positions where rate jumps well above its recent trend (refrac, workover).

    Requires the jump to persist for at least two months so a single noisy
    month isn't mistaken for a restart.
    """
    q = rate.to_numpy(dtype=float)
    restarts = []
    i = lookback
    while i < len(q) - 1:
        trailing = np.median(q[i - lookback:i])
        if trailing > 0 and q[i] > ratio * trailing and q[i + 1] > ratio * trailing:
            restarts.append(i)
            i += lookback  # don't re-detect the same event
        else:
            i += 1
    return restarts


def prepare_decline_data(df: pd.DataFrame, config: CleaningConfig | None = None) -> pd.DataFrame:
    """Full cleaning pipeline for a single well.

    Returns a frame with ``date``, ``t`` (days since decline start), ``rate``,
    ``cum`` and a boolean ``used`` column marking points kept for fitting.
    Points that were rejected are retained (``used == False``) so they can be
    shown on plots.
    """
    cfg = config or CleaningConfig()
    data = to_daily_rates(df, cfg.phase)

    valid = (data["days_on"] >= cfg.min_days_on) & (data["rate"] > 0)
    work = data[valid].copy()
    if len(work) < 3:
        raise ValueError(f"only {len(work)} usable months after removing shut-ins")

    work["outlier"] = flag_outliers(work["rate"], cfg.outlier_window, cfg.outlier_threshold)

    clean = work[~work["outlier"]]
    start_pos = find_peak(clean, cfg.peak_search_months)
    if cfg.use_last_segment:
        after_peak = clean.iloc[start_pos:]
        restarts = [r for r in find_restarts(after_peak["rate"], cfg.restart_ratio)
                    if len(after_peak) - r >= cfg.min_segment_months]
        if restarts:
            seg = after_peak.iloc[restarts[-1]:]
            start_pos = start_pos + restarts[-1] + find_peak(seg, 3)
    start_date = clean.iloc[start_pos]["date"]

    data = data.merge(work[["date", "outlier"]], on="date", how="left")
    data["outlier"] = data["outlier"].fillna(False).astype(bool)
    data["used"] = valid & ~data["outlier"] & (data["date"] >= start_date)
    data["t"] = (data["date"] - start_date).dt.days.astype(float)
    return data


def decline_arrays(prepared: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """(t, q) arrays of the points flagged for fitting."""
    used = prepared[prepared["used"]]
    return used["t"].to_numpy(float), used["rate"].to_numpy(float)
