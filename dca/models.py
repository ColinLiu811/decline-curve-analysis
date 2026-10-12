"""Arps decline curve equations.

Conventions used throughout the package:

* ``t``  - elapsed time in **days** since the start of decline
* ``qi`` - initial rate at t = 0 (volume / day, e.g. bbl/d)
* ``di`` - initial *nominal* decline rate in **1/day**
* ``b``  - hyperbolic exponent (0 = exponential, 1 = harmonic)

With rates in volume/day and time in days, the cumulative functions return
volumes directly (e.g. bbl).
"""

from __future__ import annotations

import numpy as np

DAYS_PER_YEAR = 365.25

# Below this |b| we treat the hyperbolic as exponential to avoid 1/b blow-ups.
_B_EPS = 1e-6


# ---------------------------------------------------------------------------
# Rate equations
# ---------------------------------------------------------------------------

def exponential_rate(t, qi, di):
    """q(t) = qi * exp(-di * t)"""
    t = np.asarray(t, dtype=float)
    return qi * np.exp(-di * t)


def harmonic_rate(t, qi, di):
    """q(t) = qi / (1 + di * t)"""
    t = np.asarray(t, dtype=float)
    return qi / (1.0 + di * t)


def hyperbolic_rate(t, qi, di, b):
    """q(t) = qi / (1 + b * di * t) ** (1 / b)"""
    t = np.asarray(t, dtype=float)
    if abs(b) < _B_EPS:
        return exponential_rate(t, qi, di)
    return qi / np.power(1.0 + b * di * t, 1.0 / b)


# ---------------------------------------------------------------------------
# Cumulative production
# ---------------------------------------------------------------------------

def exponential_cumulative(t, qi, di):
    """Np(t) = (qi - q) / di"""
    q = exponential_rate(t, qi, di)
    return (qi - q) / di


def harmonic_cumulative(t, qi, di):
    """Np(t) = qi / di * ln(1 + di * t)"""
    t = np.asarray(t, dtype=float)
    return qi / di * np.log1p(di * t)


def hyperbolic_cumulative(t, qi, di, b):
    """Np(t) = qi^b / ((1 - b) di) * (qi^(1-b) - q^(1-b))"""
    if abs(b) < _B_EPS:
        return exponential_cumulative(t, qi, di)
    if abs(b - 1.0) < _B_EPS:
        return harmonic_cumulative(t, qi, di)
    q = hyperbolic_rate(t, qi, di, b)
    return qi**b / ((1.0 - b) * di) * (qi ** (1.0 - b) - q ** (1.0 - b))


# ---------------------------------------------------------------------------
# Modified hyperbolic (hyperbolic -> exponential at a terminal decline)
# ---------------------------------------------------------------------------

def instantaneous_decline(t, di, b):
    """Nominal decline D(t) = di / (1 + b * di * t) for an Arps hyperbolic."""
    t = np.asarray(t, dtype=float)
    return di / (1.0 + b * di * t)


def _below_floor(di, dmin) -> bool:
    return dmin is not None and dmin > 0 and di <= dmin


def switch_time(di, b, dmin):
    """Time (days) at which a hyperbolic's decline falls to ``dmin``.

    Returns ``inf`` when the curve never reaches ``dmin`` (b == 0, or the
    initial decline is already below the terminal decline).
    """
    if b <= _B_EPS or dmin <= 0 or di <= dmin:
        return np.inf
    return (di / dmin - 1.0) / (b * di)


def modified_hyperbolic_rate(t, qi, di, b, dmin):
    """Hyperbolic decline that switches to exponential once D(t) = dmin.

    This is the standard way to keep EUR finite for b >= 1 wells (common in
    shale), where a pure hyperbolic would produce forever. ``dmin`` also acts
    as a floor: a curve whose initial decline is already below ``dmin`` just
    declines exponentially at ``dmin``.
    """
    t = np.asarray(t, dtype=float)
    if _below_floor(di, dmin):
        return exponential_rate(t, qi, dmin)
    ts = switch_time(di, b, dmin)
    if not np.isfinite(ts):
        return hyperbolic_rate(t, qi, di, b)
    qs = hyperbolic_rate(ts, qi, di, b)
    return np.where(t <= ts,
                    hyperbolic_rate(np.minimum(t, ts), qi, di, b),
                    qs * np.exp(-dmin * np.maximum(t - ts, 0.0)))


def modified_hyperbolic_cumulative(t, qi, di, b, dmin):
    """Cumulative production for :func:`modified_hyperbolic_rate`."""
    t = np.asarray(t, dtype=float)
    if _below_floor(di, dmin):
        return exponential_cumulative(t, qi, dmin)
    ts = switch_time(di, b, dmin)
    if not np.isfinite(ts):
        return hyperbolic_cumulative(t, qi, di, b)
    qs = hyperbolic_rate(ts, qi, di, b)
    np_s = hyperbolic_cumulative(ts, qi, di, b)
    early = hyperbolic_cumulative(np.minimum(t, ts), qi, di, b)
    late = np_s + exponential_cumulative(np.maximum(t - ts, 0.0), qs, dmin)
    return np.where(t <= ts, early, late)


# ---------------------------------------------------------------------------
# Decline-rate conversions
# ---------------------------------------------------------------------------

def nominal_to_effective(d_nominal, b=0.0, period=DAYS_PER_YEAR):
    """Convert a nominal decline (1/day) to an effective decline over ``period``.

    Uses the tangent-effective definition: the fractional drop in rate over
    one period, starting from the point where the nominal decline applies.
    """
    dt = d_nominal * period
    if abs(b) < _B_EPS:
        return 1.0 - np.exp(-dt)
    return 1.0 - (1.0 + b * dt) ** (-1.0 / b)


def effective_to_nominal(d_effective, b=0.0, period=DAYS_PER_YEAR):
    """Inverse of :func:`nominal_to_effective` (returns 1/day)."""
    if not 0.0 <= d_effective < 1.0:
        raise ValueError("effective decline must be in [0, 1)")
    if abs(b) < _B_EPS:
        return -np.log(1.0 - d_effective) / period
    return ((1.0 - d_effective) ** (-b) - 1.0) / (b * period)


def time_to_rate(q_target, qi, di, b, dmin=None):
    """Time (days) for a decline curve to fall to ``q_target``."""
    if q_target >= qi:
        return 0.0
    if q_target <= 0:
        return np.inf
    if _below_floor(di, dmin):
        return np.log(qi / q_target) / dmin
    if dmin is not None:
        ts = switch_time(di, b, dmin)
        if np.isfinite(ts):
            qs = float(hyperbolic_rate(ts, qi, di, b))
            if q_target < qs:
                return ts + np.log(qs / q_target) / dmin
    if abs(b) < _B_EPS:
        return np.log(qi / q_target) / di
    return ((qi / q_target) ** b - 1.0) / (b * di)
