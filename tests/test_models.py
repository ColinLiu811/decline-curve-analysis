import numpy as np
import pytest
from scipy.integrate import quad

from dca import models as m


QI, DI = 800.0, 0.004


@pytest.mark.parametrize("b", [0.0, 0.5, 1.0, 1.4])
def test_cumulative_matches_numerical_integral(b):
    t_end = 3000.0
    numeric, _ = quad(lambda t: m.hyperbolic_rate(t, QI, DI, b), 0, t_end, limit=200)
    assert m.hyperbolic_cumulative(t_end, QI, DI, b) == pytest.approx(numeric, rel=1e-6)


def test_hyperbolic_limits_reduce_to_exponential_and_harmonic():
    t = np.linspace(0, 2000, 50)
    np.testing.assert_allclose(m.hyperbolic_rate(t, QI, DI, 1e-9), m.exponential_rate(t, QI, DI))
    np.testing.assert_allclose(m.hyperbolic_rate(t, QI, DI, 1.0), m.harmonic_rate(t, QI, DI))


def test_modified_hyperbolic_is_continuous_at_switch():
    b, dmin = 1.2, m.effective_to_nominal(0.08)
    ts = m.switch_time(DI, b, dmin)
    assert np.isfinite(ts)
    left = m.modified_hyperbolic_rate(ts - 1e-6, QI, DI, b, dmin)
    right = m.modified_hyperbolic_rate(ts + 1e-6, QI, DI, b, dmin)
    assert left == pytest.approx(right, rel=1e-6)
    # Decline after the switch is exactly dmin
    q1 = m.modified_hyperbolic_rate(ts + 100, QI, DI, b, dmin)
    q2 = m.modified_hyperbolic_rate(ts + 101, QI, DI, b, dmin)
    assert np.log(q1 / q2) == pytest.approx(dmin, rel=1e-6)


def test_modified_hyperbolic_cumulative_matches_integral():
    b, dmin = 1.3, m.effective_to_nominal(0.06)
    t_end = 20000.0
    ts = m.switch_time(DI, b, dmin)
    f = lambda t: m.modified_hyperbolic_rate(t, QI, DI, b, dmin)
    numeric = quad(f, 0, ts, limit=200)[0] + quad(f, ts, t_end, limit=200)[0]
    assert m.modified_hyperbolic_cumulative(t_end, QI, DI, b, dmin) == pytest.approx(numeric, rel=1e-6)


@pytest.mark.parametrize("b", [0.0, 0.8])
def test_effective_nominal_round_trip(b):
    for de in (0.05, 0.3, 0.75):
        dn = m.effective_to_nominal(de, b)
        assert m.nominal_to_effective(dn, b) == pytest.approx(de)


@pytest.mark.parametrize("b,dmin", [(0.0, None), (0.7, None), (1.5, 1e-4)])
def test_time_to_rate_inverts_rate(b, dmin):
    target = 15.0
    t = m.time_to_rate(target, QI, DI, b, dmin)
    q = (m.modified_hyperbolic_rate(t, QI, DI, b, dmin) if dmin
         else m.hyperbolic_rate(t, QI, DI, b))
    assert q == pytest.approx(target, rel=1e-6)


def test_dmin_acts_as_decline_floor():
    dmin = m.effective_to_nominal(0.08)
    t = np.array([0.0, 365.25, 730.5])
    for b in (0.0, 1.0):
        q = m.modified_hyperbolic_rate(t, QI, dmin / 10, b, dmin)
        np.testing.assert_allclose(q, m.exponential_rate(t, QI, dmin))
    assert m.time_to_rate(QI / 2, QI, dmin / 10, 0.0, dmin) == pytest.approx(np.log(2) / dmin)
