import numpy as np
import pytest

from dca import models
from dca.fitting import best_fit, fit_all, fit_model, recency_weights


def make_data(qi, di, b, n=60, noise=0.0, seed=1):
    rng = np.random.default_rng(seed)
    t = np.arange(n) * 30.4
    q = models.hyperbolic_rate(t, qi, di, b) * np.exp(rng.normal(0, noise, n))
    return t, q


@pytest.mark.parametrize("model,b", [("exponential", 0.0), ("harmonic", 1.0)])
def test_recovers_two_parameter_models(model, b):
    t, q = make_data(500, 0.003, b)
    res = fit_model(t, q, model)
    assert res.qi == pytest.approx(500, rel=1e-4)
    assert res.di == pytest.approx(0.003, rel=1e-4)
    assert res.b == b


def test_recovers_hyperbolic_parameters():
    t, q = make_data(1200, 0.01, 1.1)
    res = fit_model(t, q, "hyperbolic")
    assert res.qi == pytest.approx(1200, rel=1e-3)
    assert res.di == pytest.approx(0.01, rel=1e-2)
    assert res.b == pytest.approx(1.1, rel=1e-2)


def test_noisy_hyperbolic_is_selected_over_exponential():
    t, q = make_data(900, 0.008, 0.9, noise=0.06)
    fits = fit_all(t, q)
    assert set(fits) == {"exponential", "hyperbolic", "harmonic"}
    assert fits["hyperbolic"].rmse_log < fits["exponential"].rmse_log
    assert best_fit(fits).model in {"hyperbolic", "harmonic"}


def test_b_bounds_respected():
    t, q = make_data(1000, 0.02, 1.6, noise=0.03)
    res = fit_model(t, q, "hyperbolic", b_bounds=(0.0, 1.0))
    assert 0.0 <= res.b <= 1.0


def test_recency_weights_shape():
    w = recency_weights(np.array([0, 365, 730]), half_life_days=365)
    np.testing.assert_allclose(w, [0.25, 0.5, 1.0])


def test_too_few_points_raises():
    with pytest.raises(ValueError):
        fit_model([0, 30], [100, 90])
