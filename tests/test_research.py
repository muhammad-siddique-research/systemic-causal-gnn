"""Model guarantees and disclosure boundaries on synthetic networks."""
from __future__ import annotations
import json
import numpy as np
import pytest
from src.research.core import (
    clear_network, draw_network, paired_bootstrap, simulate_policy
)
from src.research.dml import synthetic_plr, crossfit_plr
from src.research.prediction import (
    create_node_level_dataset, evaluate_out_of_network_prediction
)


def test_zero_obligations_and_nonnegativity():
    result = clear_network(np.zeros((4, 4)), np.array([1., 0., 1., 0.]))
    assert result.shortfall == 0
    assert not result.defaults.any()
    assert np.array_equal(result.payments, np.zeros(4))


def test_financial_clearing_fixed_point():
    rng = np.random.default_rng(8)
    l, a, _ = draw_network(rng, n=15)
    result = clear_network(l, a)
    total = l.sum(axis=1)
    pi = np.divide(
        l, total[:, None], out=np.zeros_like(l),
        where=total[:, None] > 0
    )
    expected = np.minimum(total, np.maximum(0, a + pi.T @ result.payments))
    assert np.max(np.abs(result.payments - expected)) < 1e-8
    assert np.all(result.payments <= total + 1e-8)


def test_larger_external_assets_do_not_increase_shortfall():
    rng = np.random.default_rng(16)
    l, a, _ = draw_network(rng, n=20)
    baseline = clear_network(l, a)
    extra = np.zeros_like(a)
    extra[0] += 75
    improved = clear_network(l, a+extra)
    assert improved.shortfall <= baseline.shortfall + 1e-7
    assert np.all(improved.payments >= baseline.payments - 1e-7)


@pytest.mark.parametrize("bad", [-1., float("nan")])
def test_invalid_obligations_rejected(bad):
    l = np.zeros((5, 5))
    l[1, 2] = bad
    with pytest.raises(ValueError):
        clear_network(l, np.ones(5))


def test_policy_scenario_reproduction_and_equal_budget():
    a = simulate_policy(n_networks=12, seed=12, n=10)
    b = simulate_policy(n_networks=12, seed=12, n=10)
    assert a.equals(b)
    assert len(a) == 12
    assert (a.targeted_reduction >= -1e-7).all()
    assert (a.random_reduction >= -1e-7).all()
    ci = paired_bootstrap(a.targeted_minus_random_reduction.values, n_boot=150)
    assert ci["ci_95_percentile_lower"] <= ci["ci_95_percentile_upper"]


def test_iid_plr_dml_reproduces_given_seed():
    x, d, y, truth = synthetic_plr(n=350, seed=31)
    one = crossfit_plr(x, d, y, seed=31)
    two = crossfit_plr(x, d, y, seed=31)
    assert one == two
    assert np.isfinite(one["theta_dml"])
    assert abs(one["theta_dml"] - truth) < 1.0


def test_prediction_split_network_disjoint():
    frame = create_node_level_dataset(n_networks=45, n=15, seed=9)
    metrics, calibration, importance, split = evaluate_out_of_network_prediction(
        frame, seed=9
    )
    assert set(split["train_network_ids"]).isdisjoint(split["test_network_ids"])
    assert len(metrics) == 3
    assert metrics.roc_auc.between(0, 1).all()
    assert len(calibration) == 8
    assert "not causal" in split["data_provenance"].lower() or (
        "simulated" in split["data_provenance"].lower()
    )
    assert "feature" in importance
