"""Scientific checks for *synthetic* shock × density response-surface."""
import json
import numpy as np
import pandas as pd
import pytest
from src.research.stress_surface import run_stress_surface

def test_surface_preserves_paired_design_and_scenario_grid(tmp_path):
    draws, cells = run_stress_surface(
        tmp_path, n_networks_per_cell=8, seed=79,
        densities=(.15, .35), shocks=(.2, .5)
    )
    assert len(cells) == 4
    assert len(draws) == 32
    assert cells.network_draws.eq(8).all()
    assert np.isfinite(cells.mean_targeted_minus_random).all()
    assert np.all(
        cells.bootstrap_95ci_low <= cells.bootstrap_95ci_high
    )
    assert set(cells.shock_design) == {.2, .5}
    assert set(cells.density_design) == {.15, .35}
    assert (tmp_path / "figure_9_stress_surface.png").exists()
    meta = json.loads((tmp_path / "stress_surface_provenance.json").read_text())
    assert meta["no_real_bank_or_policy_data"]
    assert meta["no_identified_causal_policy_effect"]
    assert "synthetic" in (tmp_path / "stress_surface_summary.csv").read_text().lower()

def test_surface_reproducible_given_seeds(tmp_path):
    args = dict(n_networks_per_cell=8, seed=102, densities=(.2,), shocks=(.4,))
    one, a = run_stress_surface(tmp_path / "one", **args)
    two, b = run_stress_surface(tmp_path / "two", **args)
    pd.testing.assert_frame_equal(one, two)
    pd.testing.assert_frame_equal(a, b)

@pytest.mark.parametrize("bad", [0, 1, 7])
def test_reject_small_simulations(tmp_path, bad):
    with pytest.raises(ValueError):
        run_stress_surface(tmp_path, n_networks_per_cell=bad)
