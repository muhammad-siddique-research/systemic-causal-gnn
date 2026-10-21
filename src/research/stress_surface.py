"""Distributional robustness: paired simulated capital rules by shock and density.

This is a computer-generated *mechanism response surface*; density and shock
levels are experimental simulation parameters, not observed macroprudential
treatments. Bootstrap resamples entire independent network draws.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .core import simulate_policy, paired_bootstrap

DEFAULT_DENSITIES = (.12, .22, .38)
DEFAULT_SHOCKS = (.2, .4, .65)


def run_stress_surface(
    out: Path,
    n_networks_per_cell: int = 40,
    seed: int = 20261020,
    densities: tuple[float, ...] = DEFAULT_DENSITIES,
    shocks: tuple[float, ...] = DEFAULT_SHOCKS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Nine prespecified equal-budget counterfactual simulation conditions."""
    if n_networks_per_cell < 8:
        raise ValueError("At least eight independent networks per cell required")
    if not densities or not shocks or any(not 0 < d < 1 for d in densities):
        raise ValueError("Invalid density grid")
    if any(not 0 <= s <= 1 for s in shocks):
        raise ValueError("Shock severity outside 0–1")
    out.mkdir(parents=True, exist_ok=True)
    detail, summary = [], []
    for i, density in enumerate(densities):
        for j, shock in enumerate(shocks):
            cellseed = seed + i * 101 + j * 13
            cell = simulate_policy(
                n_networks=n_networks_per_cell, seed=cellseed,
                n=20, density=density, shock=shock, budget_share=.12
            )
            cell["density_design"] = density
            cell["shock_design"] = shock
            cell["simulation_seed"] = cellseed
            detail.append(cell)
            delta = cell["targeted_minus_random_reduction"].to_numpy()
            intervals = paired_bootstrap(delta, seed=cellseed + 500, n_boot=500)
            summary.append({
                "density_design": density, "shock_design": shock,
                "network_draws": len(cell),
                "mean_baseline_shortfall": float(cell.baseline_shortfall.mean()),
                "mean_targeted_reduction": float(cell.targeted_reduction.mean()),
                "mean_random_reduction": float(cell.random_reduction.mean()),
                "mean_targeted_minus_random": float(delta.mean()),
                "bootstrap_95ci_low": intervals["ci_95_percentile_lower"],
                "bootstrap_95ci_high": intervals["ci_95_percentile_upper"],
                "probability_rule_a_outperforms_b": float((delta > 0).mean()),
                "unit": "arbitrary synthetic liabilities units, NOT real dollars",
                "identification": "controlled simulator only; NOT observed policy effects",
            })
    draws = pd.concat(detail, ignore_index=True)
    results = pd.DataFrame(summary)
    draws.to_csv(out / "stress_surface_network_draws.csv", index=False)
    results.to_csv(out / "stress_surface_summary.csv", index=False)
    provenance = {
        "source": "ENTIRELY COMPUTER-GENERATED directed liability networks",
        "n_networks_per_cell": n_networks_per_cell,
        "simulation_seed_base": seed,
        "nodes_per_network": 20,
        "equal_budget_share_of_mean_obligations": .12,
        "densities_prespecified": list(densities),
        "shock_fractions_prespecified": list(shocks),
        "bootstrap_unit": "independently simulated network",
        "no_real_bank_or_policy_data": True,
        "no_identified_causal_policy_effect": True,
        "compare_across_cells": (
            "Cells change the synthetic network-generating process. Their "
            "contrasts are mechanistic sensitivity tests; no real-world "
            "estimand or structural bank calibration is asserted."
        ),
    }
    (out / "stress_surface_provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
    )
    fig, ax = plt.subplots(figsize=(9.6, 5.8))
    for density, frame in results.groupby("density_design"):
        frame = frame.sort_values("shock_design")
        ax.errorbar(
            frame.shock_design,
            frame.mean_targeted_minus_random,
            yerr=[frame.mean_targeted_minus_random - frame.bootstrap_95ci_low,
                  frame.bootstrap_95ci_high - frame.mean_targeted_minus_random],
            marker="o", capsize=4, lw=2,
            label=f"Directed edge density={density:.2f}",
        )
    ax.axhline(0, color="#cc6b30", linestyle="--")
    ax.set_xlabel("Externally prescribed asset shock fraction (synthetic)")
    ax.set_ylabel("Targeted minus random shortfall reduction\n(arbitrary synthetic units)")
    ax.set_title("Stress-response surface: shock × network density (SIMULATED)")
    ax.legend(frameon=False)
    ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(out / "figure_9_stress_surface.png", dpi=180)
    fig.savefig(out / "figure_9_stress_surface.svg")
    plt.close(fig)
    return draws, results


def main() -> None:
    p = argparse.ArgumentParser(description="Paired synthetic network sensitivity design")
    p.add_argument("--out", type=Path, default=Path("reports/stress_surface"))
    p.add_argument("--networks-per-cell", type=int, default=40)
    p.add_argument("--seed", type=int, default=20261020)
    a = p.parse_args()
    _, results = run_stress_surface(
        a.out, n_networks_per_cell=a.networks_per_cell, seed=a.seed
    )
    print(results.to_string(index=False))
    print("All financial networks and capital policy rules are simulated.")


if __name__ == "__main__":
    main()
