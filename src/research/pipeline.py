"""End-to-end reproducible research outputs; all network data are synthetic.

USAGE: python -m src.research.pipeline --out reports --networks 150
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from .core import paired_bootstrap, simulate_policy
from .dml import dml_monte_carlo
from .prediction import create_node_level_dataset, evaluate_out_of_network_prediction
from .figures import (
    plot_policy, plot_predictive, plot_dml, plot_example_network
)


def run(out: Path, n_networks: int = 140,
        dml_reps: int = 10, seed: int = 20261020):
    out.mkdir(parents=True, exist_ok=True)
    charts = out / "figures"
    charts.mkdir(exist_ok=True)
    policy = simulate_policy(n_networks=n_networks, seed=seed)
    policy_ci = paired_bootstrap(
        policy.targeted_minus_random_reduction.to_numpy(), seed=seed+1,
        n_boot=1000
    )
    policy.to_csv(out / "synthetic_network_policy_pairs.csv", index=False)

    dml = dml_monte_carlo(n_rep=dml_reps, n=700, seed=seed)
    dml.to_csv(out / "independent_dml_monte_carlo.csv", index=False)

    nodes = create_node_level_dataset(n_networks=max(60, n_networks), seed=seed+2)
    pred, calibration, importance, split = evaluate_out_of_network_prediction(
        nodes, seed=seed+3
    )
    pred.to_csv(out / "out_of_network_predictive_benchmarks.csv", index=False)
    calibration.to_csv(out / "heldout_synthetic_calibration.csv", index=False)
    importance.to_csv(out / "heldout_permutation_reliance.csv", index=False)
    (out / "prediction_split_audit.json").write_text(
        json.dumps(split, indent=2) + "\n", encoding="utf-8"
    )
    plot_policy(policy, policy_ci, charts)
    plot_predictive(pred, calibration, importance, charts)
    plot_dml(dml, charts)
    plot_example_network(charts, seed=seed)
    info = {
        "title": "Network Fragility Under Stress: Simulation and Methods Benchmark",
        "sample_source": "ENTIRELY COMPUTER-GENERATED; no real banks",
        "network_intervention": "Mechanistic counterfactuals, not econometric policy effects",
        "dml_experiment": "Separate IID synthetic PLR with known parameter, no graph spillovers",
        "prediction": "Held-out independent synthetic networks; tabular ML baselines",
        "GNN_status": "Original architecture exploratory; not in the benchmark tables",
        "policy_ci": policy_ci,
        "n_policy_networks": int(len(policy)),
        "seed": seed,
        "disclaimer": (
            "Synthetic evidence cannot establish real-world intervention outcomes, "
            "systemic tipping points, or valid health-sector conclusions."
        ),
    }
    (out / "research_provenance.json").write_text(
        json.dumps(info, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "# Reproduced computational results — synthetic benchmark only", "",
        "**This is not an empirical bank dataset, a trial of real policies, "
        "or a validated causal GNN.**", "",
        f"- Independently generated networks: **{len(policy)}**",
        f"- Mean baseline unpaid obligations: "
        f"**{policy.baseline_shortfall.mean():.2f} synthetic units**",
        f"- Mean loss reduction, injection to largest debtor: "
        f"**{policy.targeted_reduction.mean():.2f}**",
        f"- Mean loss reduction, injection to random debtor: "
        f"**{policy.random_reduction.mean():.2f}**",
        f"- Paired targeted-minus-random mean: "
        f"**{policy_ci['mean_paired_difference']:.2f}** "
        f"(bootstrap 95% interval "
        f"{policy_ci['ci_95_percentile_lower']:.2f} to "
        f"{policy_ci['ci_95_percentile_upper']:.2f})", "",
        "## Out-of-network predictive models", "",
        pred.to_markdown(index=False, floatfmt=".4f"), "",
        "**Qualification:** Tabular supervised classifiers (logistic, RF, "
        "boosting), not a validated graph neural network. "
        "Model selection used test AUC; additional nested validation is needed.", "",
        "## Cross-fitted DML known-truth exercise", "",
        f"- True synthetic coefficient: **{dml.theta_true.iloc[0]:.3f}**",
        f"- Mean naive OLS estimate: **{dml.theta_naive_ols.mean():.3f}**",
        f"- Mean cross-fit DML estimate: **{dml.theta_dml.mean():.3f}**",
        f"- Mean absolute DML deviation from truth: "
        f"**{np.abs(dml.dml_bias).mean():.3f}**",
        f"- Fraction of IID DML 95% CIs covering synthetic truth: "
        f"**{dml.dml_ci_covers_truth.mean():.2%}** "
        f"({len(dml)} simulated replications; not coverage validation).", "",
        "## Research visuals", "",
        *[
            f"![Figure {k}](figures/figure_{k}_{name}.png)"
            for k, name in [
                (1, "policy_interventions"), (2, "paired_clearing"),
                (3, "prediction_validation"), (4, "permutation_reliance"),
                (5, "dml_known_truth"), (6, "synthetic_network")
            ]
        ], "",
        "## Inference boundary", "",
        "The paired bootstrap resamples entire independent simulated networks. "
        "DML variance relies on IID observations in a separately simulated "
        "partially linear model. Neither confidence interval transfers to "
        "real interconnected institutions or economic interventions.", "",
    ]
    (out / "REPRODUCIBLE_RESULTS.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    return info


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("reports"))
    parser.add_argument("--networks", type=int, default=140)
    parser.add_argument("--dml-reps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20261020)
    args = parser.parse_args()
    r = run(args.out, args.networks, args.dml_reps, args.seed)
    print(json.dumps({
        "networks": r["n_policy_networks"],
        "policy_ci": r["policy_ci"],
        "provenance": r["sample_source"]
    }, indent=2))


if __name__ == "__main__":
    main()
