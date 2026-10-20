"""Publication-style, source-labeled figures for simulated macroprudential science."""
from __future__ import annotations
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import networkx as nx
from .core import clear_network, draw_network

INK, TEAL, BLUE, ORANGE = "#102a43", "#0b8b82", "#2563aa", "#c96a20"


def _write(fig, path: Path):
    fig.patch.set_facecolor("#ffffff")
    fig.tight_layout(pad=2)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    fig.savefig(path.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def plot_policy(policy: pd.DataFrame, ci: dict, dest: Path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    data = [
        policy["baseline_shortfall"],
        policy["targeted_shortfall"],
        policy["random_shortfall"],
    ]
    axes[0].boxplot(data, tick_labels=[
        "No injection", "Largest debtor", "Random debtor"
    ], showfliers=False)
    axes[0].set_ylabel("Synthetic network unpaid obligations (arbitrary units)")
    axes[0].set_title("Paired mechanistic clearing scenarios")
    delta = policy["targeted_minus_random_reduction"]
    axes[1].hist(delta, bins=28, color=TEAL, alpha=.82)
    axes[1].axvline(0, color=ORANGE, ls="--", lw=2)
    axes[1].axvline(delta.mean(), color=BLUE, lw=2,
                    label=f"Mean paired difference = {delta.mean():.2f}")
    axes[1].set_xlabel("Targeted minus random reduction in unpaid obligations")
    axes[1].set_ylabel("Independent synthetic networks")
    axes[1].set_title("Policy-rule advantage is not assumed")
    axes[1].legend()
    fig.suptitle(
        "Network-stress intervention comparison — SIMULATED NETWORKS ONLY",
        fontsize=13, weight="semibold", color=INK
    )
    _write(fig, dest / "figure_1_policy_interventions.png")

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    ax.scatter(
        policy.baseline_shortfall, policy.targeted_shortfall,
        alpha=.55, color=TEAL, s=20
    )
    lim = max(policy.baseline_shortfall.max(), policy.targeted_shortfall.max())
    ax.plot([0, lim], [0, lim], ls="--", color=ORANGE)
    ax.set_xlabel("Baseline unpaid obligations (synthetic)")
    ax.set_ylabel("After injection to highest debtor (synthetic)")
    ax.set_title("Mechanism-consistent monotonicity: adding assets cannot raise unpaid claims")
    _write(fig, dest / "figure_2_paired_clearing.png")


def plot_predictive(perf: pd.DataFrame, calibration: pd.DataFrame,
                    importance: pd.DataFrame, dest: Path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    q = perf.sort_values("roc_auc")
    axes[0].barh(q.model, q.roc_auc, color=[ORANGE, BLUE, TEAL])
    axes[0].set_xlim(0, 1)
    axes[0].set_xlabel("ROC AUC on held-out synthetic networks")
    axes[0].set_title("Out-of-network benchmark — NOT GNN validation")
    axes[1].scatter(
        calibration.predicted_default_share,
        calibration.observed_default_share,
        s=58, color=TEAL
    )
    axes[1].plot([0, 1], [0, 1], ls="--", color=ORANGE)
    axes[1].set_xlabel("Mean predicted default probability")
    axes[1].set_ylabel("Observed synthetic default fraction")
    axes[1].set_title("Calibration of selected tree/tabular model")
    fig.suptitle("Synthetic network-level holdout diagnostics", color=INK)
    _write(fig, dest / "figure_3_prediction_validation.png")

    imp = importance.sort_values("test_roc_auc_drop_mean").tail(10)
    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    ax.barh(imp.feature, imp.test_roc_auc_drop_mean, color=BLUE)
    ax.axvline(0, color=ORANGE, ls="--")
    ax.set_xlabel("Held-out AUC decrease after covariate permutation")
    ax.set_title("Predictive reliance; NOT causal covariate effects")
    _write(fig, dest / "figure_4_permutation_reliance.png")


def plot_dml(reps: pd.DataFrame, dest: Path):
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.boxplot(
        [reps.theta_naive_ols, reps.theta_dml],
        tick_labels=["Naive unadjusted OLS", "Cross-fit DML"],
        showfliers=True
    )
    ax.axhline(reps.theta_true.iloc[0], color=ORANGE, linestyle="--", lw=2,
               label="Prespecified synthetic truth")
    ax.set_ylabel("Simulated treatment parameter estimate")
    ax.set_title("Independent Monte Carlo DGP — not actual liquidity-policy effects")
    ax.legend(frameon=False)
    _write(fig, dest / "figure_5_dml_known_truth.png")


def plot_example_network(dest: Path, seed: int = 20261020):
    rng = np.random.default_rng(seed)
    liabilities, e, expos = draw_network(rng, n=18)
    result = clear_network(liabilities, e * (1 - .5 * expos))
    g = nx.from_numpy_array(liabilities, create_using=nx.DiGraph)
    positions = nx.spring_layout(g, seed=seed, weight="weight")
    fig, ax = plt.subplots(figsize=(9, 7))
    colors = [ORANGE if result.defaults[i] else TEAL for i in g.nodes]
    nx.draw_networkx_edges(g, positions, arrows=True,
        alpha=.25, ax=ax, width=.9, arrowsize=7)
    nx.draw_networkx_nodes(g, positions, node_color=colors, node_size=520, ax=ax,
        edgecolors="white")
    nx.draw_networkx_labels(g, positions, font_size=9, font_color="white", ax=ax)
    ax.set_title("Illustrative directed obligations graph under synthetic shock")
    ax.set_axis_off()
    _write(fig, dest / "figure_6_synthetic_network.png")
