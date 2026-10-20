"""GAT/GRU on independent synthetic network draws: prediction, NOT causality.

This replaces the historical in-sample training demonstration. The original
class has a treatment head, but it is deliberately unused here; the class
name does not establish causal interpretation.
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
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

from src.models.spatio_temporal_causal_gnn import SpatioTemporalCausalGNN
from src.research.core import draw_network, clear_network


def generate_graphs(count=48, nodes=14, periods=4, seed=20261020):
    """One graph = one IID synthetic draw; all nodes share its loss mechanism."""
    if count < 12 or nodes < 6 or periods < 2:
        raise ValueError("Require >=12 graphs, >=6 nodes, >=2 periods")
    rng = np.random.default_rng(seed)
    graphs = []
    for gid in range(count):
        density = float(rng.uniform(.12, .45))
        shock = float(rng.uniform(.2, .8))
        liabilities, assets, vulnerability = draw_network(rng, nodes, density)
        owed, incoming = liabilities.sum(axis=1), liabilities.sum(axis=0)
        adj = ((liabilities > 0).astype("float32") + np.eye(nodes, dtype="float32"))
        snapshots = []
        for t in range(periods):
            level = shock * (t+1)/periods
            current = assets * (1 - level * vulnerability)
            snapshots.append(np.column_stack([
                np.log1p(owed), np.log1p(incoming), np.log1p(current),
                current / (1 + owed), vulnerability,
                np.full(nodes, level),
            ]).astype("float32"))
        result = clear_network(liabilities, assets * (1-shock*vulnerability))
        graphs.append((
            gid,
            torch.tensor(np.stack(snapshots)),
            torch.tensor(np.repeat(adj[None], periods, axis=0)),
            torch.tensor(result.defaults.astype("float32")),
        ))
    return graphs


def evaluate_gnn(count=48, nodes=14, epochs=24,
                 seed=20261020, output=Path("reports/gnn")):
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    examples = generate_graphs(count, nodes, seed=seed)
    ids = np.random.default_rng(seed+1).permutation(count)
    cutoff = int(.75*count)
    train_idx, test_idx = ids[:cutoff], ids[cutoff:]
    assert not set(train_idx).intersection(test_idx)
    train = [examples[int(i)] for i in train_idx]
    test = [examples[int(i)] for i in test_idx]
    model = SpatioTemporalCausalGNN(in_features=6, hidden_dim=20, dropout=.05)
    opt = torch.optim.AdamW(model.parameters(), lr=.007, weight_decay=.001)
    for _ in range(epochs):
        model.train()
        for _, x, a, y in train:
            opt.zero_grad()
            prob, _, _ = model(x, a)
            loss = F.binary_cross_entropy(prob.flatten(), y)
            loss.backward()
            opt.step()
    records = []
    model.eval()
    with torch.no_grad():
        for gid, x, a, y in test:
            p, _, _ = model(x, a)
            for node_id, (truth, score) in enumerate(zip(
                y.numpy(), p.flatten().numpy()
            )):
                records.append({
                    "graph": gid, "node": node_id,
                    "synthetic_default": float(truth), "predicted_risk": float(score),
                })
    out = pd.DataFrame(records)
    y, p = out.synthetic_default.to_numpy(), out.predicted_risk.to_numpy()
    if len(np.unique(y)) < 2:
        raise RuntimeError("Test collection contains a single class")
    summary = {
        "data_source": "ENTIRELY SYNTHETIC, no actual banks",
        "purpose": "Out-of-network supervised risk prediction, NOT causal DML",
        "train_networks": len(train), "test_networks": len(test),
        "nodes_per_graph": nodes, "epochs": epochs, "seed": seed,
        "torch_version": torch.__version__, "numpy_version": np.__version__,
        "device": "cpu", "deterministic_algorithms_requested": True,
        "test_roc_auc": float(roc_auc_score(y, p)),
        "test_average_precision": float(average_precision_score(y, p)),
        "test_brier": float(brier_score_loss(y, p)),
        "test_default_share": float(y.mean()),
        "caveat": (
            "Independent synthetic test graphs, but no genuine observational "
            "network validation. Model treatment head is unused and untrained. "
            "Tabular baselines elsewhere in the repo use different draws, "
            "so their scores are not paired comparisons."
        ),
    }
    output.mkdir(parents=True, exist_ok=True)
    out.to_csv(output/"gnn_heldout_predictions.csv", index=False)
    (output/"gnn_heldout_metrics.json").write_text(
        json.dumps(summary, indent=2)+"\n", encoding="utf-8"
    )
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].hist(p[y==0], bins=16, alpha=.65, label="Solvent")
    axes[0].hist(p[y==1], bins=16, alpha=.65, label="Default")
    axes[0].legend()
    axes[0].set_xlabel("Predicted default risk")
    axes[0].set_ylabel("Synthetic network nodes")
    axes[0].set_title("Held-out simulated graph predictions")
    cal = out.copy()
    cal["bin"] = pd.qcut(
        cal.predicted_risk.rank(method="first"), q=6,
        labels=[f"B{i}" for i in range(1,7)]
    )
    grouped = cal.groupby("bin", observed=True).agg(
        predicted=("predicted_risk", "mean"),
        observed=("synthetic_default", "mean")
    )
    axes[1].scatter(grouped.predicted, grouped.observed)
    axes[1].plot([0, 1], [0, 1], ls="--", color="gray")
    axes[1].set_xlim(0,1)
    axes[1].set_ylim(0,1)
    axes[1].set_xlabel("Mean predicted risk")
    axes[1].set_ylabel("Fraction default")
    axes[1].set_title("Held-out simulation calibration")
    fig.suptitle("GAT–GRU benchmark — SYNTHETIC GRAPHS ONLY")
    fig.tight_layout()
    fig.savefig(output/"gnn_out_of_network.png", dpi=170)
    fig.savefig(output/"gnn_out_of_network.svg")
    plt.close(fig)
    return summary


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--graphs",type=int,default=48)
    ap.add_argument("--nodes",type=int,default=14)
    ap.add_argument("--epochs",type=int,default=24)
    ap.add_argument("--seed",type=int,default=20261020)
    ap.add_argument("--output",type=Path,default=Path("reports/gnn"))
    a=ap.parse_args()
    print(json.dumps(evaluate_gnn(
        count=a.graphs, nodes=a.nodes, epochs=a.epochs, seed=a.seed,
        output=a.output,
    ),indent=2))


if __name__=="__main__":
    main()
