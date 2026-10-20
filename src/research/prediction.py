"""Out-of-network default prediction from independent *synthetic* exposures.

Avoid in-sample node leakage: train/test assignment is grouped by independently
drawn financial networks, never randomly by nodes of the same network.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import (
    roc_auc_score, average_precision_score, brier_score_loss, log_loss
)
from sklearn.inspection import permutation_importance
from .core import clear_network, draw_network

FEATURES = [
    "debt_owed", "claims_receivable", "initial_external_assets",
    "assets_after_shock", "asset_to_obligations",
    "weighted_in_degree", "weighted_out_degree", "exposure_to_shock",
    "network_density", "shock_strength",
]


def create_node_level_dataset(
    n_networks: int = 120, n: int = 20,
    seed: int = 20261020
) -> pd.DataFrame:
    if n_networks < 20:
        raise ValueError("Out-of-network benchmark requires at least 20 networks")
    rng = np.random.default_rng(seed)
    records = []
    for idx in range(n_networks):
        density = float(rng.uniform(.10, .4))
        strength = float(rng.uniform(.15, .75))
        liabilities, assets, exposure = draw_network(rng, n, density)
        owed, receivable = liabilities.sum(axis=1), liabilities.sum(axis=0)
        shocked = assets * (1 - strength * exposure)
        clearing = clear_network(liabilities, shocked)
        for j in range(n):
            records.append({
                "network_id": idx, "node_id": j,
                "default": int(clearing.defaults[j]),
                "debt_owed": float(owed[j]),
                "claims_receivable": float(receivable[j]),
                "initial_external_assets": float(assets[j]),
                "assets_after_shock": float(shocked[j]),
                "asset_to_obligations": float(shocked[j] / max(owed[j], 1e-6)),
                "weighted_in_degree": float((liabilities[:, j] > 0).sum()),
                "weighted_out_degree": float((liabilities[j] > 0).sum()),
                "exposure_to_shock": float(exposure[j]),
                "network_density": density,
                "shock_strength": strength,
            })
    return pd.DataFrame(records)


def evaluate_out_of_network_prediction(
    frame: pd.DataFrame, seed: int = 20261020
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """Report out-of-network probability metrics and feature reliance.

    The bank-feature models are comparison baselines, not a trained GNN.
    Results do not establish real bank default forecast performance.
    """
    splitter = GroupShuffleSplit(
        n_splits=1, test_size=.25, random_state=seed
    )
    train, test = next(splitter.split(
        frame[FEATURES], frame["default"], groups=frame["network_id"]
    ))
    xtr, xte = frame.iloc[train][FEATURES], frame.iloc[test][FEATURES]
    ytr, yte = frame.iloc[train]["default"], frame.iloc[test]["default"]
    if ytr.nunique() != 2 or yte.nunique() != 2:
        raise RuntimeError("Generated sample lacks both classes in split")
    models = {
        "Regularized Logistic": make_pipeline(
            StandardScaler(), LogisticRegression(
                C=1.0, max_iter=700, random_state=seed
            )
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=120, min_samples_leaf=9,
            max_depth=7, n_jobs=1, random_state=seed
        ),
        "Gradient-Boosted Trees": HistGradientBoostingClassifier(
            learning_rate=.07, max_iter=110, max_leaf_nodes=12,
            l2_regularization=5., random_state=seed,
            min_samples_leaf=25, early_stopping=False
        ),
    }
    scores, fitted = [], {}
    for name, model in models.items():
        model.fit(xtr, ytr)
        p = np.clip(model.predict_proba(xte)[:, 1], 1e-9, 1 - 1e-9)
        fitted[name] = p
        scores.append({
            "model": name, "n_train": len(train), "n_test": len(test),
            "train_networks": int(frame.iloc[train].network_id.nunique()),
            "test_networks": int(frame.iloc[test].network_id.nunique()),
            "test_event_rate": float(yte.mean()),
            "roc_auc": float(roc_auc_score(yte, p)),
            "average_precision": float(average_precision_score(yte, p)),
            "brier": float(brier_score_loss(yte, p)),
            "log_loss": float(log_loss(yte, p)),
        })
    summary = pd.DataFrame(scores).sort_values("roc_auc", ascending=False)
    best_name = str(summary.iloc[0]["model"])
    pred = fitted[best_name]
    calibration = pd.DataFrame({"y": yte.to_numpy(), "p": pred})
    calibration["bin"] = pd.qcut(
        calibration.p.rank(method="first"), q=8,
        labels=[f"B{k}" for k in range(1, 9)]
    )
    calibration = calibration.groupby("bin", observed=True).agg(
        n=("y", "size"), observed_default_share=("y", "mean"),
        predicted_default_share=("p", "mean")
    ).reset_index()
    best = models[best_name]
    pi = permutation_importance(
        best, xte, yte, scoring="roc_auc", n_repeats=4,
        random_state=seed, n_jobs=1
    )
    importance = pd.DataFrame({
        "feature": FEATURES,
        "test_roc_auc_drop_mean": pi.importances_mean,
        "test_roc_auc_drop_sd": pi.importances_std,
    }).sort_values("test_roc_auc_drop_mean", ascending=False)
    split_audit = {
        "random_seed": seed,
        "unit_of_split": "entire independent synthetic network",
        "train_network_ids": sorted(frame.iloc[train].network_id.unique().tolist()),
        "test_network_ids": sorted(frame.iloc[test].network_id.unique().tolist()),
        "best_model": best_name,
        "selection_caution": (
            "Selecting the best model by test AUC and reporting its "
            "test-calibration on that same split can introduce selection "
            "optimism. Add nested tuning and a separate untouched test "
            "collection before publication."
        ),
        "data_provenance": "All financial networks simulated, none observed.",
    }
    assert not set(split_audit["train_network_ids"]).intersection(
        split_audit["test_network_ids"]
    )
    return summary, calibration, importance, split_audit
