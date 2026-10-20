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
    "initial_asset_to_obligations",
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
                "initial_asset_to_obligations": float(assets[j] / max(owed[j], 1e-6)),
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
    """Three-way independent-network split: fit, choose, then evaluate.

    Model selection uses validation networks ONLY. The untouched test networks
    evaluate all models; after validation chooses the winner, its heldout
    calibration and permutation reliance are computed on test networks.
    """
    splitter = GroupShuffleSplit(
        n_splits=1, test_size=.20, random_state=seed
    )
    trainval, test = next(splitter.split(
        frame[FEATURES], frame["default"], groups=frame["network_id"]
    ))
    development = frame.iloc[trainval]
    internal = GroupShuffleSplit(
        n_splits=1, test_size=.25, random_state=seed + 1
    )
    itrain, ival = next(internal.split(
        development[FEATURES], development["default"],
        groups=development["network_id"]
    ))
    train, validation = trainval[itrain], trainval[ival]
    train_ids = set(frame.iloc[train].network_id)
    validation_ids = set(frame.iloc[validation].network_id)
    test_ids = set(frame.iloc[test].network_id)
    if train_ids & validation_ids or train_ids & test_ids or validation_ids & test_ids:
        raise AssertionError("Network-level split leakage")
    xtr, xval, xte = (
        frame.iloc[train][FEATURES],
        frame.iloc[validation][FEATURES],
        frame.iloc[test][FEATURES],
    )
    ytr, yval, yte = (
        frame.iloc[train]["default"],
        frame.iloc[validation]["default"],
        frame.iloc[test]["default"],
    )
    if any(x.nunique() != 2 for x in (ytr, yval, yte)):
        raise RuntimeError("Every split must contain both default classes")
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
    metrics, probabilities, validation_auc = [], {}, {}
    for name, model in models.items():
        model.fit(xtr, ytr)
        vprob = np.clip(model.predict_proba(xval)[:, 1], 1e-9, 1 - 1e-9)
        probability = np.clip(model.predict_proba(xte)[:, 1], 1e-9, 1 - 1e-9)
        probabilities[name] = probability
        validation_auc[name] = float(roc_auc_score(yval, vprob))
        metrics.append({
            "model": name,
            "n_train": len(train),
            "n_validation": len(validation),
            "n_test": len(test),
            "train_networks": len(train_ids),
            "validation_networks": len(validation_ids),
            "test_networks": len(test_ids),
            "test_event_rate": float(yte.mean()),
            "validation_roc_auc": validation_auc[name],
            "roc_auc": float(roc_auc_score(yte, probability)),
            "average_precision": float(average_precision_score(yte, probability)),
            "brier": float(brier_score_loss(yte, probability)),
            "log_loss": float(log_loss(yte, probability)),
        })
    best_name = max(validation_auc, key=validation_auc.get)
    summary = pd.DataFrame(metrics).sort_values(
        "validation_roc_auc", ascending=False
    )
    summary["selected_on_validation"] = summary["model"] == best_name
    pred = probabilities[best_name]
    calibration = pd.DataFrame({"y": yte.to_numpy(), "p": pred})
    calibration["bin"] = pd.qcut(
        calibration.p.rank(method="first"), q=8,
        labels=[f"B{k}" for k in range(1, 9)]
    )
    calibration = calibration.groupby("bin", observed=True).agg(
        n=("y", "size"), observed_default_share=("y", "mean"),
        predicted_default_share=("p", "mean")
    ).reset_index()
    pi = permutation_importance(
        models[best_name], xte, yte, scoring="roc_auc", n_repeats=4,
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
        "train_network_ids": sorted(train_ids),
        "validation_network_ids": sorted(validation_ids),
        "test_network_ids": sorted(test_ids),
        "selected_model": best_name,
        "selection_rule": "Maximize validation ROC AUC; untouched test for final evaluation",
        "strict_test_note": (
            "All model hyperparameters were fixed in source code before "
            "the current benchmark. Cross-DGP transport remains untested."
        ),
        "data_provenance": "All liabilities/defaults simulated; no real banks.",
    }
    return summary, calibration, importance, split_audit
