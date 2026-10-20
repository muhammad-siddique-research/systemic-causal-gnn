"""Known-truth DML exercise, SEPARATE from interbank clearing simulations.

An unconfounded, partially linear synthetic data-generating process permits
mechanism-specific identification checks. Its IID assumption does NOT extend
automatically to network-dependent shocks or real macroprudential actions.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold


def synthetic_plr(n: int, seed: int, theta: float = -1.5):
    if n < 80:
        raise ValueError("Synthetic PLR exercise requires at least 80 observations")
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, 6))
    propensity = .9 * x[:, 0] - .7 * x[:, 1] + .6 * np.sin(x[:, 2])
    treatment = propensity + rng.normal(0, .85, n)
    nuisance = 2 * np.sin(x[:, 0]) + 1.2 * x[:, 1]**2 + .6*x[:, 2]*x[:, 3]
    y = theta * treatment + nuisance + rng.normal(0, .8, n)
    return x, treatment, y, theta


def crossfit_plr(x: np.ndarray, d: np.ndarray, y: np.ndarray,
                 seed: int = 42, folds: int = 5) -> dict:
    """Orthogonal partially linear residual-on-residual DML with cross-fitting."""
    if len(d) < folds * 20 or folds < 2:
        raise ValueError("Invalid DML sample size or folds")
    v, u = np.zeros(len(d)), np.zeros(len(d))
    kf = KFold(n_splits=folds, shuffle=True, random_state=seed)
    for train, test in kf.split(x):
        for target, destination in ((d, v), (y, u)):
            model = RandomForestRegressor(
                n_estimators=65, max_depth=6, min_samples_leaf=8,
                n_jobs=1, random_state=seed
            )
            model.fit(x[train], target[train])
            destination[test] = target[test] - model.predict(x[test])
    j = float(np.mean(v**2))
    if j < 1e-10:
        raise ValueError("Near-zero treatment residual variance")
    beta = float(np.mean(v*u) / j)
    psi = v * (u - beta * v)
    se = float(np.sqrt(np.mean(psi**2) / (j**2 * len(d))))
    naive = sm.OLS(y, sm.add_constant(d)).fit(cov_type="HC3")
    return {
        "theta_dml": beta, "se_dml_iid": se,
        "ci95_low_iid": beta - 1.96 * se,
        "ci95_high_iid": beta + 1.96 * se,
        "naive_ols_slope": float(naive.params[1]),
        "naive_ols_hc3_se": float(naive.bse[1]),
        "folds": folds,
        "caveat": "Simulated IID PLR identification only; not financial network inference",
    }


def dml_monte_carlo(n_rep: int = 12, n: int = 700,
                    seed: int = 20261020) -> pd.DataFrame:
    if n_rep < 2:
        raise ValueError("Need repeated independent experiments")
    rows = []
    for i in range(n_rep):
        x, d, y, true = synthetic_plr(n, seed + i)
        est = crossfit_plr(x, d, y, seed + i)
        rows.append({
            "replication": i, "theta_true": true,
            "theta_dml": est["theta_dml"],
            "theta_naive_ols": est["naive_ols_slope"],
            "dml_bias": est["theta_dml"] - true,
            "naive_bias": est["naive_ols_slope"] - true,
            "dml_ci_covers_truth": est["ci95_low_iid"] <= true <= est["ci95_high_iid"],
            "dml_se_iid": est["se_dml_iid"],
        })
    return pd.DataFrame(rows)
