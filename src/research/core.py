"""Financial network clearing and controlled *synthetic* intervention benchmarks.

A_{ij} >= 0 is a liability of debtor i to creditor j. Results are properties
of a chosen model and synthetic data-generating process, NOT observations of
actual bank networks or evidence of effects of monetary policy.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ClearingResult:
    payments: np.ndarray
    defaults: np.ndarray
    shortfall: float
    iterations: int
    residual: float


def clear_network(
    liabilities: np.ndarray,
    external_assets: np.ndarray,
    tol: float = 1e-10,
    max_iter: int = 1500,
) -> ClearingResult:
    """Compute the *greatest* clearing fixed point, starting at full payments.

    p = min(pbar, max(0, e + Pi.T @ p)).  The limit need not be unique in
    arbitrary nonregular networks. Convergence is tested and failure raised.
    """
    liabilities = np.asarray(liabilities, dtype=float)
    e = np.asarray(external_assets, dtype=float)
    if liabilities.ndim != 2 or liabilities.shape[0] != liabilities.shape[1]:
        raise ValueError("Liabilities must be a square matrix")
    n = liabilities.shape[0]
    if e.shape != (n,) or n < 2:
        raise ValueError("External assets must have one element per node")
    if not np.isfinite(liabilities).all() or not np.isfinite(e).all():
        raise ValueError("Nonfinite financial inputs")
    if (liabilities < 0).any() or (e < 0).any():
        raise ValueError("Financial resources and liabilities must be nonnegative")
    if tol <= 0 or max_iter < 1:
        raise ValueError("Invalid solver tolerance or iteration limit")
    pbar = liabilities.sum(axis=1)
    pi = np.divide(
        liabilities, pbar[:, None], out=np.zeros_like(liabilities),
        where=pbar[:, None] > 0,
    )
    payment = pbar.copy()
    for iteration in range(1, max_iter + 1):
        nxt = np.minimum(pbar, np.maximum(0, e + pi.T @ payment))
        residual = float(np.max(np.abs(nxt - payment)))
        payment = nxt
        if residual < tol:
            return ClearingResult(
                payments=payment,
                defaults=(payment < pbar - 1e-8),
                shortfall=float(np.maximum(pbar - payment, 0).sum()),
                iterations=iteration,
                residual=residual,
            )
    raise RuntimeError("Financial clearing did not converge")


def draw_network(rng: np.random.Generator, n: int = 20,
                 density: float = .22) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate one directed synthetic bilateral liabilities network.

    Parameters are illustrative and NOT calibrated to FR Y-9C or supervisory
    bilateral exposures. No real bank entities, confidential data or IDs.
    """
    if n < 4 or not 0 < density < 1:
        raise ValueError("Invalid network size or edge density")
    mask = rng.random((n, n)) < density
    np.fill_diagonal(mask, False)
    liabilities = rng.gamma(shape=2.0, scale=10.0, size=(n, n)) * mask
    owed = liabilities.sum(axis=1)
    assets = owed * rng.uniform(.25, .9, size=n) + rng.gamma(2.0, 15.0, n)
    exposure_to_shock = rng.uniform(.35, 1.0, size=n)
    return liabilities, assets, exposure_to_shock


def evaluate_one_network(rng: np.random.Generator, network_id: int,
                         n: int = 20, density: float = .22,
                         shock: float = .38, budget_share: float = .12) -> dict:
    """Compare *paired* capital allocation rules holding simulated shocks fixed.

    Spending is equal across rules. Target selection uses pre-shock debt only.
    These are mechanistic model counterfactuals, not identified policy effects.
    """
    if not 0 <= shock <= 1 or budget_share < 0:
        raise ValueError("Shock and capital allocation must be nonnegative")
    liabilities, assets, exposure = draw_network(rng, n, density)
    e = assets * (1 - shock * exposure)
    obligations = liabilities.sum(axis=1)
    budget = float(budget_share * max(obligations.mean(), 1e-8))
    ranked_bank = int(obligations.argmax())
    random_bank = int(rng.integers(n))
    targeted_e = e.copy()
    targeted_e[ranked_bank] += budget
    random_e = e.copy()
    random_e[random_bank] += budget
    base = clear_network(liabilities, e)
    targeted = clear_network(liabilities, targeted_e)
    random_policy = clear_network(liabilities, random_e)
    if targeted.shortfall > base.shortfall + 1e-6 or (
        random_policy.shortfall > base.shortfall + 1e-6
    ):
        raise AssertionError("Simulated added resources unexpectedly increased shortfall")
    return {
        "network_id": network_id, "n_nodes": n, "edge_density_realized": float(
            (liabilities > 0).sum() / (n * (n - 1))
        ),
        "shock": shock, "capital_budget": budget,
        "baseline_shortfall": base.shortfall,
        "targeted_shortfall": targeted.shortfall,
        "random_shortfall": random_policy.shortfall,
        "baseline_defaults": int(base.defaults.sum()),
        "targeted_defaults": int(targeted.defaults.sum()),
        "random_defaults": int(random_policy.defaults.sum()),
        "targeted_reduction": base.shortfall - targeted.shortfall,
        "random_reduction": base.shortfall - random_policy.shortfall,
        "targeted_minus_random_reduction": (
            random_policy.shortfall - targeted.shortfall
        ),
    }


def simulate_policy(n_networks: int = 160, seed: int = 20261020,
                    n: int = 20, density: float = .22,
                    shock: float = .38, budget_share: float = .12) -> pd.DataFrame:
    if n_networks < 3:
        raise ValueError("At least three independently drawn networks required")
    rng = np.random.default_rng(seed)
    rows = [
        evaluate_one_network(rng, idx, n, density, shock, budget_share)
        for idx in range(n_networks)
    ]
    return pd.DataFrame.from_records(rows)


def paired_bootstrap(values: np.ndarray, seed: int = 20261020,
                     n_boot: int = 1000) -> dict:
    """IID intervals over independently drawn synthetic network scenarios."""
    x = np.asarray(values, dtype=float)
    if x.ndim != 1 or len(x) < 3 or not np.isfinite(x).all():
        raise ValueError("Paired differences must be a finite 1d array")
    if n_boot < 100:
        raise ValueError("At least 100 bootstrap replicates required")
    rng = np.random.default_rng(seed)
    indexes = rng.integers(0, len(x), size=(n_boot, len(x)))
    estimates = x[indexes].mean(axis=1)
    lower, upper = np.quantile(estimates, [.025, .975])
    return {
        "n_networks": len(x),
        "mean_paired_difference": float(x.mean()),
        "ci_95_percentile_lower": float(lower),
        "ci_95_percentile_upper": float(upper),
        "bootstrap_iterations": n_boot,
        "interpretation": (
            "Conditional on independent synthetic networks from the specified "
            "DGP. This is not real-world central bank policy uncertainty."
        ),
    }
