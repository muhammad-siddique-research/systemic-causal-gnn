"""Compatibility wrapper for convergence-checked greatest clearing payments.

For complete numerical diagnostics use src.research.core.clear_network.
An arbitrary nonregular system may admit multiple clearing fixed points.
"""
from __future__ import annotations

import numpy as np
from src.research.core import clear_network


def compute_eisenberg_noe_clearing(
    liabilities: np.ndarray, external_assets: np.ndarray,
    max_iter: int = 1000, tol: float = 1e-8
) -> tuple[np.ndarray, np.ndarray]:
    outcome = clear_network(
        liabilities, external_assets, tol=tol, max_iter=max_iter
    )
    return outcome.payments, outcome.defaults
