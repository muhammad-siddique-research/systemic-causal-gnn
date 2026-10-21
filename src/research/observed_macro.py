"""Real U.S. macro-financial forecasting using independently downloaded FRED CSVs.

Source: Federal Reserve Bank of St. Louis FRED (BAA, GS10, UNRATE, INDPRO,
FEDFUNDS). This module NEVER constructs bank-by-bank edges, tests real network
contagion or identifies causal policy interventions.

Real FRED files must be downloaded and provided by the user/workflow;
sample training data are not generated or claimed to be observed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SERIES = ("BAA", "GS10", "UNRATE", "INDPRO", "FEDFUNDS")
SOURCE = "https://fred.stlouisfed.org/series/"
REGRESSORS = ("spread_t", "unemployment_t", "fedfunds_t", "indpro_growth_t")
DATA_NOTE = (
    "Observed U.S. aggregate time series, NOT interbank exposures. "
    "Revised FRED snapshots may embed information unavailable to forecasters "
    "at an original historical release date; this is retrospective, not "
    "vintage-real-time validation."
)


def read_fred_csv(path: Path, series: str) -> pd.Series:
    """Strictly validate a FRED CSV series with one observation per calendar month."""
    if series not in SERIES:
        raise ValueError(f"Unrecognized FRED series: {series}")
    df = pd.read_csv(path)
    time_col = next((c for c in ("observation_date", "DATE") if c in df), None)
    if time_col is None or series not in df:
        raise ValueError(f"{path.name}: expected date column and {series}")
    ts = pd.to_datetime(df[time_col], errors="coerce")
    if ts.isna().any() or ts.duplicated().any():
        raise ValueError(f"{series}: invalid or duplicated observation dates")
    obs = pd.to_numeric(df[series].replace(".", np.nan), errors="coerce")
    frame = pd.DataFrame({"date": ts, series: obs}).set_index("date").sort_index()
    if frame.empty or frame[series].dropna().empty:
        raise ValueError(f"{series}: no usable records")
    # Monthly FRED series specified in the study protocol.
    months = frame.index.to_period("M")
    if months.duplicated().any():
        raise ValueError(f"{series}: expected monthly frequency; duplicate months")
    frame.index = months.to_timestamp()
    return frame[series].rename(series)


def load_research_panel(data_dir: Path) -> tuple[pd.DataFrame, dict]:
    """Align authentic monthly inputs and build next-month credit spread target."""
    found = {name: data_dir / f"{name}.csv" for name in SERIES}
    missing = [name for name, path in found.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing genuine FRED CSVs: " + ", ".join(missing))
    streams = [read_fred_csv(path, name) for name, path in found.items()]
    original = pd.concat(streams, axis=1, join="inner").sort_index()
    # A Baa-minus-Treasury yield spread in percentage points, not an observed
    # bilateral bank-network liability or regulatory distress label.
    original["spread_t"] = original["BAA"] - original["GS10"]
    original["unemployment_t"] = original["UNRATE"]
    original["fedfunds_t"] = original["FEDFUNDS"]
    original["indpro_growth_t"] = 100 * np.log(original["INDPRO"]).diff()
    original["spread_t_plus_1"] = original["spread_t"].shift(-1)
    model = original[["spread_t_plus_1", *REGRESSORS]].replace(
        [np.inf, -np.inf], np.nan
    ).dropna()
    # Enforce adjacent months for proper horizon alignment, even around gaps.
    expected = model.index.to_period("M") + 1
    future_months = original.index.to_series().shift(-1).reindex(model.index)
    keep = (future_months.dt.to_period("M") == expected).to_numpy()
    model = model.loc[keep]
    if len(model) < 120 or not model.index.is_monotonic_increasing:
        raise ValueError("Insufficient valid monthly observations for temporal evaluation")
    manifest = {
        "observed_source": "Federal Reserve Bank of St. Louis FRED",
        "series": {
            key: {"url": SOURCE + key, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                  "bytes": path.stat().st_size}
            for key, path in found.items()
        },
        "source_is_real_macro_data": True,
        "is_real_bank_network_exposure_data": False,
        "data_origin_validation": (
            "SHA-256 fingerprints identify local input bytes, not proof that "
            "a user-supplied CSV came from FRED; official origin requires independent checks."
        ),
        "limitations": DATA_NOTE,
        "observations_model": int(len(model)),
        "start_month": str(model.index.min().date()),
        "end_month": str(model.index.max().date()),
        "outcome": "One-month-ahead Moody's Baa corporate yield minus GS10, percentage points",
        "regressors": list(REGRESSORS),
    }
    return model, manifest


def _hac_loss_test(errors_benchmark: np.ndarray, errors_candidate: np.ndarray) -> dict:
    """Newey-West HAC uncertainty for candidate minus persistence squared loss."""
    diff = errors_candidate**2 - errors_benchmark**2
    fit = sm.OLS(diff, np.ones((len(diff), 1))).fit(
        cov_type="HAC", cov_kwds={"maxlags": 3}
    )
    estimate = float(fit.params[0])
    standard_error = float(fit.bse[0])
    return {
        "mean_candidate_minus_persistence_squared_loss": estimate,
        "hac_se": standard_error,
        "hac_95ci_low": estimate - 1.959964 * standard_error,
        "hac_95ci_high": estimate + 1.959964 * standard_error,
        "lag_truncation": 3,
        "interpretation": (
            "Exploratory HAC interval for squared-error difference; "
            "not a guaranteed Diebold-Mariano test or causal policy finding."
        ),
    }


def analyze_real_macro(data_dir: Path, out: Path, test_fraction: float = .25) -> dict:
    panel, manifest = load_research_panel(data_dir)
    if not .15 <= test_fraction <= .4:
        raise ValueError("test_fraction must be between .15 and .40")
    n_train = int(len(panel) * (1 - test_fraction))
    if n_train < 80 or len(panel) - n_train < 25:
        raise ValueError("Not enough temporally held-out observations")
    train, test = panel.iloc[:n_train], panel.iloc[n_train:]
    xtr = train[list(REGRESSORS)].to_numpy(dtype=float)
    xte = test[list(REGRESSORS)].to_numpy(dtype=float)
    ytr = train["spread_t_plus_1"].to_numpy(dtype=float)
    yte = test["spread_t_plus_1"].to_numpy(dtype=float)

    # Only training data determine fitted model parameters. This evaluation
    # uses a fixed historical training window, not a rolling refit strategy.
    models = {
        "Persistence benchmark": None,
        "Ridge linear": make_pipeline(StandardScaler(), Ridge(alpha=10.0)),
        "Histogram boosted trees": HistGradientBoostingRegressor(
            loss="squared_error", max_iter=100, learning_rate=.045,
            max_leaf_nodes=9, min_samples_leaf=20, l2_regularization=5.0,
            early_stopping=False, random_state=20261020
        ),
    }
    records = []
    predictions = pd.DataFrame({"month": test.index.strftime("%Y-%m"),
                                "actual_spread": yte})
    errors = {}
    for label, estimator in models.items():
        if estimator is None:
            fitted = test["spread_t"].to_numpy(dtype=float)
        else:
            estimator.fit(xtr, ytr)
            fitted = estimator.predict(xte)
        err = yte - fitted
        errors[label] = err
        predictions[label] = fitted
        records.append({
            "model": label, "n_training_months": len(train),
            "n_testing_months": len(test),
            "mae_percentage_points": float(mean_absolute_error(yte, fitted)),
            "rmse_percentage_points": float(np.sqrt(mean_squared_error(yte, fitted))),
            "interpretation": "Retrospective out-of-time forecasting only",
        })
    model_table = pd.DataFrame(records)
    inference = {
        label: _hac_loss_test(errors["Persistence benchmark"], errors[label])
        for label in ("Ridge linear", "Histogram boosted trees")
    }

    # HAC on training-period *conditional projection* coefficients. Endogenous
    # macroeconomic indicators imply associations, never interventions.
    x = sm.add_constant(train[list(REGRESSORS)], has_constant="add")
    ols = sm.OLS(train["spread_t_plus_1"], x).fit(
        cov_type="HAC", cov_kwds={"maxlags": 3}
    )
    table = pd.DataFrame({
        "term": list(ols.params.index),
        "coefficient": ols.params.to_numpy(),
        "hac_se": ols.bse.to_numpy(),
        "hac_95ci_lower": ols.conf_int().iloc[:, 0].to_numpy(),
        "hac_95ci_upper": ols.conf_int().iloc[:, 1].to_numpy(),
        "exploratory_pvalue": ols.pvalues.to_numpy(),
    })
    out.mkdir(exist_ok=True, parents=True)
    table.to_csv(out / "observed_macro_hac_ols.csv", index=False)
    model_table.to_csv(out / "observed_macro_holdout_forecasts.csv", index=False)
    predictions.to_csv(out / "observed_macro_predictions.csv", index=False)
    manifest["train_end"] = str(train.index.max().date())
    manifest["test_start"] = str(test.index.min().date())
    manifest["test_end"] = str(test.index.max().date())
    manifest["forecast_comparison_hac"] = inference
    (out / "observed_macro_provenance.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    fig, axes = plt.subplots(2, 1, figsize=(10.8, 7.8), sharex=True)
    colors = ["#334155", "#0b8b82", "#c96a20"]
    axes[0].plot(test.index, yte, lw=2.5, color="black", label="Observed credit spread")
    for color, model in zip(colors, models):
        axes[0].plot(test.index, predictions[model], lw=1.3, alpha=.8,
                     color=color, label=model)
    axes[0].set_ylabel("Corporate Baa minus GS10 (percentage points)")
    axes[0].legend(ncol=2, fontsize=9)
    axes[0].set_title("Observed U.S. data: out-of-time credit spread forecasting")
    axes[1].barh(model_table.model, model_table.rmse_percentage_points,
                 color=colors)
    axes[1].set_xlabel("Held-out RMSE (percentage points; lower is better)")
    axes[1].grid(axis="x", alpha=.15)
    fig.suptitle("FRED macro-financial indicators — NOT interbank network contagion")
    fig.tight_layout()
    fig.savefig(out / "figure_8_real_macro_validation.png", dpi=180)
    fig.savefig(out / "figure_8_real_macro_validation.svg")
    plt.close(fig)
    return {"models": model_table.to_dict(orient="records"), "manifest": manifest}


def main() -> None:
    p = argparse.ArgumentParser(description="Analyze authentic user-downloaded FRED indicators")
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, default=Path("reports/observed_macro"))
    a = p.parse_args()
    result = analyze_real_macro(a.data_dir, a.output_dir)
    print(json.dumps({"status": "real observed macro data (not bank networks)",
                      "models": result["models"]}, indent=2))


if __name__ == "__main__":
    main()
