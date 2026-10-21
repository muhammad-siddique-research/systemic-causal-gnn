"""Offline fixture tests for real-data pipeline; do not claim a FRED execution."""
import numpy as np
import pandas as pd
import pytest
from src.research.observed_macro import load_research_panel, analyze_real_macro, read_fred_csv


def _fixture(folder, months=180):
    index = pd.date_range("2002-01-01", periods=months, freq="MS")
    rng = np.random.default_rng(12)
    fields = {
        "BAA": 6.2 + .4 * np.sin(np.arange(months) / 8) + rng.normal(0, .15, months),
        "GS10": 4.1 + .2 * np.cos(np.arange(months) / 12),
        "UNRATE": 5.2 + .7 * np.cos(np.arange(months) / 16),
        "INDPRO": 100 * np.exp(np.arange(months) * .001),
        "FEDFUNDS": 2.5 + .7 * np.sin(np.arange(months) / 21),
    }
    folder.mkdir(parents=True)
    for k, v in fields.items():
        pd.DataFrame({"observation_date": index.strftime("%Y-%m-%d"), k: v}).to_csv(
            folder / f"{k}.csv", index=False
        )


def test_validates_and_preserves_monthly_time_origin(tmp_path):
    data_dir = tmp_path / "invented_fixture_NOT_FRED"
    _fixture(data_dir)
    frame, manifest = load_research_panel(data_dir)
    assert len(frame) == 178
    assert frame.index.is_monotonic_increasing
    assert frame.index.min() == pd.Timestamp("2002-02-01")
    assert frame.index.max() == pd.Timestamp("2016-11-01")
    assert manifest["is_real_bank_network_exposure_data"] is False
    assert "not proof" in manifest["data_origin_validation"]


def test_out_of_time_models_and_hac_exports(tmp_path):
    folder = tmp_path / "invented_fixture_NOT_FRED"
    _fixture(folder)
    out = tmp_path / "results"
    result = analyze_real_macro(folder, out)
    assert len(result["models"]) == 3
    assert (out / "figure_8_real_macro_validation.png").exists()
    assert (out / "observed_macro_hac_ols.csv").exists()
    assert (out / "observed_macro_predictions.csv").exists()
    assert result["manifest"]["train_end"] < result["manifest"]["test_start"]


def test_rejects_duplicate_monthly_dates(tmp_path):
    file = tmp_path / "BAA.csv"
    pd.DataFrame({"DATE": ["2005-01-01", "2005-01-12"], "BAA": [5, 5]}).to_csv(
        file, index=False
    )
    with pytest.raises(ValueError, match="monthly"):
        read_fred_csv(file, "BAA")


def test_rejects_missing_real_data(tmp_path):
    with pytest.raises(FileNotFoundError, match="Missing genuine FRED"):
        load_research_panel(tmp_path)
