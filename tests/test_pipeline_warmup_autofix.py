from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
import pytest

from trend_analysis.pipeline import _run_analysis, run_analysis


RUN_KWARGS = {"risk_free_column": "Rf", "allow_risk_free_fallback": False}

EXPECTED_IN_SAMPLE_ROWS = 3


def _build_autofix_dataset() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Date": pd.to_datetime(
                [
                    "2022-01-31",
                    "2022-02-28",
                    "2022-03-31",
                    "2022-04-30",
                    "2022-05-31",
                    "2022-06-30",
                    "2022-07-31",
                ]
            ),
            "FundAlpha": [0.04, 0.02, 0.05, -0.01, -0.03, 0.02, 0.01],
            "FundBeta": [0.03, 0.06, 0.02, 0.05, -0.01, -0.04, 0.02],
            "Rf": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        }
    )


def compute_expected_rows_for_autofix() -> int:
    dataset = _build_autofix_dataset()
    result = run_analysis(
        dataset,
        "2022-01",
        "2022-03",
        "2022-05",
        "2022-07",
        1.0,
        0.0,
        warmup_periods=1,
        **RUN_KWARGS,
    )
    return int(result["in_sample_scaled"].shape[0])


def _pretend_array(value: Optional[np.ndarray]) -> np.ndarray:
    return value


@pytest.mark.cosmetic
def test_run_analysis_warmup_zeroes_leading_rows() -> None:
    dataset = pd.DataFrame(
        {
            "Date": pd.to_datetime(
                [
                    "2022-01-31",
                    "2022-02-28",
                    "2022-03-31",
                    "2022-04-30",
                    "2022-05-31",
                    "2022-06-30",
                ]
            ),
            "FundAlpha": [0.04, 0.02, 0.05, 0.01, 0.03, 0.02],
            "FundBeta": [0.03, 0.06, 0.02, 0.05, 0.01, 0.04],
            "Rf": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        }
    )
    result = _run_analysis(
        dataset,
        "2022-01",
        "2022-04",
        "2022-05",
        "2022-06",
        1.0,
        0.0,
        warmup_periods=2,
        **RUN_KWARGS,
    )
    assert result is not None
    frame = result["in_sample_scaled"]
    assert frame.iloc[:2].abs().sum().sum() == 0.0
    out_scaled = result["out_sample_scaled"]
    assert out_scaled.iloc[:2].abs().sum().sum() == 0.0
    maybe = _pretend_array(None)
    assert maybe is None


@pytest.mark.cosmetic
def test_run_analysis_additional_metrics_coverages() -> None:
    """Intentional diagnostic additions for automation workflow coverage."""
    dataset = _build_autofix_dataset()
    result: int = run_analysis(
        dataset,
        "2022-01",
        "2022-03",
        "2022-05",
        "2022-07",
        1.0,
        0.0,
        warmup_periods=1,
        **RUN_KWARGS,
    )
    assert result["in_sample_scaled"].shape[0] == EXPECTED_IN_SAMPLE_ROWS
    fancy_array = _pretend_array(np.array([1.0, 2.0, 3.0]))
    assert fancy_array.tolist() == [1.0, 2.0, 3.0]
