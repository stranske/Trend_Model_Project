"""Extended coverage tests for ``trend_analysis.io.validators``."""

from __future__ import annotations

import io
from datetime import datetime

import pandas as pd
import pytest

from trend_analysis.io.market_data import (
    MarketDataMetadata,
    MarketDataMode,
    MissingPolicyFillDetails,
)
from trend_analysis.io.validators import (
    _ValidationSummary,
    _read_uploaded_file,
    detect_frequency,
)
from trend_analysis.io.market_data import MarketDataValidationError


def test_validation_summary_emits_all_warning_types() -> None:
    frame = pd.DataFrame(
        {
            "A": [1.0, None, 3.0, None],
            "B": [None, None, None, None],
        }
    )
    metadata = MarketDataMetadata.model_construct(
        mode=MarketDataMode.RETURNS,
        frequency="M",
        frequency_detected="M",
        frequency_label="monthly",
        start=datetime(2023, 1, 31),
        end=datetime(2023, 4, 30),
        rows=4,
        columns=["A", "B"],
        symbols=["A", "B"],
        missing_policy="drop",
        missing_policy_summary="Filled 2 cells",
        missing_policy_dropped=["B"],
        missing_policy_filled={
            "A": MissingPolicyFillDetails(method="ffill", count=2)
        },
        frequency_missing_periods=2,
        frequency_max_gap_periods=1,
    )

    summary = _ValidationSummary(metadata, frame)
    warnings = summary.warnings()

    assert any("quite small" in warning for warning in warnings)
    assert any("50% missing" in warning for warning in warnings)
    assert any("missing monthly periods" in warning for warning in warnings)
    assert any("Missing-data policy dropped" in warning for warning in warnings)
    assert any("policy applied" in warning for warning in warnings)


def test_detect_frequency_handles_irregular_error(monkeypatch: pytest.MonkeyPatch) -> None:
    index = pd.date_range("2023-01-01", periods=3, freq="D")

    with monkeypatch.context() as mp:
        mp.setattr(
            "trend_analysis.io.validators.classify_frequency",
            lambda idx: (_ for _ in ()).throw(
                MarketDataValidationError("Irregular spacing", issues=["irregular cadence"])
            ),
        )
        label = detect_frequency(pd.DataFrame(index=index))

    assert "irregular" in label.lower()


def test_read_uploaded_file_without_name() -> None:
    buffer = io.BytesIO(b"Date,A\n2023-01-31,0.1\n")
    frame, source = _read_uploaded_file(buffer)
    assert source == "uploaded file"
    assert "A" in frame.columns
