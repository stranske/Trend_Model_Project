"""Additional coverage for ``trend_analysis.util.missing`` helpers."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from trend_analysis.data import load_csv
from trend_analysis.io.market_data import (
    MarketDataValidationError,
    _normalise_policy_value,
)
from trend_analysis.util import missing


@pytest.fixture
def sample_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "A": [1.0, None, None],
            "B": [1.0, 1.0, 1.0],
        },
        index=pd.date_range("2024-01-31", periods=3, freq="ME"),
    )


def _make_result(**overrides: object) -> missing.MissingPolicyResult:
    base = dict(
        policy={"A": "drop"},
        default_policy="drop",
        limit={"A": None},
        default_limit=None,
        filled={"A": 0},
        dropped_assets=("B",),
        summary="default=drop(unbounded)",
    )
    base.update(overrides)
    return missing.MissingPolicyResult(**base)  # type: ignore[arg-type]


def test_missing_policy_result_get_returns_default() -> None:
    result = _make_result()

    assert result.get("policy_map") == {"A": "drop"}
    assert result.get("missing", "fallback") == "fallback"


@pytest.mark.parametrize(
    "value",
    ["invalid", "bfill", "backfill", "both", "zero_fill", "zeros", "fillzero"],
)
def test_coerce_policy_rejects_unknown(value: str | None) -> None:
    with pytest.raises(ValueError, match="Unsupported missing-data policy"):
        missing._coerce_policy(value)


@pytest.mark.parametrize("value", ["", None])
def test_coerce_policy_defaults_empty_to_drop(value: str | None) -> None:
    assert missing._coerce_policy(value) == "drop"


@pytest.mark.parametrize("value", ["bfill", "backfill", "both", "zero_fill"])
def test_missing_policy_layers_reject_undocumented_aliases(value: str) -> None:
    with pytest.raises(ValueError, match="Unsupported missing-data policy"):
        missing._coerce_policy(value)

    with pytest.raises(ValueError, match="Unknown missing-data policy"):
        _normalise_policy_value(value)


@pytest.mark.parametrize("value", ["bfill", "backfill", "both", "zero_fill", "zeros"])
def test_load_csv_rejects_undocumented_missing_policy_aliases(tmp_path: Path, value: str) -> None:
    csv_path = tmp_path / "returns.csv"
    csv_path.write_text(
        "Date,Fund\n2024-01-31,1.0\n2024-02-29,\n",
        encoding="utf-8",
    )

    with pytest.raises(MarketDataValidationError, match="Unknown missing-data policy"):
        load_csv(str(csv_path), errors="raise", missing_policy=value)


@pytest.mark.parametrize("limit", [-1, -5])
def test_coerce_limit_rejects_negative(limit: int) -> None:
    with pytest.raises(ValueError, match="Forward-fill limit must be non-negative"):
        missing._coerce_limit(limit)


def test_resolve_mapping_with_overrides() -> None:
    default, overrides = missing._resolve_mapping(
        {"default": "ffill", "A": "zero", "B": "drop"},
        "drop",
    )

    assert default == "ffill"
    assert overrides == {"A": "zero", "B": "drop"}


def test_resolve_limits_with_overrides() -> None:
    default, overrides = missing._resolve_limits(
        {"default": 2, "A": 1, "B": None},
    )

    assert default == 2
    assert overrides == {"A": 1, "B": None}


def test_policy_display_with_limit_overrides() -> None:
    text = missing._policy_display(
        "ffill",
        {"A": "ffill", "B": "zero"},
        2,
        {"A": 1},
    )

    assert "default=ffill(limit=2)" in text
    assert "A=ffill(limit=1)" in text
    assert "B=zero" in text


def test_apply_missing_policy_ffill_drops_incomplete_column(
    sample_frame: pd.DataFrame,
) -> None:
    cleaned, result = missing.apply_missing_policy(
        sample_frame,
        {"A": "ffill", "B": "drop"},
        limit={"A": 1},
    )

    assert list(cleaned.columns) == ["B"]
    assert result.dropped_assets == ("A",)
    assert result.policy["A"] == "ffill"
    assert result.limit["A"] == 1


def test_apply_missing_policy_ffill_retains_when_not_enforcing(
    sample_frame: pd.DataFrame,
) -> None:
    cleaned, result = missing.apply_missing_policy(
        sample_frame,
        {"A": "ffill"},
        limit={"A": 1},
        enforce_completeness=False,
    )

    assert "A" in cleaned.columns
    assert cleaned["A"].isna().any()
    assert result.policy["A"] == "ffill"
    assert result.limit["A"] == 1


def test_apply_missing_policy_completes_leading_gap_in_single_column() -> None:
    frame = pd.DataFrame({"A": [None, 1.0, 2.0]})

    cleaned, result = missing.apply_missing_policy(frame, "ffill")

    assert cleaned["A"].tolist() == [1.0, 1.0, 2.0]
    assert result.dropped_assets == ()


def test_apply_missing_policy_normalizes_mapping_keys_and_preserves_limits() -> None:
    frame = pd.DataFrame({1: [None, 0.02], "B": [0.01, 0.02]})

    cleaned, result = missing.apply_missing_policy(
        frame,
        {"1": "zero", "B": "drop"},
        limit={"1": 2, "B": 3},
    )

    assert cleaned.loc[0, 1] == 0.0
    assert result.policy == {"1": "zero", "B": "drop"}
    assert result.limit == {"1": 2, "B": 3}


def test_apply_missing_policy_guard_for_unhandled_policy(
    sample_frame: pd.DataFrame, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_resolve_mapping(policy: object, default: str) -> tuple[str, dict[str, str]]:
        return "drop", {"A": "mystery"}

    monkeypatch.setattr(missing, "_resolve_mapping", fake_resolve_mapping)

    with pytest.raises(AssertionError, match="Unhandled policy"):
        missing.apply_missing_policy(sample_frame, "drop")


def test_wildcard_policy_is_shared_by_market_data_validation() -> None:
    from trend_analysis.io.market_data import validate_market_data

    frame = pd.DataFrame(
        {
            "Date": ["2024-01-31", "2024-02-29", "2024-03-31"],
            "A": [0.01, None, 0.03],
        }
    )

    validated = validate_market_data(frame, missing_policy={"*": "zero"})

    assert validated.frame.loc[pd.Timestamp("2024-02-29"), "A"] == 0.0
