import pandas as pd

from streamlit_app.components.guardrails import (
    estimate_resource_usage,
    infer_frequency,
    prepare_dry_run_plan,
    validate_startup_payload,
)


def test_infer_frequency_detects_daily():
    idx = pd.date_range("2024-01-01", periods=5, freq="D")
    assert infer_frequency(idx) == "D"


def test_estimate_resource_usage_warns_on_large_dataset():
    estimate = estimate_resource_usage(200_000, 250)
    assert estimate.rows == 200_000
    assert any("512 MB" in msg or "five minutes" in msg for msg in estimate.warnings)


def test_prepare_dry_run_plan_builds_monotonic_windows():
    periods = pd.period_range("2020-01", periods=24, freq="M")
    df = pd.DataFrame(
        {"A": range(len(periods)), "B": range(len(periods))},
        index=periods.to_timestamp("M", "end"),
    )
    plan = prepare_dry_run_plan(df, lookback_months=12)
    assert plan.frame.index.is_monotonic_increasing
    assert plan.in_start <= plan.in_end < plan.out_start <= plan.out_end


def test_validate_startup_payload_round_trip(tmp_path):
    csv_path = tmp_path / "data.csv"
    frame = pd.DataFrame(
        {"Date": pd.date_range("2022-01-31", periods=12, freq="M"), "A": 0.01}
    )
    frame.to_csv(csv_path, index=False)
    validated, errors = validate_startup_payload(
        csv_path=str(csv_path),
        date_column="Date",
        risk_target=0.1,
        timestamps=pd.to_datetime(frame["Date"]),
    )
    assert errors == []
    assert validated is not None


def test_validate_startup_payload_requires_csv():
    validated, errors = validate_startup_payload(
        csv_path=None,
        date_column="Date",
        risk_target=0.1,
        timestamps=pd.date_range("2020-01-31", periods=6, freq="ME"),
    )
    assert validated is None
    assert errors and "Upload" in errors[0]
