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
    plan = prepare_dry_run_plan(df, lookback_periods=12)
    assert plan.frame.index.is_monotonic_increasing
    assert plan.in_start <= plan.in_end < plan.out_start <= plan.out_end


def test_validate_startup_payload_round_trip(tmp_path):
    csv_path = tmp_path / "data.csv"
    frame = pd.DataFrame(
        {"Date": pd.date_range("2022-01-31", periods=12, freq="ME"), "A": 0.01}
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


def test_validate_startup_payload_delegates_to_validator(tmp_path, monkeypatch):
    csv_path = tmp_path / "data.csv"
    frame = pd.DataFrame(
        {"Date": pd.date_range("2022-01-31", periods=8, freq="ME"), "A": 0.02}
    )
    frame.to_csv(csv_path, index=False)

    captured: dict[str, object] = {}

    def fake_validate(payload, *, base_path):
        captured["payload"] = payload
        captured["base_path"] = base_path
        return ({**payload, "validated": True}, None)

    monkeypatch.setattr(
        "streamlit_app.components.guardrails.validate_payload", fake_validate
    )

    validated, errors = validate_startup_payload(
        csv_path=str(csv_path),
        date_column="Date",
        risk_target=0.2,
        timestamps=pd.to_datetime(frame["Date"]),
    )

    assert errors == []
    assert validated is not None and validated.get("validated") is True
    assert captured["base_path"] == csv_path.parent
    assert captured["payload"]  # sanity check payload returned
    # mypy: captured["payload"] is stored as object; assert dict shape before indexing
    assert isinstance(captured["payload"], dict)
    assert captured["payload"]["data"]["csv_path"] == str(csv_path)


def test_validate_startup_payload_surfaces_validator_errors(tmp_path, monkeypatch):
    csv_path = tmp_path / "data.csv"
    frame = pd.DataFrame(
        {"Date": pd.date_range("2022-01-31", periods=8, freq="ME"), "A": 0.02}
    )
    frame.to_csv(csv_path, index=False)

    def fake_validate(payload, *, base_path):
        return None, "vol_adjust -> target_vol\n must be greater than zero"

    monkeypatch.setattr(
        "streamlit_app.components.guardrails.validate_payload", fake_validate
    )

    validated, errors = validate_startup_payload(
        csv_path=str(csv_path),
        date_column="Date",
        risk_target=0.2,
        timestamps=pd.to_datetime(frame["Date"]),
    )

    assert validated is None
    assert "vol_adjust" in errors[0]
    assert "greater than zero" in errors[1]
