from __future__ import annotations

from pathlib import Path

from trend_analysis.config.bridge import build_config_payload, validate_payload


def _payload(tmp_path: Path, **overrides: object) -> dict[str, object]:
    csv_path = tmp_path / "returns.csv"
    csv_path.write_text("Date,A\n2024-01-31,0.01\n", encoding="utf-8")
    payload = build_config_payload(
        csv_path=str(csv_path),
        universe_membership_path=None,
        managers_glob=None,
        date_column="Date",
        frequency="ME",
        rebalance_calendar="NYSE",
        max_turnover=0.5,
        transaction_cost_bps=0.0,
        slippage_bps=0.0,
        target_vol=0.1,
    )
    for section, values in overrides.items():
        if isinstance(values, dict):
            payload.setdefault(section, {}).update(values)  # type: ignore[union-attr]
        else:
            payload[section] = values
    return payload


def test_rejects_negative_target_vol(tmp_path: Path) -> None:
    payload = _payload(tmp_path, vol_adjust={"target_vol": -0.1})

    validated, error = validate_payload(payload, base_path=tmp_path)

    assert validated is None
    assert error is not None
    assert "vol_adjust.target_vol" in error


def test_rejects_oversized_max_turnover(tmp_path: Path) -> None:
    payload = _payload(tmp_path, portfolio={"max_turnover": 5.0})

    validated, error = validate_payload(payload, base_path=tmp_path)

    assert validated is None
    assert error is not None
    assert "portfolio.max_turnover" in error


def test_accepts_valid_demo_payload(tmp_path: Path) -> None:
    payload = _payload(tmp_path, portfolio={"max_turnover": "0.75"})

    validated, error = validate_payload(payload, base_path=tmp_path)

    assert error is None
    assert validated is not None
    assert validated["portfolio"]["max_turnover"] == 0.75
    assert validated["vol_adjust"]["target_vol"] == 0.1
