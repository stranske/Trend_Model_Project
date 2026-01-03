from __future__ import annotations

import warnings
from pathlib import Path

from trend_analysis.config import bridge as core_bridge


def _payload_inputs(csv_path: str) -> dict[str, object]:
    return {
        "csv_path": csv_path,
        "universe_membership_path": None,
        "managers_glob": None,
        "date_column": "Date",
        "frequency": "M",
        "rebalance_calendar": "NYSE",
        "max_turnover": 0.5,
        "transaction_cost_bps": 5.0,
        "target_vol": 0.1,
    }


def test_config_bridge_payload_types_match(tmp_path: Path) -> None:
    csv = tmp_path / "returns.csv"
    csv.write_text("Date,A\n2020-01-31,0.1\n", encoding="utf-8")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        from streamlit_app import config_bridge as streamlit_bridge

    payload_core = core_bridge.build_config_payload(
        **_payload_inputs(str(csv)),
    )
    payload_streamlit = streamlit_bridge.build_config_payload(
        **_payload_inputs(str(csv)),
    )

    assert type(payload_streamlit) is type(payload_core)
    assert payload_streamlit == payload_core

    validated_core, error_core = core_bridge.validate_payload(
        payload_core, base_path=tmp_path
    )
    validated_streamlit, error_streamlit = streamlit_bridge.validate_payload(
        payload_streamlit, base_path=tmp_path
    )

    assert type(validated_streamlit) is type(validated_core)
    assert type(error_streamlit) is type(error_core)
    assert validated_streamlit == validated_core
    assert error_streamlit == error_core
