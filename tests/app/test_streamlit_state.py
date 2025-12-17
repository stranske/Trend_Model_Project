from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from streamlit_app import state


@pytest.fixture
def session_state(monkeypatch: pytest.MonkeyPatch) -> dict:
    """Provide an isolated fake Streamlit session_state mapping."""

    store: dict[str, object] = {}
    monkeypatch.setattr(state.st, "session_state", store)
    return store


def test_initialize_session_state_preserves_existing_values(
    session_state: dict,
) -> None:
    session_state["returns_df"] = "already-set"

    state.initialize_session_state()

    assert session_state["returns_df"] == "already-set"
    assert session_state["schema_meta"] is None
    assert session_state["benchmark_candidates"] == []
    assert session_state["validation_report"] is None
    assert session_state["upload_status"] == "pending"
    assert session_state["data_hash"] is None
    assert session_state["data_saved_path"] is None
    assert session_state["saved_model_states"] == {}


def test_clear_upload_data_removes_payload_but_resets_status(
    session_state: dict,
) -> None:
    session_state.update(
        {
            "returns_df": pd.DataFrame({"x": [1]}),
            "schema_meta": {"frequency": "Monthly"},
            "benchmark_candidates": ["Bench"],
            "validation_report": {"ok": True},
            "upload_status": "success",
            "analysis_result": "cached",
            "analysis_result_key": "key",
            "analysis_error": {"message": "boom"},
            "other": "keep-me",
            "data_hash": "abc",
            "data_saved_path": "/tmp/data.csv",
            "data_loaded_key": "sample::foo",
            "data_fingerprint": "abc",
            "data_summary": "summary",
            "uploaded_file_path": "foo",
        }
    )

    state.clear_upload_data()

    for key in [
        "returns_df",
        "schema_meta",
        "benchmark_candidates",
        "validation_report",
        "data_hash",
        "data_saved_path",
        "data_loaded_key",
        "data_fingerprint",
        "data_summary",
        "uploaded_file_path",
    ]:
        assert key not in session_state
    assert session_state["upload_status"] == "pending"
    assert session_state["other"] == "keep-me"
    for key in ["analysis_result", "analysis_result_key", "analysis_error"]:
        assert key not in session_state


def test_clear_upload_data_without_existing_payload(session_state: dict) -> None:
    state.clear_upload_data()

    assert session_state["upload_status"] == "pending"
    assert session_state.get("returns_df") is None


def test_store_and_read_validated_data_updates_state(session_state: dict) -> None:
    df = pd.DataFrame(
        {"value": [0.1, 0.2]},
        index=pd.to_datetime(["2024-01-31", "2024-02-29"]),
    )
    meta = {"frequency": "Monthly", "validation": {"issues": [], "warnings": []}}

    session_state.update(
        {
            "analysis_result": "stale",
            "analysis_result_key": "old",
            "analysis_error": {"message": "old"},
            "data_hash": "old",
            "data_saved_path": "old",
        }
    )

    state.store_validated_data(
        df, meta, data_hash="hash", saved_path=Path("/tmp/data.csv")
    )

    stored_df, stored_meta = state.get_uploaded_data()
    assert stored_df is df
    assert stored_meta is meta
    assert state.has_valid_upload()
    assert session_state["validation_report"] == meta["validation"]
    assert session_state["data_hash"] == "hash"
    assert session_state["data_saved_path"] == str(Path("/tmp/data.csv"))
    for key in ["analysis_result", "analysis_result_key", "analysis_error"]:
        assert key not in session_state


def test_has_valid_upload_requires_success_status(session_state: dict) -> None:
    session_state.update({"returns_df": pd.DataFrame(), "schema_meta": {}})
    session_state["upload_status"] = "error"

    assert not state.has_valid_upload()


@pytest.mark.parametrize(
    "meta, expected_suffix",
    [({}, ""), ({"frequency": "Weekly"}, " | Frequency: Weekly")],
)
def test_get_upload_summary_formats_output(
    session_state: dict, meta: dict, expected_suffix: str
) -> None:
    df = pd.DataFrame(
        {"value": [1.0, 2.0, 3.0]},
        index=pd.to_datetime(["2024-01-31", "2024-02-29", "2024-03-31"]),
    )
    summary_without_data = state.get_upload_summary()
    assert summary_without_data == "No data uploaded"

    state.store_validated_data(df, meta)
    summary = state.get_upload_summary()
    base = "3 rows × 1 columns | Range: 2024-01-31 to 2024-03-31"
    assert summary == f"{base}{expected_suffix}"


def test_record_upload_error_sets_error_state(session_state: dict) -> None:
    session_state.update(
        {
            "analysis_result": "cached",
            "analysis_result_key": "key",
            "analysis_error": {"message": "old"},
        }
    )

    state.record_upload_error("problem")

    assert session_state["returns_df"] is None
    assert session_state["schema_meta"] is None
    assert session_state["benchmark_candidates"] == []
    assert session_state["validation_report"] == {
        "message": "problem",
        "issues": [],
    }
    assert session_state["upload_status"] == "error"
    assert session_state["data_hash"] is None
    assert session_state["data_saved_path"] is None
    for key in ["analysis_result", "analysis_result_key", "analysis_error"]:
        assert key not in session_state


def test_record_upload_error_records_detail(session_state: dict) -> None:
    state.record_upload_error("problem", ["issue"], detail="raw message")

    assert session_state["validation_report"] == {
        "message": "problem",
        "issues": ["issue"],
        "detail": "raw message",
    }


def test_clear_analysis_results_removes_cached_outputs(session_state: dict) -> None:
    session_state.update(
        {
            "analysis_result": "cached",
            "analysis_result_key": "key",
            "analysis_error": {"message": "problem"},
            "other": "stay",
        }
    )

    state.clear_analysis_results()

    for key in ["analysis_result", "analysis_result_key", "analysis_error"]:
        assert key not in session_state
    assert session_state["other"] == "stay"


def test_save_and_load_model_state_round_trip(session_state: dict) -> None:
    state.initialize_session_state()
    payload = {
        "lookback_periods": 3,
        "metric_weights": {"sharpe": 1.0},
        "flag": True,
    }

    state.save_model_state("alpha", payload)
    payload["lookback_periods"] = 99

    loaded = state.load_saved_model_state("alpha")
    assert loaded["lookback_periods"] == 3
    loaded["metric_weights"]["sharpe"] = 9.0

    reloaded = state.load_saved_model_state("alpha")
    assert reloaded["metric_weights"]["sharpe"] == 1.0

    state.save_model_state("beta", {"lookback_periods": 5})
    assert set(state.get_saved_model_states()) == {"alpha", "beta"}


def test_rename_and_delete_saved_model_state(session_state: dict) -> None:
    state.initialize_session_state()
    state.save_model_state("alpha", {"lookback_periods": 3})
    state.save_model_state("beta", {"lookback_periods": 6})

    state.rename_saved_model_state("beta", "gamma")
    assert "beta" not in state.get_saved_model_states()
    assert state.load_saved_model_state("gamma") == {"lookback_periods": 6}

    state.delete_saved_model_state("alpha")
    assert "alpha" not in state.get_saved_model_states()
    assert "gamma" in state.get_saved_model_states()


def test_export_import_round_trip_preserves_types(session_state: dict) -> None:
    state.initialize_session_state()
    original = {
        "lookback_periods": 3,
        "selection_count": 10,
        "metric_weights": {"sharpe": 1.0, "drawdown": 0.5},
        "risk_target": 0.15,
        "long_only": True,
    }

    state.save_model_state("source", original)
    exported = state.export_model_state("source")

    imported = state.import_model_state("copy", exported)
    assert imported == state.load_saved_model_state("source")
    assert isinstance(imported["selection_count"], int)
    assert isinstance(imported["long_only"], bool)


def test_save_model_state_rejects_empty_name(session_state: dict) -> None:
    state.initialize_session_state()

    with pytest.raises(ValueError):
        state.save_model_state("   ", {"lookback_periods": 1})


def test_loading_nonexistent_model_state_raises_key_error(session_state: dict) -> None:
    state.initialize_session_state()

    with pytest.raises(KeyError):
        state.load_saved_model_state("missing")


def test_renaming_validates_existence_and_duplicate_names(session_state: dict) -> None:
    state.initialize_session_state()
    state.save_model_state("alpha", {"lookback_periods": 1})

    with pytest.raises(KeyError):
        state.rename_saved_model_state("beta", "gamma")

    state.save_model_state("beta", {"lookback_periods": 2})
    with pytest.raises(ValueError):
        state.rename_saved_model_state("alpha", " beta ")


def test_export_model_state_requires_existing_entry(session_state: dict) -> None:
    state.initialize_session_state()

    with pytest.raises(KeyError):
        state.export_model_state("missing")


def test_import_model_state_validates_payload(session_state: dict) -> None:
    state.initialize_session_state()

    with pytest.raises(ValueError):
        state.import_model_state("name", "{not json")

    with pytest.raises(ValueError):
        state.import_model_state("name", "[]")
