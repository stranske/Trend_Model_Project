from __future__ import annotations

import sys
from types import SimpleNamespace

import pandas as pd
import pytest

from trend_analysis.config.validation import ValidationError, ValidationResult


def test_execute_analysis_raises_on_invalid_config(monkeypatch) -> None:
    stub = SimpleNamespace()
    stub.session_state = {}
    stub.cache_data = lambda *args, **kwargs: (
        args[0] if args and callable(args[0]) else (lambda fn: fn)
    )
    stub.cache_resource = stub.cache_data

    monkeypatch.setitem(sys.modules, "streamlit", stub)

    from streamlit_app.components import analysis_runner
    import trend_analysis.api as api

    issue = ValidationError(
        path="data.csv_path",
        message="Data source is required.",
        expected="csv_path or managers_glob",
        actual="missing",
        suggestion="Set data.csv_path to a CSV file or data.managers_glob to a CSV glob.",
    )

    monkeypatch.setattr(
        analysis_runner,
        "validate_config",
        lambda config_payload: ValidationResult(valid=False, errors=[issue], warnings=[]),
    )
    monkeypatch.setattr(
        analysis_runner,
        "format_validation_messages",
        lambda result: [
            "data.csv_path: Data source is required. Expected csv_path or managers_glob, "
            "got missing. Suggestion: Set data.csv_path to a CSV file or data.managers_glob "
            "to a CSV glob."
        ],
    )
    monkeypatch.setattr(
        api,
        "run_simulation",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("run should not run")),
    )

    returns = pd.DataFrame(
        {"FundA": [0.01]},
        index=pd.to_datetime(["2020-01-31"]),
    )
    payload = analysis_runner.AnalysisPayload(
        returns=returns,
        model_state={},
        benchmark=None,
    )

    with pytest.raises(ValueError, match="Config validation failed") as exc:
        analysis_runner._execute_analysis(payload)

    message = str(exc.value)
    assert "data.csv_path" in message
    assert "Expected" in message
    assert "Suggestion" in message
