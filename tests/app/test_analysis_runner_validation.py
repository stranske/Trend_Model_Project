from __future__ import annotations

import sys
from types import SimpleNamespace

import pandas as pd
import pytest


def test_execute_analysis_raises_on_invalid_config(monkeypatch) -> None:
    stub = SimpleNamespace()
    stub.session_state = {}
    stub.cache_data = lambda *args, **kwargs: (
        args[0] if args and callable(args[0]) else (lambda fn: fn)
    )
    stub.cache_resource = stub.cache_data

    monkeypatch.setitem(sys.modules, "streamlit", stub)

    import trend_analysis.api as api
    from streamlit_app.components import analysis_runner

    monkeypatch.setattr(
        analysis_runner,
        "build_config_payload",
        lambda **_kwargs: {},
    )
    monkeypatch.setattr(
        analysis_runner,
        "validate_payload",
        lambda _payload, *, base_path: (
            None,
            "data.csv_path must point to the returns CSV file",
        ),
    )
    monkeypatch.setattr(
        api,
        "run_simulation",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("run should not run")
        ),
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
