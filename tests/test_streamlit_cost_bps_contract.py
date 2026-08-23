from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize("field", ("per_trade_bps", "half_spread_bps"))
@pytest.mark.parametrize("value", (0.5, 1.4, 2.5, 9.9))
def test_run_path_preserves_the_validated_cost_bps(monkeypatch, field, value):
    """The run configuration must retain every fractional bps value validation accepts."""

    streamlit_stub = SimpleNamespace(session_state={})
    streamlit_stub.cache_data = lambda *args, **kwargs: (
        args[0] if args and callable(args[0]) else (lambda fn: fn)
    )
    streamlit_stub.cache_resource = streamlit_stub.cache_data
    monkeypatch.setitem(sys.modules, "streamlit", streamlit_stub)

    from streamlit_app.components.analysis_runner import (
        _build_portfolio_config,
        _coerce_positive_float,
    )
    from trend_analysis.config.ui_mapping import build_portfolio_config

    model_state = {field: value}
    expected = _coerce_positive_float(value, default=0.0)

    assert _build_portfolio_config(model_state, {})["cost_model"][field] == expected
    assert build_portfolio_config(model_state, {})["cost_model"][field] == expected
