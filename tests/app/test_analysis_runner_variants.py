from __future__ import annotations

import sys
from types import SimpleNamespace

import pandas as pd
import pytest


def _install_streamlit_stub(monkeypatch) -> None:
    stub = SimpleNamespace()
    stub.session_state = {}
    stub.cache_data = lambda *args, **kwargs: (
        args[0] if args and callable(args[0]) else (lambda fn: fn)
    )
    stub.cache_resource = stub.cache_data
    monkeypatch.setitem(sys.modules, "streamlit", stub)


def test_apply_variant_patch_wraps_errors(monkeypatch) -> None:
    _install_streamlit_stub(monkeypatch)

    from streamlit_app.components import analysis_runner
    from trend_analysis.config.patch import ConfigPatch

    monkeypatch.setattr(
        analysis_runner,
        "apply_config_patch",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(ValueError("boom")),
    )

    returns = pd.DataFrame(
        {"FundA": [0.01]},
        index=pd.to_datetime(["2020-01-31"]),
    )
    patch = ConfigPatch(operations=[], summary="noop")

    with pytest.raises(analysis_runner.VariantRunError) as exc:
        analysis_runner.apply_variant_patch(
            {"lookback_periods": 3},
            patch,
            returns=returns,
            benchmark=None,
            label="baseline",
        )

    assert exc.value.label == "baseline"
    assert exc.value.stage == "patch application"
    assert "boom" in str(exc.value)


def test_run_variant_analysis_returns_results(monkeypatch) -> None:
    _install_streamlit_stub(monkeypatch)

    from streamlit_app.components import analysis_runner
    from trend_analysis.config.patch import ConfigPatch

    monkeypatch.setattr(analysis_runner, "_validate_streamlit_payload", lambda *_: None)
    monkeypatch.setattr(analysis_runner, "run_analysis", lambda *_args, **_kwargs: {"ok": True})

    returns = pd.DataFrame(
        {"FundA": [0.01]},
        index=pd.to_datetime(["2020-01-31"]),
    )
    variants = [
        SimpleNamespace(label="conservative", patch=ConfigPatch(operations=[], summary="c")),
        SimpleNamespace(label="baseline", patch=ConfigPatch(operations=[], summary="b")),
        SimpleNamespace(label="aggressive", patch=ConfigPatch(operations=[], summary="a")),
    ]

    results = analysis_runner.run_variant_analysis(returns, {"lookback_periods": 3}, None, variants)

    assert list(results.keys()) == ["conservative", "baseline", "aggressive"]
    assert all(value == {"ok": True} for value in results.values())
