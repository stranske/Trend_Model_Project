"""Targeted tests for the Streamlit demo runner helpers."""

from __future__ import annotations

import sys
from types import SimpleNamespace
from typing import Any, cast

import pandas as pd
import pytest

from streamlit_app.components import demo_runner, disclaimer
from trend_analysis.config import Config


@pytest.fixture()
def sample_returns() -> pd.DataFrame:
    """Build a small month-end return frame used by multiple tests."""

    dates = pd.date_range("2020-01-31", periods=6, freq="ME")
    data = {
        "Alpha": [0.01, 0.02, 0.015, -0.01, 0.012, 0.018],
        "Beta": [0.005, 0.007, 0.01, 0.008, 0.006, 0.009],
        "SPX Index": [0.004, 0.006, 0.005, 0.002, 0.007, 0.009],
    }
    return pd.DataFrame(data, index=dates)


def test_normalise_metric_weights_alias_handling() -> None:
    """Metric aliases and non-numeric values should be normalised correctly."""

    raw = {
        "Sharpe_Ratio": "2",  # alias should map to ``sharpe``
        "annual_return": 1,
        "volatility": "abc",  # ignored because it cannot be coerced to float
        "mystery": 5,  # ignored because it is not a recognised alias
    }

    weights = demo_runner._normalise_metric_weights(raw)

    assert sum(weights.values()) == pytest.approx(1.0)
    assert set(weights) == {"sharpe", "return_ann"}
    assert weights["sharpe"] == pytest.approx(2 / 3)
    assert weights["return_ann"] == pytest.approx(1 / 3)


def test_normalise_metric_weights_default_when_empty() -> None:
    """If no usable weights are provided the balanced fallback is returned."""

    assert demo_runner._normalise_metric_weights({}) == {
        "sharpe": pytest.approx(1 / 3),
        "return_ann": pytest.approx(1 / 3),
        "drawdown": pytest.approx(1 / 3),
    }


def test_select_benchmark_prefers_spx(sample_returns: pd.DataFrame) -> None:
    """The helper should prioritise SPX-style benchmarks when present."""

    choice = demo_runner._select_benchmark(sample_returns.columns)
    assert choice == "SPX Index"


def test_select_benchmark_falls_back_to_first_candidate() -> None:
    """When no SPX candidate exists the first inferred benchmark is chosen."""

    columns = ["Alpha", "Global Bond", "CoreAgg"]
    choice = demo_runner._select_benchmark(columns)
    assert choice in {"Global Bond", "CoreAgg"}


def test_select_benchmark_returns_none_for_absent_candidates() -> None:
    """If no benchmark-like columns exist the helper should return ``None``."""

    assert demo_runner._select_benchmark(["Alpha", "Beta"]) is None


def test_derive_window_caps_start_bounds(sample_returns: pd.DataFrame) -> None:
    """Month-end windows should respect the available history and OOS
    length."""

    start, end = demo_runner._derive_window(sample_returns, lookback_periods=48)

    # With a long lookback the function should clamp start to the data's end
    assert end == pd.Timestamp("2020-06-30 23:59:59.999999999")
    assert start == end


def test_derive_window_respects_earliest_bound() -> None:
    """The rolling window should advance start to the earliest allowed
    value."""

    dates = pd.date_range("2022-01-31", periods=12, freq="ME")
    df = pd.DataFrame({"Alpha": range(12)}, index=dates)

    start, end = demo_runner._derive_window(df, lookback_periods=10)

    assert start == pd.Timestamp("2022-11-30 23:59:59.999999999")
    assert end == pd.Timestamp("2022-12-31 23:59:59.999999999")


def test_derive_window_handles_single_period() -> None:
    """A single observation should collapse start to match the end
    timestamp."""

    dates = pd.date_range("2020-01-31", periods=1, freq="ME")
    df = pd.DataFrame({"Alpha": [0.1]}, index=dates)

    start, end = demo_runner._derive_window(df, lookback_periods=1)

    assert start == end
    assert end == pd.Timestamp("2020-01-31 23:59:59.999999999")


def test_derive_window_no_adjustment_needed() -> None:
    """When sufficient history exists the original window should be
    retained."""

    dates = pd.date_range("2020-01-31", periods=24, freq="ME")
    df = pd.DataFrame({"Alpha": range(24)}, index=dates)

    start, end = demo_runner._derive_window(df, lookback_periods=6)

    assert start == pd.Timestamp("2021-01-31 23:59:59.999999999")
    assert end == pd.Timestamp("2021-12-31 23:59:59.999999999")


def test_build_policy_uses_metric_weights() -> None:
    """Weights are passed through into ``PolicyConfig`` metric specs."""

    preset = {
        "selection_count": 8,
        "min_track_months": 18,
        "portfolio": {"cooldown_months": 2, "max_weight": 0.2},
    }

    policy = demo_runner._build_policy({"sharpe": 0.6, "return_ann": 0.4}, preset)

    assert policy.top_k == 8
    assert policy.cooldown_months == 2
    assert policy.metrics[0].name == "sharpe"
    assert policy.metrics[0].weight == pytest.approx(0.6)
    assert policy.metrics[1].name == "return_ann"


def test_build_pipeline_config_translates_weights() -> None:
    """Pipeline config should translate UI aliases into metric registry
    names."""

    sim_cfg = {
        "start": "2020-03-31",
        "end": "2020-06-30",
        "lookback_periods": 3,
        "policy": {"top_k": 5},
        "portfolio": {"weighting_scheme": "equal"},
        "risk_target": 0.12,
    }
    weights = {"sharpe": 0.5, "return_ann": 0.3, "drawdown": 0.2}

    config = demo_runner._build_pipeline_config(sim_cfg, weights, benchmark="SPX")

    dumped = config.model_dump()
    assert dumped["metrics"]["registry"] == ["Sharpe", "AnnualReturn", "MaxDrawdown"]
    assert dumped["portfolio"]["rank"]["blended_weights"] == {
        "Sharpe": 0.5,
        "AnnualReturn": 0.3,
        "MaxDrawdown": 0.2,
    }
    assert dumped["benchmarks"] == {"SPX": "SPX"}
    assert dumped["vol_adjust"] == {
        "target_vol": 0.12,
        "floor_vol": 0.015,
        "warmup_periods": 0,
    }


def test_prepare_demo_setup_builds_consistent_state(
    sample_returns: pd.DataFrame, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``_prepare_demo_setup`` should construct matching configs for the
    app."""

    preset = {
        "metrics": {"sharpe": 0.7, "return_ann": 0.2, "drawdown": 0.1},
        "lookback_periods": 3,
        "selection_count": 6,
        "min_track_months": 12,
        "risk_target": 0.11,
    }

    monkeypatch.setattr(demo_runner, "DEFAULT_PRESET", "Balanced")
    monkeypatch.setattr(demo_runner, "_load_preset", lambda _: preset)

    setup = demo_runner._prepare_demo_setup(sample_returns)

    assert setup.benchmark == "SPX Index"
    assert setup.config_state["preset_name"] == "Balanced"
    assert setup.sim_config["lookback_periods"] == 3
    assert setup.sim_config["policy"]["top_k"] == 6
    assert setup.pipeline_config.sample_split["out_end"] == "2020-06"


def test_update_session_state_populates_streamlit_state(
    sample_returns: pd.DataFrame,
) -> None:
    """Ensure the Streamlit session state is updated with demo metadata."""

    class DummySt:
        def __init__(self) -> None:
            self.session_state: dict[str, Any] = {}

    setup = demo_runner.DemoSetup(
        config_state={"preset_name": "Balanced"},
        sim_config={"preset_name": "Balanced"},
        pipeline_config=cast(Config, SimpleNamespace()),
        benchmark="SPX Index",
    )
    meta = demo_runner.SchemaMeta(
        n_rows=len(sample_returns), original_columns=list(sample_returns.columns)
    )

    st_module = DummySt()
    demo_runner._update_session_state(st_module, setup, sample_returns, meta)

    assert st_module.session_state["returns_df"].equals(sample_returns)
    assert st_module.session_state["schema_meta"] == meta
    assert st_module.session_state["demo_last_run"]["rows"] == len(sample_returns)
    assert st_module.session_state["sim_config"]["preset_name"] == "Balanced"


def test_load_demo_returns_reads_first_candidate(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Existing demo data files should be loaded via the validator helper."""

    candidate = tmp_path / "demo.csv"
    candidate.write_text("dummy")
    monkeypatch.setattr(demo_runner, "DEMO_DATA_CANDIDATES", (candidate,))

    df = pd.DataFrame(
        {"Date": pd.date_range("2024-01-31", periods=2, freq="ME"), "Fund": [0.1, 0.2]}
    ).set_index("Date")
    meta = demo_runner.SchemaMeta(n_rows=2)

    monkeypatch.setattr(demo_runner, "load_and_validate_file", lambda handle: (df, meta))

    loaded_df, loaded_meta = demo_runner._load_demo_returns()
    assert loaded_df is df
    assert loaded_meta is meta


def test_load_demo_returns_raises_when_missing(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Missing demo files should trigger a clear FileNotFoundError."""

    monkeypatch.setattr(demo_runner, "DEMO_DATA_CANDIDATES", (tmp_path / "missing.csv",))

    with pytest.raises(FileNotFoundError):
        demo_runner._load_demo_returns()


def test_load_preset_reads_yaml(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """Preset loading should parse YAML content when available."""

    preset_dir = tmp_path / "presets"
    preset_dir.mkdir()
    preset_path = preset_dir / "balanced.yml"
    preset_path.write_text("metrics:\n  sharpe: 1.0\n")

    monkeypatch.setattr(demo_runner, "PRESET_DIR", preset_dir)

    data = demo_runner._load_preset("Balanced")
    assert data == {"metrics": {"sharpe": 1.0}}


def test_load_preset_returns_empty_for_missing(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Absent preset files should return an empty mapping."""

    monkeypatch.setattr(demo_runner, "PRESET_DIR", tmp_path)

    assert demo_runner._load_preset("Missing") == {}


def test_run_one_click_demo_happy_path(monkeypatch: pytest.MonkeyPatch) -> None:
    """The demo runner should orchestrate loading, running and state
    updates."""

    df = pd.DataFrame(
        {
            "Date": pd.date_range("2020-01-31", periods=3, freq="ME"),
            "Fund": [0.01, 0.02, 0.03],
        }
    ).set_index("Date")
    meta = demo_runner.SchemaMeta(n_rows=3)

    pipeline_config = demo_runner.Config(
        version="1",
        data={},
        preprocessing={},
        vol_adjust={"target_vol": 0.1},
        sample_split={
            "in_start": "2019-10",
            "in_end": "2019-12",
            "out_start": "2020-01",
            "out_end": "2020-03",
        },
        portfolio={"selection_mode": "rank", "rank": {}, "weighting_scheme": "equal"},
        benchmarks={},
        metrics={"registry": []},
        export={},
        run={"monthly_cost": 0.0},
        seed=1,
    )
    setup = demo_runner.DemoSetup(
        config_state={"preset_name": "Balanced"},
        sim_config={"preset_name": "Balanced"},
        pipeline_config=pipeline_config,
        benchmark=None,
    )

    class DummySt:
        def __init__(self) -> None:
            self.session_state: dict[str, Any] = {}
            self.errors: list[str] = []

        def error(self, message: str) -> None:  # pragma: no cover - defensive
            self.errors.append(message)

    st_module = DummySt()
    sentinel_result = object()

    monkeypatch.setattr(demo_runner, "_load_demo_returns", lambda: (df, meta))
    monkeypatch.setattr(demo_runner, "_prepare_demo_setup", lambda _: setup)

    captured = {}

    def fake_run_simulation(config: demo_runner.Config, returns: pd.DataFrame) -> object:
        captured["config"] = config
        captured["returns"] = returns
        return sentinel_result

    monkeypatch.setattr(demo_runner, "run_simulation", fake_run_simulation)

    assert demo_runner.run_one_click_demo(st_module) is True

    assert captured["config"] is pipeline_config
    assert captured["returns"].columns.tolist() == ["Date", "Fund"]
    assert st_module.session_state["sim_results"] is sentinel_result
    assert st_module.session_state["demo_show_export_prompt"] is True


def test_run_one_click_demo_handles_simulation_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Failures during the simulation should surface as user-facing errors."""

    df = pd.DataFrame(
        {
            "Date": pd.date_range("2020-01-31", periods=2, freq="ME"),
            "Fund": [0.01, 0.02],
        }
    ).set_index("Date")
    meta = demo_runner.SchemaMeta(n_rows=2)
    setup = demo_runner.DemoSetup(
        config_state={},
        sim_config={},
        pipeline_config=SimpleNamespace(),
        benchmark=None,
    )

    class DummySt:
        def __init__(self) -> None:
            self.session_state: dict[str, Any] = {}
            self.errors: list[str] = []

        def error(self, message: str) -> None:
            self.errors.append(message)

    st_module = DummySt()

    monkeypatch.setattr(demo_runner, "_load_demo_returns", lambda: (df, meta))
    monkeypatch.setattr(demo_runner, "_prepare_demo_setup", lambda _: setup)
    monkeypatch.setattr(
        demo_runner,
        "run_simulation",
        lambda *_: (_ for _ in ()).throw(RuntimeError("boom")),
    )

    assert demo_runner.run_one_click_demo(st_module) is False
    assert st_module.errors and "boom" in st_module.errors[0]


class ModalStub:
    """Context manager stub returned by the fake Streamlit modal."""

    def __init__(self, owner: StreamlitStub, title: str) -> None:
        self.owner = owner
        self.title = title

    def __enter__(self) -> ModalStub:  # pragma: no cover - trivial
        self.owner.modal_calls.append(self.title)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # pragma: no cover - trivial
        return None


class StreamlitStub:
    """Minimal stand-in for ``streamlit`` used to test the disclaimer flow."""

    def __init__(self, checkbox_value: bool) -> None:
        self.session_state: dict[str, Any] = {}
        self.checkbox_value = checkbox_value
        self.rerun_called = False
        self.modal_calls: list[str] = []
        self.markdown_calls: list[str] = []
        self.checkbox_args: tuple[str, str] | None = None
        self.errors: list[str] = []

    def modal(self, title: str) -> ModalStub:
        return ModalStub(self, title)

    def markdown(self, message: str) -> None:
        self.markdown_calls.append(message)

    def checkbox(self, label: str, *, key: str) -> bool:
        self.checkbox_args = (label, key)
        return self.checkbox_value

    def rerun(self) -> None:
        self.rerun_called = True

    def error(self, message: str) -> None:
        self.errors.append(message)


def test_show_disclaimer_first_visit(monkeypatch: pytest.MonkeyPatch) -> None:
    """First-time visitors should see the modal and remain unaccepted."""

    stub = StreamlitStub(checkbox_value=False)
    monkeypatch.setattr(disclaimer, "st", stub)

    accepted = disclaimer.show_disclaimer()

    assert accepted is False
    assert stub.session_state["disclaimer_accepted"] is False
    assert stub.modal_calls == ["Disclaimer"]
    assert stub.checkbox_args == ("I understand and accept", "disclaimer_checkbox")
    assert stub.rerun_called is False


def test_show_disclaimer_acceptance(monkeypatch: pytest.MonkeyPatch) -> None:
    """Accepting the disclaimer should persist state and trigger rerun."""

    stub = StreamlitStub(checkbox_value=True)
    monkeypatch.setattr(disclaimer, "st", stub)

    accepted = disclaimer.show_disclaimer()

    assert accepted is True
    assert stub.session_state["disclaimer_accepted"] is True
    assert stub.rerun_called is True

    # Subsequent calls should skip the modal entirely
    stub.checkbox_value = False
    stub.modal_calls.clear()
    accepted_again = disclaimer.show_disclaimer()
    assert accepted_again is True
    assert stub.modal_calls == []


def test_run_one_click_demo_imports_streamlit(monkeypatch: pytest.MonkeyPatch) -> None:
    """Calling the demo runner without a module should import ``streamlit``."""

    df = pd.DataFrame(
        {
            "Date": pd.date_range("2020-01-31", periods=2, freq="ME"),
            "Fund": [0.01, 0.02],
        }
    ).set_index("Date")
    meta = demo_runner.SchemaMeta(n_rows=2)
    pipeline_config = demo_runner.Config(
        version="1",
        data={},
        preprocessing={},
        vol_adjust={"target_vol": 0.1},
        sample_split={
            "in_start": "2019-11",
            "in_end": "2019-12",
            "out_start": "2020-01",
            "out_end": "2020-02",
        },
        portfolio={"selection_mode": "rank", "rank": {}, "weighting_scheme": "equal"},
        benchmarks={},
        metrics={"registry": []},
        export={},
        run={"monthly_cost": 0.0},
        seed=1,
    )
    setup = demo_runner.DemoSetup(
        config_state={},
        sim_config={},
        pipeline_config=pipeline_config,
        benchmark=None,
    )

    sentinel = object()
    stub = StreamlitStub(checkbox_value=False)

    monkeypatch.setattr(demo_runner, "_load_demo_returns", lambda: (df, meta))
    monkeypatch.setattr(demo_runner, "_prepare_demo_setup", lambda _: setup)
    monkeypatch.setattr(demo_runner, "run_simulation", lambda *_: sentinel)

    monkeypatch.setitem(sys.modules, "streamlit", stub)
    try:
        assert demo_runner.run_one_click_demo() is True
        assert stub.session_state["sim_results"] is sentinel
    finally:
        monkeypatch.delitem(sys.modules, "streamlit", raising=False)
