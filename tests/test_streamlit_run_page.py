"""Tests for the Streamlit run page hosted under ``streamlit_app``."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

SRC_PATH = Path(__file__).parent.parent / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

pytestmark = pytest.mark.filterwarnings("ignore:Could not infer format.*:UserWarning")

RUN_PAGE_PATH = Path(__file__).parent.parent / "streamlit_app" / "pages" / "3_Run.py"


def _ctx_mock() -> MagicMock:
    ctx = MagicMock()
    ctx.__enter__.return_value = ctx
    ctx.__exit__.return_value = False
    return ctx


def create_mock_streamlit() -> MagicMock:
    mock_st = MagicMock()
    mock_st.session_state = {}
    mock_st.title = MagicMock()
    mock_st.button = MagicMock(return_value=False)
    mock_st.error = MagicMock()
    mock_st.warning = MagicMock(side_effect=lambda *args, **kwargs: _ctx_mock())
    mock_st.caption = MagicMock()
    mock_st.success = MagicMock()
    mock_st.write = MagicMock()
    mock_st.json = MagicMock()
    mock_st.checkbox = MagicMock(return_value=True)
    mock_st.modal = MagicMock(return_value=_ctx_mock())
    mock_st.markdown = MagicMock()
    mock_st.rerun = MagicMock()
    mock_st.progress = MagicMock(return_value=SimpleNamespace(progress=MagicMock()))
    mock_st.spinner = MagicMock(return_value=_ctx_mock())
    mock_st.expander = MagicMock(return_value=_ctx_mock())
    mock_st.code = MagicMock()
    mock_st.info = MagicMock()
    return mock_st


def _spec_from_path(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        raise AssertionError(f"Unable to load spec for {module_name}")
    return spec


def _exec_module(spec, module: ModuleType) -> None:
    if spec.loader is None:  # pragma: no cover - defensive
        raise AssertionError(f"Spec loader unavailable for {module.__name__}")
    spec.loader.exec_module(module)


def load_run_page_module(mock_st: MagicMock | None = None):
    if mock_st is None:
        mock_st = create_mock_streamlit()

    spec = _spec_from_path("run_page_module", RUN_PAGE_PATH)
    run_page = importlib.util.module_from_spec(spec)

    with patch.dict("sys.modules", {"streamlit": mock_st}):
        _exec_module(spec, run_page)

    return run_page, mock_st


@pytest.fixture()
def run_page_module():
    module, mock_st = load_run_page_module()
    return module, mock_st


def test_format_error_message_hints_for_known_exceptions(run_page_module):
    run_page, _ = run_page_module
    message = run_page.format_error_message(KeyError("missing_field"))
    assert "Missing required data field" in message
    assert "missing_field" in message


def test_format_error_message_handles_sample_split(run_page_module):
    run_page, _ = run_page_module
    message = run_page.format_error_message(ValueError("sample_split invalid"))
    assert "Invalid date ranges" in message
    assert "in-sample" in message


def test_coerce_positive_int_fallbacks(run_page_module):
    run_page, _ = run_page_module
    assert run_page._coerce_positive_int("5", default=3) == 5
    assert run_page._coerce_positive_int(None, default=2, minimum=4) == 4
    assert run_page._coerce_positive_int("bad", default=1, minimum=1) == 1


def test_infer_date_bounds_sorts_index(run_page_module):
    run_page, _ = run_page_module
    idx = pd.to_datetime(["2023-04-30", "2023-02-28", "2023-03-31"])
    df = pd.DataFrame({"val": [1, 2, 3]}, index=idx)
    start, end = run_page._infer_date_bounds(df)
    assert str(start.date()) == "2023-02-28"
    assert str(end.date()) == "2023-04-30"


def test_config_from_model_state_generates_payload(run_page_module):
    run_page, _ = run_page_module
    idx = pd.date_range("2022-01-31", periods=24, freq="ME")
    df = pd.DataFrame({"A": range(len(idx))}, index=idx)
    model_state = {
        "evaluation_months": 6,
        "lookback_months": 12,
        "risk_target": 0.2,
        "weighting_scheme": "risk",
        "trend_spec": {"kind": "ema", "window": 9},
    }
    cfg = run_page._config_from_model_state(model_state, df)
    assert cfg["lookback_months"] == 12
    assert cfg["evaluation_months"] == 6
    assert cfg["portfolio"]["weighting_scheme"] == "risk"
    assert cfg["signals"]["kind"] == "ema"
    assert cfg["start"] < cfg["end"]


def test_config_from_model_state_rejects_invalid_index(run_page_module):
    run_page, _ = run_page_module
    df = pd.DataFrame({"A": [1, 2, 3]}, index=["foo", "bar", "baz"])
    with pytest.raises(ValueError):
        run_page._config_from_model_state({}, df)


def test_main_requires_data_and_config(run_page_module):
    run_page, mock_st = run_page_module
    mock_st.button.side_effect = [False, False]
    mock_st.session_state.clear()
    with (
        patch.object(run_page, "show_disclaimer", return_value=True),
        patch.dict("sys.modules", {"streamlit": mock_st}),
    ):
        run_page.main()
    mock_st.error.assert_called_with("Upload data and set configuration first.")


def test_main_dry_run_flow_populates_session_state(run_page_module):
    run_page, mock_st = run_page_module
    idx = pd.date_range("2020-01-31", periods=18, freq="ME")
    returns_df = pd.DataFrame({"fund": 0.01}, index=idx)
    mock_st.session_state = {
        "returns_df": returns_df,
        "sim_config": {
            "lookback_months": 6,
            "start": idx[-6],
            "end": idx[-1],
            "portfolio": {"weighting_scheme": "equal"},
        },
    }

    class DummyPlan:
        def __init__(self):
            self.frame = returns_df.iloc[:6]

        def sample_split(self) -> dict[str, str]:
            return {
                "in_start": "2019-01",
                "in_end": "2019-06",
                "out_start": "2019-07",
                "out_end": "2019-09",
            }

        def summary(self) -> dict[str, int]:
            return {"rows": int(self.frame.shape[0])}

    mock_run = MagicMock(return_value={"status": "ok"})
    estimate = SimpleNamespace(
        approx_memory_mb=10.0, estimated_runtime_s=30.0, warnings=()
    )

    with (
        patch.object(run_page, "show_disclaimer", return_value=True),
        patch.object(run_page, "prepare_dry_run_plan", return_value=DummyPlan()),
        patch.object(run_page, "estimate_resource_usage", return_value=estimate),
        patch.object(run_page, "_resolve_run_simulation", return_value=mock_run),
        patch.dict("sys.modules", {"streamlit": mock_st}),
    ):
        mock_st.button.side_effect = [True, False]
        run_page.main()

    assert mock_run.called
    assert mock_st.session_state["dry_run_results"] == {"status": "ok"}
    assert mock_st.session_state["dry_run_summary"] == {"rows": 6}


def test_main_full_run_executes_simulation(run_page_module, tmp_path):
    run_page, mock_st = run_page_module
    idx = pd.date_range("2021-01-31", periods=36, freq="ME")
    returns_df = pd.DataFrame({"fund": 0.01}, index=idx)
    config = {
        "lookback_months": 12,
        "risk_target": 0.15,
        "start": idx[-12],
        "end": idx[-1],
        "portfolio": {"weighting_scheme": "equal"},
    }
    mock_st.session_state = {"returns_df": returns_df, "sim_config": config}
    estimate = SimpleNamespace(
        approx_memory_mb=20.0, estimated_runtime_s=60.0, warnings=()
    )
    fake_result = SimpleNamespace(
        metrics=pd.DataFrame({"Sharpe": [1.0]}), fallback_info=None
    )
    mock_run = MagicMock(return_value=fake_result)
    fake_logger = SimpleNamespace(info=MagicMock())

    with (
        patch.object(run_page, "show_disclaimer", return_value=True),
        patch.object(run_page, "estimate_resource_usage", return_value=estimate),
        patch.object(run_page, "_resolve_run_simulation", return_value=mock_run),
        patch.object(run_page, "init_run_logger", return_value=fake_logger),
        patch.object(
            run_page, "get_default_log_path", return_value=tmp_path / "log.jsonl"
        ),
        patch.object(run_page, "log_step"),
        patch.dict("sys.modules", {"streamlit": mock_st}),
    ):
        mock_st.button.side_effect = [False, True]
        run_page.main()

    assert mock_run.called
    assert mock_st.session_state["sim_results"] is fake_result
    assert "run_log_path" in mock_st.session_state
    mock_st.success.assert_called()
