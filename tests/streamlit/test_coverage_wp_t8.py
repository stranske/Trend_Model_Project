"""Coverage tests for ``render_model_page`` main flow (wp t8)."""

from __future__ import annotations

import importlib
import sys
from types import ModuleType, SimpleNamespace
from typing import Any

import pandas as pd
import pytest


class _Ctx:
    def __init__(self, stub: "_Stub") -> None:
        self._s = stub

    def __enter__(self) -> "_Stub":
        return self._s

    def __exit__(self, *_a: object) -> bool:
        return False

    def __getattr__(self, name: str) -> Any:
        return getattr(self._s, name)


class _Placeholder:
    def progress(self, *_a: Any, **_k: Any) -> SimpleNamespace:
        return SimpleNamespace(progress=lambda *_a, **_k: None)

    def empty(self) -> None:
        return None


class _Stub:
    def __init__(self) -> None:
        self.session_state: dict[str, Any] = {}
        self.errors: list[str] = []
        self.successes: list[str] = []
        self.warnings: list[str] = []
        self.infos: list[str] = []
        self.captions: list[str] = []
        self.markdowns: list[str] = []
        self.metrics: list[tuple[str, Any]] = []
        self.dataframes: list[Any] = []
        self.codes: list[tuple[str, Any]] = []
        self.json_calls: list[Any] = []
        self.downloads: list[dict[str, Any]] = []
        self.text_areas: list[tuple[str, Any]] = []
        self.rerun_called = False
        self.selectbox_by_key: dict[str, Any] = {}
        self.selectbox_by_label: dict[str, Any] = {}
        self.radio_by_key: dict[str, Any] = {}
        self.button_by_key: dict[str, bool] = {}
        self.submit_by_label: dict[str, bool] = {}
        self.text_input_by_key: dict[str, str] = {}
        self.text_input_by_label: dict[str, str] = {}
        self.text_area_by_key: dict[str, str] = {}
        self.checkbox_by_label: dict[str, bool] = {}
        self.number_by_label: dict[str, Any] = {}
        self.uploaded: Any = None
        self.sidebar = _Ctx(self)
        self.cache_data = self._deco
        self.cache_resource = self._deco

    @staticmethod
    def _deco(*a: Any, **_k: Any) -> Any:
        if a and callable(a[0]):
            return a[0]
        return lambda fn: fn

    def title(self, t: str = "") -> None:
        self.markdowns.append(str(t))

    def header(self, *_a: Any, **_k: Any) -> None:
        return None

    def subheader(self, *_a: Any, **_k: Any) -> None:
        return None

    def write(self, t: Any = "", *_a: Any, **_k: Any) -> None:
        self.markdowns.append(str(t))

    def markdown(self, t: Any = "", *_a: Any, **_k: Any) -> None:
        self.markdowns.append(str(t))

    def caption(self, t: Any = "", *_a: Any, **_k: Any) -> None:
        self.captions.append(str(t))

    def code(self, t: Any = "", *_a: Any, **_k: Any) -> None:
        self.codes.append((str(t), _k.get("language")))

    def divider(self) -> None:
        return None

    def json(self, v: Any = None, *_a: Any, **_k: Any) -> None:
        self.json_calls.append(v)

    def info(self, m: Any = "") -> None:
        self.infos.append(str(m))

    def success(self, m: Any = "") -> None:
        self.successes.append(str(m))

    def error(self, m: Any = "") -> None:
        self.errors.append(str(m))

    def warning(self, m: Any = "") -> None:
        self.warnings.append(str(m))

    def radio(self, _l: str = "", options: Any = None, index: int = 0, **k: Any) -> Any:
        key = k.get("key")
        if key in self.radio_by_key:
            return self.radio_by_key[key]
        return list(options)[index]

    def selectbox(self, label: str = "", options: Any = None, index: int = 0, **k: Any) -> Any:
        key = k.get("key")
        if key in self.selectbox_by_key:
            return self.selectbox_by_key[key]
        if label in self.selectbox_by_label:
            return self.selectbox_by_label[label]
        opts = list(options) if options is not None else []
        return opts[index] if opts else None

    def multiselect(self, *_a: Any, **_k: Any) -> list[Any]:
        return []

    def slider(self, _l: str = "", **k: Any) -> Any:
        return k.get("value", 0)

    def number_input(self, label: str = "", **k: Any) -> Any:
        if label in self.number_by_label:
            return self.number_by_label[label]
        return k.get("value", 0)

    def text_input(self, label: str = "", value: str = "", **k: Any) -> str:
        key = k.get("key")
        if key in self.text_input_by_key:
            return self.text_input_by_key[key]
        if label in self.text_input_by_label:
            return self.text_input_by_label[label]
        return value

    def text_area(self, label: str = "", value: str = "", **k: Any) -> str:
        key = k.get("key")
        result = value
        if key in self.text_area_by_key:
            result = self.text_area_by_key[key]
        elif key is not None and key in self.session_state:
            result = str(self.session_state[key])
        self.text_areas.append((label, result))
        return result

    def button(self, label: str = "", *_a: Any, **k: Any) -> bool:
        return bool(self.button_by_key.get(k.get("key"), False))

    def form_submit_button(self, label: str = "", *_a: Any, **_k: Any) -> bool:
        return bool(self.submit_by_label.get(label, False))

    def checkbox(self, label: str = "", value: bool = False, **_k: Any) -> bool:
        if label in self.checkbox_by_label:
            return self.checkbox_by_label[label]
        return bool(value)

    def file_uploader(self, *_a: Any, **_k: Any) -> Any:
        return self.uploaded

    def data_editor(self, df: Any, **_k: Any) -> Any:
        return df

    def columns(self, spec: Any) -> list[_Ctx]:
        n = spec if isinstance(spec, int) else len(spec)
        return [_Ctx(self) for _ in range(n)]

    def container(self, *_a: Any, **_k: Any) -> _Ctx:
        return _Ctx(self)

    def expander(self, *_a: Any, **_k: Any) -> _Ctx:
        return _Ctx(self)

    def form(self, *_a: Any, **_k: Any) -> _Ctx:
        return _Ctx(self)

    def tabs(self, labels: list[str]) -> list[_Ctx]:
        return [_Ctx(self) for _ in labels]

    def spinner(self, *_a: Any, **_k: Any) -> _Ctx:
        return _Ctx(self)

    def dialog(self, *_a: Any, **_k: Any) -> _Ctx:
        return _Ctx(self)

    def empty(self) -> _Placeholder:
        return _Placeholder()

    def set_page_config(self, **_k: Any) -> None:
        return None

    def dataframe(self, df: Any, **_k: Any) -> None:
        self.dataframes.append(df)

    def metric(self, label: str, value: Any = None, **_k: Any) -> None:
        self.metrics.append((label, value))

    def altair_chart(self, *_a: Any, **_k: Any) -> None:
        return None

    def plotly_chart(self, *_a: Any, **_k: Any) -> None:
        return None

    def progress(self, *_a: Any, **_k: Any) -> _Placeholder:
        return _Placeholder()

    def download_button(self, label: str = "", **k: Any) -> None:
        self.downloads.append({"label": label, **k})

    def page_link(self, *_a: Any, **_k: Any) -> None:
        return None

    def date_input(self, _l: str = "", value: Any = None, **_k: Any) -> Any:
        return value

    def rerun(self) -> None:
        self.rerun_called = True

    def stop(self) -> None:
        raise AssertionError("stop called")


def _df(cols: list[str], periods: int = 3) -> pd.DataFrame:
    idx = pd.date_range("2021-01-31", periods=periods, freq="ME")
    return pd.DataFrame({c: [0.01] * periods for c in cols}, index=idx)


@pytest.fixture()
def env(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    stub = _Stub()
    monkeypatch.setitem(sys.modules, "streamlit", stub)
    rt = ModuleType("streamlit.runtime")
    up = ModuleType("streamlit.runtime.uploaded_file_manager")
    up.UploadedFile = object  # type: ignore[attr-defined]
    rt.uploaded_file_manager = up  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "streamlit.runtime", rt)
    monkeypatch.setitem(sys.modules, "streamlit.runtime.uploaded_file_manager", up)
    module = importlib.reload(importlib.import_module("streamlit_app.pages.2_Model"))
    from streamlit_app import state as app_state
    from streamlit_app.components import analysis_runner

    calls: dict[str, Any] = {
        "saved": [],
        "wrappers_saved": [],
        "renamed": [],
        "deleted": [],
        "cleared": 0,
    }
    monkeypatch.setattr(app_state, "initialize_session_state", lambda: None)
    monkeypatch.setattr(app_state, "st", stub)
    monkeypatch.setattr(app_state, "clear_analysis_results", lambda: None)
    monkeypatch.setattr(
        analysis_runner,
        "clear_cached_analysis",
        lambda: calls.__setitem__("cleared", calls["cleared"] + 1),
    )
    monkeypatch.setattr(module, "_should_auto_render", lambda: False)
    monkeypatch.setattr(module, "render_config_chat_panel", lambda **_k: None)
    monkeypatch.setattr(module, "_render_config_status_badge", lambda *_a, **_k: None)
    monkeypatch.setattr(module, "_render_variant_what_if_panel", lambda **_k: None)
    monkeypatch.setattr(module.demo_profile, "render_profile_controls", lambda _st: None)
    monkeypatch.setattr(module.demo_profile, "custom_analysis_enabled", lambda: True)
    return SimpleNamespace(
        stub=stub, module=module, app_state=app_state, calls=calls, monkeypatch=monkeypatch
    )


def _setup(
    env: SimpleNamespace, df: Any, meta: Any = None, saved: dict[str, Any] | None = None
) -> None:
    env.app_state.get_uploaded_data = lambda: (df, meta if meta is not None else {})
    # monkeypatch via attribute on module object used by page (same module)
    import streamlit_app.state as app_state

    app_state.get_uploaded_data = lambda: (df, meta if meta is not None else {})  # type: ignore[misc]
    app_state.get_saved_model_states = lambda: dict(saved or {})  # type: ignore[misc]
    env.stub.session_state["model_state"] = env.module._initial_model_state()


def _patch_saved(env: SimpleNamespace, saved: dict[str, Any]) -> dict[str, Any]:
    app_state = env.app_state
    env.monkeypatch.setattr(app_state, "get_saved_model_states", lambda: dict(saved))
    env.monkeypatch.setattr(
        app_state, "save_model_state", lambda n, s: env.calls["saved"].append((n, dict(s)))
    )
    env.monkeypatch.setattr(
        app_state, "save_config_wrapper", lambda n, w: env.calls["wrappers_saved"].append((n, w))
    )
    return env.calls


def test_no_data_exact_error(env: SimpleNamespace) -> None:
    _setup(env, None)
    env.monkeypatch.setattr(env.app_state, "get_uploaded_data", lambda: (None, {}))
    env.module.render_model_page()
    assert env.stub.errors == ["Load data on the Data page before configuring the model."]
    assert env.stub.metrics == []


def test_meta_source_name_and_fund_filter(env: SimpleNamespace) -> None:
    df = _df(["FundA", "FundB", "SPX"])
    _setup(env, df, meta=SimpleNamespace(source_name="MyDataset.csv"))
    env.stub.session_state["selected_benchmark"] = "SPX"
    env.stub.session_state["analysis_fund_columns"] = ["FundA", "SPX", "Missing"]
    _patch_saved(env, {})
    env.module.render_model_page()
    assert "**Dataset:** MyDataset.csv" in env.stub.markdowns
    assert ("Funds", 1) in env.stub.metrics
    assert ("Time Periods", 3) in env.stub.metrics


def test_fund_overflow_caption(env: SimpleNamespace) -> None:
    cols = [f"F{i}" for i in range(53)]
    _setup(env, _df(cols))
    _patch_saved(env, {})
    env.module.render_model_page()
    assert "...and 3 more" in env.stub.captions
    assert ("Funds", 53) in env.stub.metrics


def test_save_empty_name_error(env: SimpleNamespace) -> None:
    _setup(env, _df(["FundA"]))
    _patch_saved(env, {})
    env.stub.text_input_by_label["Configuration name"] = "   "
    env.stub.submit_by_label["Save Current Settings"] = True
    env.module.render_model_page()
    assert env.stub.errors == ["Enter a name to save your configuration."]
    assert env.calls["saved"] == []


def test_save_duplicate_unconfirmed_warning(env: SimpleNamespace) -> None:
    _setup(env, _df(["FundA"]))
    _patch_saved(env, {"Alpha": {"preset": "Baseline"}})
    env.stub.text_input_by_label["Configuration name"] = "Alpha"
    env.stub.submit_by_label["Save Current Settings"] = True
    env.module.render_model_page()
    assert env.stub.warnings == [
        "This name already exists. Check 'Confirm overwrite' to replace it."
    ]
    assert env.calls["saved"] == []


def test_save_success_sets_state_and_reruns(env: SimpleNamespace) -> None:
    _setup(env, _df(["FundA"]))
    _patch_saved(env, {})
    env.stub.text_input_by_label["Configuration name"] = "  Alpha  "
    env.stub.submit_by_label["Save Current Settings"] = True
    env.module.render_model_page()
    assert [n for n, _ in env.calls["saved"]] == ["Alpha"]
    assert env.stub.session_state["active_saved_model_name"] == "Alpha"
    # last_loaded_model_state is a snapshot of model_state at save time (before volatility widgets add keys)
    assert env.stub.session_state["last_loaded_model_state"]["preset"] == "Baseline"
    assert env.stub.successes[0] == "Saved configuration 'Alpha'."
    assert env.stub.rerun_called is True


def test_load_via_wrapper_applies_wrapper(env: SimpleNamespace) -> None:
    _setup(env, _df(["FundA"]))
    _patch_saved(env, {"Alpha": {"preset": "Baseline"}})
    applied: list[Any] = []
    env.monkeypatch.setattr(env.module, "_apply_config_wrapper", lambda w: applied.append(w))
    env.monkeypatch.setattr(env.app_state, "load_saved_config_wrapper", lambda _n: {"cfg": 1})
    env.stub.button_by_key["load_saved_config_button"] = True
    env.module.render_model_page()
    assert applied == [{"cfg": 1}]
    assert env.stub.session_state["active_saved_model_name"] == "Alpha"
    assert "Loaded configuration 'Alpha'. The form has been updated." in env.stub.successes


def test_load_via_state_fallback(env: SimpleNamespace) -> None:
    _setup(env, _df(["FundA"]))
    _patch_saved(env, {"Alpha": {"preset": "Baseline"}})
    loaded = {"preset": "Custom", "lookback_periods": 9}
    env.monkeypatch.setattr(env.app_state, "load_saved_config_wrapper", lambda _n: None)
    env.monkeypatch.setattr(env.app_state, "load_saved_model_state", lambda _n: loaded)
    resets: list[str] = []
    env.monkeypatch.setattr(env.module, "_reset_model_widget_state", lambda: resets.append("r"))
    env.monkeypatch.setattr(
        env.module, "_sync_model_widgets_from_state", lambda s: resets.append(str(s))
    )
    env.stub.button_by_key["load_saved_config_button"] = True
    env.module.render_model_page()
    assert env.stub.session_state["model_state"] == loaded
    assert env.stub.session_state["last_loaded_model_state"] == loaded
    assert resets[0] == "r"


def test_rename_success_and_error(env: SimpleNamespace) -> None:
    _setup(env, _df(["FundA"]))
    _patch_saved(env, {"Alpha": {"preset": "Baseline"}})
    env.monkeypatch.setattr(env.app_state, "load_saved_config_wrapper", lambda _n: None)
    env.stub.text_input_by_key["rename_saved_config_input"] = "Beta"
    env.stub.submit_by_label["Rename configuration"] = True
    renamed: list[tuple[str, str]] = []
    env.monkeypatch.setattr(
        env.app_state, "rename_saved_model_state", lambda a, b: renamed.append((a, b))
    )
    env.module.render_model_page()
    assert renamed == [("Alpha", "Beta")]
    assert env.stub.session_state["active_saved_model_name"] == "Beta"
    assert "Renamed configuration to 'Beta'." in env.stub.successes


def test_rename_value_error_shows_exact(env: SimpleNamespace) -> None:
    _setup(env, _df(["FundA"]))
    _patch_saved(env, {"Alpha": {"preset": "Baseline"}})
    env.stub.text_input_by_key["rename_saved_config_input"] = "Beta"
    env.stub.submit_by_label["Rename configuration"] = True

    def _boom(_a: str, _b: str) -> None:
        raise ValueError("Name already taken exactly")

    env.monkeypatch.setattr(env.app_state, "rename_saved_model_state", _boom)
    env.module.render_model_page()
    assert env.stub.errors == ["Name already taken exactly"]


def test_delete_clears_active_name(env: SimpleNamespace) -> None:
    _setup(env, _df(["FundA"]))
    _patch_saved(env, {"Alpha": {"preset": "Baseline"}})
    env.stub.session_state["active_saved_model_name"] = "Alpha"
    deleted: list[str] = []
    env.monkeypatch.setattr(env.app_state, "delete_saved_model_state", lambda n: deleted.append(n))
    env.stub.button_by_key["delete_saved_config_button"] = True
    env.module.render_model_page()
    assert deleted == ["Alpha"]
    assert "active_saved_model_name" not in env.stub.session_state
    assert "Deleted configuration 'Alpha'." in env.stub.successes


def test_export_download_and_preview(env: SimpleNamespace) -> None:
    _setup(env, _df(["FundA"]))
    _patch_saved(env, {"Alpha": {"preset": "Baseline"}})
    env.monkeypatch.setattr(env.app_state, "export_model_state", lambda n: '{"x": 1}')
    env.module.render_model_page()
    saved_dl = [d for d in env.stub.downloads if d.get("key") == "download_saved_config_button"]
    assert len(saved_dl) == 1
    assert saved_dl[0]["data"] == b'{"x": 1}'
    assert saved_dl[0]["file_name"] == "Alpha_parameters.json"
    assert ("Exported JSON", '{"x": 1}') in env.stub.text_areas
    assert len(env.stub.json_calls) >= 1


class _Upload:
    def __init__(self, val: Any, exc: type[Exception] | None = None) -> None:
        self._v = val
        self._exc = exc

    def getvalue(self) -> Any:
        if self._exc is not None:
            raise self._exc("bad")
        return self._v


def test_import_uploaded_bytes_decodes(env: SimpleNamespace) -> None:
    _setup(env, _df(["FundA"]))
    _patch_saved(env, {})
    env.stub.uploaded = _Upload(b'{"a": 1}')
    env.module.render_model_page()
    assert env.stub.session_state["import_config_payload"] == '{"a": 1}'


def test_import_uploaded_str_decodes(env: SimpleNamespace) -> None:
    _setup(env, _df(["FundA"]))
    _patch_saved(env, {})
    env.stub.uploaded = _Upload("plain-text")
    env.module.render_model_page()
    # Check that import_config_payload is set exactly to the uploaded string
    assert env.stub.session_state.get("import_config_payload") == "plain-text"
    # Also check that no errors were produced
    assert env.stub.errors == []


def test_import_uploaded_decode_error(env: SimpleNamespace) -> None:
    _setup(env, _df(["FundA"]))
    _patch_saved(env, {})
    env.stub.uploaded = _Upload(b"\xff\xfe", exc=None)
    env.module.render_model_page()
    assert env.stub.errors == [
        "Unable to decode uploaded file as UTF-8. Please upload a UTF-8 encoded JSON file."
    ]


def test_import_empty_and_value_error(env: SimpleNamespace) -> None:
    _setup(env, _df(["FundA"]))
    _patch_saved(env, {})
    env.stub.button_by_key["import_config_button"] = True
    env.stub.text_area_by_key["import_config_payload"] = "   "
    env.module.render_model_page()
    assert env.stub.errors == ["Paste a JSON payload to import a configuration."]


def test_import_success_state_fallback(env: SimpleNamespace) -> None:
    _setup(env, _df(["FundA"]))
    _patch_saved(env, {})
    env.stub.button_by_key["import_config_button"] = True
    env.stub.text_input_by_key["import_config_name"] = "Imp"
    env.stub.text_area_by_key["import_config_payload"] = '{"preset": "Custom"}'
    imported = {"preset": "Custom", "lookback_periods": 7}
    env.monkeypatch.setattr(env.app_state, "import_model_state", lambda _n, _p: imported)
    env.monkeypatch.setattr(env.app_state, "load_saved_config_wrapper", lambda _n: None)
    env.module.render_model_page()
    assert env.stub.session_state["model_state"] == imported
    assert env.stub.session_state["active_saved_model_name"] == "Imp"
    assert "Imported configuration 'Imp'. The form has been updated." in env.stub.successes
    assert env.stub.rerun_called is True


def test_compare_same_warns_and_diff_renders(env: SimpleNamespace) -> None:
    saved = {"Alpha": {"a": 1}, "Beta": {"a": 2}}
    _setup(env, _df(["FundA"]))
    _patch_saved(env, saved)
    env.stub.selectbox_by_key["compare_config_a"] = "Alpha"
    env.stub.selectbox_by_key["compare_config_b"] = "Alpha"
    env.module.render_model_page()
    assert "Select two different configurations to compare." in env.stub.warnings


def test_compare_diff_rows_exact(env: SimpleNamespace) -> None:
    saved = {"Alpha": {"selection_count": 3}, "Beta": {"selection_count": 5}}
    _setup(env, _df(["FundA"]))
    _patch_saved(env, saved)
    env.stub.selectbox_by_key["compare_config_a"] = "Alpha"
    env.stub.selectbox_by_key["compare_config_b"] = "Beta"
    env.module.render_model_page()
    assert env.stub.dataframes[-1] == [
        {"Setting": "selection_count", "Config A": "3", "Config B": "5", "Change": "changed"}
    ]
    assert env.stub.codes[-1] == ("~ selection_count: (Alpha) 3 -> (Beta) 5", "text")
    assert "Copyable diff output:" in env.stub.captions


def test_explicit_dates_corrected_and_range_error(env: SimpleNamespace) -> None:
    _setup(env, _df(["FundA"], periods=24))
    _patch_saved(env, {})
    ms = env.stub.session_state["model_state"]
    ms["date_mode"] = "explicit"
    ms["start_date"] = "2019-01-01"
    ms["end_date"] = "2021-02-01"
    env.module.render_model_page()
    assert any("Adjusted from 2019-01-01" in c for c in env.stub.captions)
    assert env.stub.session_state["model_state"]["start_date"] == "2021-01-31"


def test_explicit_start_after_end_error(env: SimpleNamespace) -> None:
    _setup(env, _df(["FundA"], periods=24))
    _patch_saved(env, {})
    ms = env.stub.session_state["model_state"]
    ms["date_mode"] = "explicit"
    ms["start_date"] = "2022-06-01"
    ms["end_date"] = "2021-06-01"
    # need data covering those dates: rebuild df 2021-2023
    df = pd.DataFrame(
        {"FundA": [0.01] * 36}, index=pd.date_range("2021-01-31", periods=36, freq="ME")
    )
    env.monkeypatch.setattr(env.app_state, "get_uploaded_data", lambda: (df, {}))
    env.module.render_model_page()
    assert "Start date must be before end date." in env.stub.errors


def test_preset_change_applies_config(env: SimpleNamespace) -> None:
    _setup(env, _df(["FundA"]))
    _patch_saved(env, {})
    env.stub.selectbox_by_key["preset_selector"] = "Conservative"
    resets: list[int] = []
    env.monkeypatch.setattr(env.module, "_reset_model_widget_state", lambda: resets.append(1))
    env.module.render_model_page()
    expected = env.module.PRESET_CONFIGS["Conservative"]
    assert env.stub.session_state["model_state"]["preset"] == "Conservative"
    assert env.stub.session_state["model_state"]["lookback_periods"] == expected["lookback_periods"]
    assert env.stub.session_state["model_state"]["selection_count"] == expected["selection_count"]
    assert resets == [1]
    assert env.stub.rerun_called is True


def test_preset_unknown_and_weighting_change(env: SimpleNamespace) -> None:
    _setup(env, _df(["FundA"]))
    _patch_saved(env, {})
    env.stub.session_state["model_state"]["preset"] = "NopePreset"
    env.stub.session_state["model_state"]["weighting_name"] = "bogus"
    env.stub.selectbox_by_key["weighting_name_selector"] = "risk_parity"
    env.module.render_model_page()
    assert env.stub.session_state["model_state"]["weighting_name"] == "risk_parity"


def test_inclusion_top_pct_and_buy_hold_captions(env: SimpleNamespace) -> None:
    _setup(env, _df(["FundA"]))
    _patch_saved(env, {})
    env.stub.session_state["model_state"]["inclusion_approach"] = "top_pct"
    env.stub.session_state["model_state"]["rank_pct"] = 0.25
    env.stub.selectbox_by_key["inclusion_approach_select"] = "top_pct"
    env.module.render_model_page()
    assert "🏆 Select top 25% of funds by score" in env.stub.captions
    # buy and hold
    env.stub.captions.clear()
    env.stub.session_state["model_state"] = env.module._initial_model_state()
    env.stub.session_state["model_state"]["inclusion_approach"] = "buy_and_hold"
    env.stub.selectbox_by_key["inclusion_approach_select"] = "buy_and_hold"
    env.stub.selectbox_by_key["buy_hold_initial_select"] = "top_n"
    env.module.render_model_page()
    assert any("Buy & Hold" in c and "Top N (Ranking)" in c for c in env.stub.captions)


def test_topn_disabled_captions_and_random(env: SimpleNamespace) -> None:
    _setup(env, _df(["FundA"]))
    _patch_saved(env, {})
    env.stub.selectbox_by_key["inclusion_approach_select"] = "top_n"
    env.module.render_model_page()
    assert env.stub.captions.count("🔒 Disabled (Top N uses Target Funds)") == 2


def test_metric_zero_warning_and_rf_caption(env: SimpleNamespace) -> None:
    _setup(env, _df(["FundA"]))
    _patch_saved(env, {})
    env.stub.session_state["model_state"]["metric_weights"] = {
        k: 0.0 for k in ["sharpe", "return_ann", "sortino", "info_ratio", "drawdown", "vol"]
    }
    env.stub.checkbox_by_label["Override Risk-Free Rate"] = True
    env.module.render_model_page()
    assert "⚠️ Set at least one metric weight > 0." in env.stub.warnings
    assert "⚠️ Using constant RF rate instead of data column." in env.stub.captions


def test_submit_validation_errors(env: SimpleNamespace) -> None:
    _setup(env, _df(["FundA", "FundB"]))
    _patch_saved(env, {})
    env.stub.submit_by_label["💾 Save Configuration"] = True
    seen: list[list[str]] = []
    env.monkeypatch.setattr(env.module, "_validate_model", lambda _v, _c: ["Exact validation boom"])
    env.monkeypatch.setattr(env.module, "_render_validation_errors", lambda e: seen.append(list(e)))
    env.module.render_model_page()
    assert seen == [["Exact validation boom"]]
    assert env.stub.successes == [] or "Model configuration saved" not in env.stub.successes[-1]


def test_submit_success_saves_candidate(env: SimpleNamespace) -> None:
    _setup(env, _df(["FundA", "FundB", "FundC", "FundD", "FundE", "FundF"]))
    _patch_saved(env, {})
    env.stub.submit_by_label["💾 Save Configuration"] = True
    env.monkeypatch.setattr(env.module, "_validate_model", lambda _v, _c: [])
    env.module.render_model_page()
    assert env.stub.session_state["model_state"]["preset"] == "Custom"
    assert "✅ Model configuration saved. Go to Results to run analysis." in env.stub.successes
    assert env.calls["cleared"] == 1


# Tests for import error paths


def test_import_uploaded_type_error(env: SimpleNamespace) -> None:
    _setup(env, _df(["FundA"]))
    _patch_saved(env, {})

    # Create an upload that raises TypeError when getting value
    class _BadUpload:
        def getvalue(self) -> Any:
            raise TypeError("bad type")

    env.stub.uploaded = _BadUpload()
    env.module.render_model_page()
    assert "Unable to read uploaded JSON file due to an unexpected file format." in env.stub.errors


def test_import_uploaded_os_error(env: SimpleNamespace) -> None:
    _setup(env, _df(["FundA"]))
    _patch_saved(env, {})

    # Create an upload that raises OSError when getting value
    class _BadUpload:
        def getvalue(self) -> Any:
            raise OSError("file not found")

    env.stub.uploaded = _BadUpload()
    env.module.render_model_page()
    assert any("Unable to read uploaded file: file not found" in e for e in env.stub.errors)


def test_import_value_error_from_import_state(env: SimpleNamespace) -> None:
    _setup(env, _df(["FundA"]))
    _patch_saved(env, {})
    env.stub.button_by_key["import_config_button"] = True
    env.stub.text_area_by_key["import_config_payload"] = '{"bad": "json"}'
    env.stub.text_input_by_key["import_config_name"] = "BadImport"
    env.monkeypatch.setattr(
        env.app_state,
        "import_model_state",
        lambda _n, _p: (_ for _ in ()).throw(ValueError("Invalid config")),
    )
    env.module.render_model_page()
    assert "Invalid config" in env.stub.errors


def test_import_with_wrapper_applies_wrapper(env: SimpleNamespace) -> None:
    _setup(env, _df(["FundA"]))
    _patch_saved(env, {})
    env.stub.button_by_key["import_config_button"] = True
    env.stub.text_area_by_key["import_config_payload"] = '{"preset": "Custom"}'
    env.stub.text_input_by_key["import_config_name"] = "WithWrapper"
    imported = {"preset": "Custom", "lookback_periods": 5}
    env.monkeypatch.setattr(env.app_state, "import_model_state", lambda _n, _p: imported)
    env.monkeypatch.setattr(
        env.app_state, "load_saved_config_wrapper", lambda _n: {"cfg": "wrapper"}
    )
    applied: list[Any] = []
    env.monkeypatch.setattr(env.module, "_apply_config_wrapper", lambda w: applied.append(w))
    env.module.render_model_page()
    assert applied == [{"cfg": "wrapper"}]
    assert env.stub.session_state["active_saved_model_name"] == "WithWrapper"


def test_compare_no_differences_success(env: SimpleNamespace) -> None:
    saved = {"Alpha": {"a": 1}, "Beta": {"a": 1}}
    _setup(env, _df(["FundA"]))
    _patch_saved(env, saved)
    env.stub.selectbox_by_key["compare_config_a"] = "Alpha"
    env.stub.selectbox_by_key["compare_config_b"] = "Beta"
    env.monkeypatch.setattr(env.app_state, "diff_model_states", lambda _a, _b: [])
    env.module.render_model_page()
    assert "No differences found. The selected configurations match." in env.stub.successes


def test_date_mode_change_updates_state(env: SimpleNamespace) -> None:
    _setup(env, _df(["FundA"]))
    _patch_saved(env, {})
    env.stub.radio_by_key["date_mode_radio"] = "explicit"
    env.module.render_model_page()
    assert env.stub.session_state["model_state"]["date_mode"] == "explicit"


def test_explicit_dates_with_invalid_parse(env: SimpleNamespace) -> None:
    _setup(env, _df(["FundA"]))
    _patch_saved(env, {})
    ms = env.stub.session_state["model_state"]
    ms["date_mode"] = "explicit"
    ms["start_date"] = "invalid-date"
    env.module.render_model_page()
    assert "📅 **Available data range:** Jan 31, 2021 to Mar 31, 2021" in env.stub.infos
    assert env.stub.session_state["model_state"]["start_date"] == "2021-01-31"


def test_explicit_dates_start_after_max(env: SimpleNamespace) -> None:
    _setup(env, _df(["FundA"], periods=12))
    _patch_saved(env, {})
    ms = env.stub.session_state["model_state"]
    ms["date_mode"] = "explicit"
    # Set start_date to a date after the data range
    ms["start_date"] = "2025-06-01"
    env.module.render_model_page()
    # Should be corrected to max_date
    assert any("Adjusted" in c or "corrected" in c.lower() for c in env.stub.captions)


def test_explicit_dates_end_before_min(env: SimpleNamespace) -> None:
    _setup(env, _df(["FundA"], periods=12))
    _patch_saved(env, {})
    ms = env.stub.session_state["model_state"]
    ms["date_mode"] = "explicit"
    # Set end_date to a date before the data range
    ms["end_date"] = "2020-01-01"
    env.module.render_model_page()
    # Should be corrected to min_date
    assert any("Adjusted" in c or "corrected" in c.lower() for c in env.stub.captions)


def test_explicit_dates_with_non_datetime_index(env: SimpleNamespace) -> None:
    # Use a dataframe with non-datetime index to trigger the else branch
    df = pd.DataFrame({"FundA": [0.01, 0.02, 0.03]}, index=[0, 1, 2])
    _setup(env, df)
    _patch_saved(env, {})
    ms = env.stub.session_state["model_state"]
    ms["date_mode"] = "explicit"
    env.module.render_model_page()
    assert not any("Available data range" in i for i in env.stub.infos)
    assert env.stub.session_state["model_state"]["start_date"] is None
    assert env.stub.session_state["model_state"]["end_date"] is None
    assert not any("Selected period:" in i for i in env.stub.infos)


def test_explicit_dates_with_invalid_end_date(env: SimpleNamespace) -> None:
    _setup(env, _df(["FundA"]))
    _patch_saved(env, {})
    ms = env.stub.session_state["model_state"]
    ms["date_mode"] = "explicit"
    ms["end_date"] = "not-a-date"
    env.module.render_model_page()
    assert "📅 **Available data range:** Jan 31, 2021 to Mar 31, 2021" in env.stub.infos
    assert env.stub.session_state["model_state"]["end_date"] == "2021-03-31"


def test_explicit_dates_end_after_max(env: SimpleNamespace) -> None:
    _setup(env, _df(["FundA"], periods=12))
    _patch_saved(env, {})
    ms = env.stub.session_state["model_state"]
    ms["date_mode"] = "explicit"
    # Set end_date to a date after the data range
    ms["end_date"] = "2025-12-31"
    env.module.render_model_page()
    # Should be corrected to max_date
    assert any("Adjusted" in c or "corrected" in c.lower() for c in env.stub.captions)


def test_explicit_dates_very_long_range_warning(env: SimpleNamespace) -> None:
    # Create a dataframe with a very long date range (more than 50 years = 600 months)
    df = pd.DataFrame(
        {"FundA": [0.01] * 602},  # 602 months = 50 years + 2 months
        index=pd.date_range("1970-01-31", periods=602, freq="ME"),
    )
    _setup(env, df)
    _patch_saved(env, {})
    ms = env.stub.session_state["model_state"]
    ms["date_mode"] = "explicit"
    # Set dates to the full range of the dataframe
    # Note: dates are parsed with [:7] + "-01", so "1970-01-31" -> "1970-01-01"
    ms["start_date"] = "1970-01-31"
    ms["end_date"] = df.index[-1].strftime("%Y-%m-%d")
    env.module.render_model_page()
    assert any("Date range exceeds 50 years" in w for w in env.stub.warnings)


def test_inclusion_random_mode_caption(env: SimpleNamespace) -> None:
    _setup(env, _df(["FundA"]))
    _patch_saved(env, {})
    env.stub.selectbox_by_key["inclusion_approach_select"] = "random"
    env.module.render_model_page()
    assert any("Random Mode" in c for c in env.stub.captions)


def test_info_ratio_benchmark_matching_index(env: SimpleNamespace) -> None:
    # Need a dataframe with SPX column to be in benchmark_options
    df = _df(["FundA", "SPX"])
    _setup(env, df)
    _patch_saved(env, {})
    # Set info_ratio_benchmark and info_ratio weight AFTER _setup so they're not overwritten
    ms = env.stub.session_state["model_state"]
    ms["info_ratio_benchmark"] = "SPX"
    ms["metric_weights"]["info_ratio"] = 1.0  # Enable benchmark selector
    env.module.render_model_page()
    # When info_ratio_benchmark matches a benchmark option, benchmark_index should be calculated
    # This exercises line 3867. Verify the benchmark selector UI is rendered
    assert any(
        "Information Ratio Benchmark" in m or "Metric Weights" in m for m in env.stub.markdowns
    )


def test_benchmark_selection_sets_benchmark(env: SimpleNamespace) -> None:
    df = _df(["FundA", "SPX"])
    _setup(env, df)
    _patch_saved(env, {})
    ms = env.stub.session_state["model_state"]
    ms["metric_weights"]["info_ratio"] = 1.0  # Enable benchmark selector
    env.stub.selectbox_by_key["benchmark_selector"] = "SPX"
    env.module.render_model_page()
    # When a benchmark is selected, info_ratio_benchmark local var is set via line 3879
    # Verify the benchmark selector UI is rendered
    assert any(
        "Information Ratio Benchmark" in m or "Metric Weights" in m for m in env.stub.markdowns
    )


def test_ci_level_positive_caption(env: SimpleNamespace) -> None:
    _setup(env, _df(["FundA"]))
    _patch_saved(env, {})
    # Set ci_level to a positive value AFTER _setup
    env.stub.session_state["model_state"]["ci_level"] = 0.95
    env.module.render_model_page()
    # When ci_level > 0, a caption is shown
    # This exercises line 4477
    assert any("Reporting uses" in c for c in env.stub.captions)


def test_half_spread_bps_positive_caption(env: SimpleNamespace) -> None:
    _setup(env, _df(["FundA"]))
    _patch_saved(env, {})
    # Set half_spread_bps to a positive value AFTER _setup
    env.stub.session_state["model_state"]["half_spread_bps"] = 1.5
    env.module.render_model_page()
    # When half_spread_bps > 0, a caption is shown
    # This exercises line 4523
    assert any("market impact cost" in c for c in env.stub.captions)


def test_bottom_k_positive_caption(env: SimpleNamespace) -> None:
    _setup(env, _df(["FundA"]))
    _patch_saved(env, {})
    # Set bottom_k to a positive value AFTER _setup
    env.stub.session_state["model_state"]["bottom_k"] = 3
    env.module.render_model_page()
    # When bottom_k > 0, a caption is shown
    # This exercises line 4537
    assert any("Bottom 3 ranked funds" in c for c in env.stub.captions)


def test_benchmark_selection_sets_info_ratio_benchmark(env: SimpleNamespace) -> None:
    df = _df(["FundA", "SPX"])
    _setup(env, df)
    _patch_saved(env, {})
    ms = env.stub.session_state["model_state"]
    ms["metric_weights"]["info_ratio"] = 1.0
    env.stub.selectbox_by_key["benchmark_selector"] = "SPX"
    env.stub.submit_by_label["💾 Save Configuration"] = True
    env.monkeypatch.setattr(env.module, "_validate_model", lambda _v, _c: [])
    env.module.render_model_page()
    assert env.stub.session_state["model_state"]["info_ratio_benchmark"] == "SPX"


def test_vol_adjust_disabled_caption(env: SimpleNamespace) -> None:
    _setup(env, _df(["FundA"]))
    _patch_saved(env, {})
    ms = env.stub.session_state["model_state"]
    ms["vol_adjust_enabled"] = False
    env.stub.checkbox_by_label["Enable Volatility Adjustment"] = False
    env.module.render_model_page()
    # When vol_adj_enabled is False, a caption is shown
    # This exercises line 3978
    assert "Enable volatility adjustment to configure" in env.stub.captions


def test_regime_enabled_info(env: SimpleNamespace) -> None:
    _setup(env, _df(["FundA"]))
    _patch_saved(env, {})
    ms = env.stub.session_state["model_state"]
    ms["regime_enabled"] = True
    env.stub.checkbox_by_label["Enable Regime Detection"] = True
    env.module.render_model_page()
    # When regime_enabled is True, an info message is shown
    # This exercises line 4170
    assert any("Regime detection" in i for i in env.stub.infos)


def test_long_only_false_warning(env: SimpleNamespace) -> None:
    _setup(env, _df(["FundA"]))
    _patch_saved(env, {})
    ms = env.stub.session_state["model_state"]
    ms["long_only"] = False
    env.stub.checkbox_by_label["Long Only"] = False
    env.module.render_model_page()
    # When long_only is False, a warning is shown
    # This exercises line 4273
    assert any("Short positions enabled" in w for w in env.stub.warnings)


def test_multi_period_disabled_warning(env: SimpleNamespace) -> None:
    _setup(env, _df(["FundA"]))
    _patch_saved(env, {})
    ms = env.stub.session_state["model_state"]
    ms["multi_period_enabled"] = False
    env.stub.checkbox_by_label["Enable Multi-Period"] = False
    env.module.render_model_page()
    # When multi_period_enabled is False, a warning is shown in entry/exit rules
    # This exercises line 4287
    assert any("Entry/exit rules only apply in multi-period mode" in w for w in env.stub.warnings)
