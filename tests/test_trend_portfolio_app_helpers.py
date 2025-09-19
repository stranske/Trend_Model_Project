import importlib
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, Iterable, Sequence, cast
from unittest.mock import MagicMock, Mock

import pandas as pd
import pytest


class _SessionState(dict):
    """Minimal mapping supporting attribute-style access."""

    def __getattr__(self, name: str):  # pragma: no cover - attribute proxy
        try:
            return self[name]
        except KeyError as exc:  # pragma: no cover - defensive
            raise AttributeError(name) from exc

    def __setattr__(self, name: str, value):
        self[name] = value

    def __delattr__(self, name: str):  # pragma: no cover - defensive
        try:
            del self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc


class _Context:
    """No-op context manager used for Streamlit layout primitives."""

    def __init__(self, st_module: ModuleType) -> None:
        self._st = st_module

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class _DummyStreamlit(ModuleType):
    """Lightweight stand-in for the Streamlit API used during import."""

    def __init__(self) -> None:
        super().__init__("streamlit")
        self.session_state: _SessionState = _SessionState()
        self.sidebar = _Context(self)
        self._rerun_calls: int = 0

    # Basic page/layout helpers -------------------------------------------------
    def set_page_config(self, *_, **__):
        return None

    def title(self, *_, **__):
        return None

    def header(self, *_, **__):
        return None

    def subheader(self, *_, **__):
        return None

    def caption(self, *_, **__):
        return None

    def markdown(self, *_, **__):
        return None

    def write(self, *_, **__):
        return None

    def info(self, *_, **__):
        return None

    def success(self, *_, **__):
        return None

    def warning(self, *_, **__):
        return None

    def error(self, *_, **__):
        return None

    def exception(self, *_, **__):
        return None

    # Interactive widgets ------------------------------------------------------
    def button(self, *_, **__):
        return False

    def checkbox(self, *_, value: bool = False, **__):
        return value

    def text_input(self, *_, value: str | None = "", **__):
        return value or ""

    def text_area(self, *_, value: str | None = "", **__):
        return value or ""

    def number_input(self, *_, value=0, **__):
        return value

    def slider(self, *_, value=None, **__):
        return value

    def selectbox(
        self, _label: str, options: Sequence | Iterable, *_, index: int = 0, **__
    ):
        options_list = list(options)
        if not options_list:
            return None
        return options_list[min(index, len(options_list) - 1)]

    def multiselect(
        self, _label: str, options: Sequence | Iterable, *_, default=None, **__
    ):
        if default is None:
            return []
        return list(default)

    def radio(
        self, _label: str, options: Sequence | Iterable, *_, index: int = 0, **__
    ):
        options_list = list(options)
        if not options_list:
            return None
        return options_list[min(index, len(options_list) - 1)]

    def file_uploader(self, *_, **__):
        return None

    def download_button(self, *_, **__):
        return None

    def dataframe(self, *_, **__):
        return None

    def tabs(self, names: Sequence[str]):
        return [_Context(self) for _ in names]

    def columns(self, spec):
        if isinstance(spec, int):
            count = spec
        else:
            count = len(list(spec))
        return [_Context(self) for _ in range(count)]

    def expander(self, *_, **__):
        return _Context(self)

    def spinner(self, *_, **__):
        return _Context(self)

    def rerun(self):
        self._rerun_calls += 1

    def stop(self):  # pragma: no cover - sanity guard
        raise RuntimeError("st.stop invoked during test")


def _load_app(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    """Import ``trend_portfolio_app.app`` with a stub Streamlit module."""

    dummy = _DummyStreamlit()
    monkeypatch.setitem(sys.modules, "streamlit", dummy)
    sys.modules.pop("trend_portfolio_app.app", None)

    module = ModuleType("trend_portfolio_app.app")
    module.__file__ = str(Path("src/trend_portfolio_app/app.py"))

    source = Path(module.__file__).read_text(encoding="utf-8")
    prefix = source.split("st.set_page_config", 1)[0]
    code = compile(prefix + "    pass\n", module.__file__, "exec")
    exec(code, module.__dict__)

    is_mock_impl = module._is_mock_streamlit  # type: ignore[attr-defined]

    def _compat_streamlit_mock(candidate: Any) -> bool:
        name = type(candidate).__name__
        if name in {"Mock", "MagicMock"}:
            return True
        return is_mock_impl(candidate)

    module._is_streamlit_mock = _compat_streamlit_mock  # type: ignore[attr-defined]
    module._STREAMLIT_IS_MOCK = _compat_streamlit_mock(module.st)  # type: ignore[attr-defined]
    module._SessionState = _SessionState  # type: ignore[attr-defined]

    sys.modules[module.__name__] = module
    return module


def _load_app_with_magicmock(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[ModuleType, MagicMock]:
    """Import the app module while ``streamlit`` is a ``MagicMock`` instance."""

    stub = MagicMock()
    # Ensure placeholder helpers fall back to the module-defined ``_NullContext``.
    stub.empty = None
    stub.columns.return_value = []
    stub.button.return_value = False
    stub.session_state = _SessionState()
    monkeypatch.setitem(sys.modules, "streamlit", stub)
    sys.modules.pop("trend_portfolio_app.app", None)

    module = ModuleType("trend_portfolio_app.app")
    module.__file__ = str(Path("src/trend_portfolio_app/app.py"))

    source = Path(module.__file__).read_text(encoding="utf-8")
    prefix = source.split("st.set_page_config", 1)[0]
    code = compile(prefix + "    pass\n", module.__file__, "exec")
    exec(code, module.__dict__)

    is_mock_impl = module._is_mock_streamlit  # type: ignore[attr-defined]

    def _compat_streamlit_mock(candidate: Any) -> bool:
        name = type(candidate).__name__
        if name in {"Mock", "MagicMock"}:
            return True
        return is_mock_impl(candidate)

    module._is_streamlit_mock = _compat_streamlit_mock  # type: ignore[attr-defined]
    module._STREAMLIT_IS_MOCK = _compat_streamlit_mock(module.st)  # type: ignore[attr-defined]
    module._SessionState = _SessionState  # type: ignore[attr-defined]

    sys.modules[module.__name__] = module
    return module, stub


def test_read_defaults_populates_expected_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    app_mod = _load_app(monkeypatch)

    defaults: dict[str, Any] = app_mod._read_defaults()
    assert "data" in defaults
    assert "portfolio" in defaults

    raw_portfolio = defaults["portfolio"]
    assert isinstance(raw_portfolio, dict)
    portfolio_section = cast(dict[str, Any], raw_portfolio)
    assert "policy" in portfolio_section

    raw_data = defaults["data"]
    assert isinstance(raw_data, dict)
    data_section = cast(dict[str, Any], raw_data)
    assert "csv_path" in data_section


def test_merge_update_deep_merges_nested_dicts(monkeypatch: pytest.MonkeyPatch) -> None:
    app_mod = _load_app(monkeypatch)

    base = {"a": 1, "nested": {"x": 10, "y": 20}}
    updates = {"nested": {"y": 99, "z": 5}, "b": 2}

    merged = app_mod._merge_update(base, updates)

    assert merged == {"a": 1, "nested": {"x": 10, "y": 99, "z": 5}, "b": 2}
    # Original dictionaries should remain unmodified
    nested_section = base["nested"]
    assert isinstance(nested_section, dict)
    assert nested_section["y"] == 20
    assert "z" not in nested_section


def test_build_cfg_accepts_roundtrip_from_yaml(monkeypatch: pytest.MonkeyPatch) -> None:
    app_mod = _load_app(monkeypatch)

    defaults: dict[str, Any] = app_mod._read_defaults()
    data_section = defaults.setdefault("data", {})
    if not isinstance(data_section, dict):
        raise AssertionError("Expected mapping for defaults['data']")
    data_section["csv_path"] = "demo.csv"
    yaml_text = app_mod._to_yaml(defaults)
    reconstructed = app_mod._build_cfg(app_mod.yaml.safe_load(yaml_text))

    assert reconstructed.data["csv_path"] == "demo.csv"


def test_summarise_run_df_rounds_numeric_columns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app_mod = _load_app(monkeypatch)

    df = pd.DataFrame({"metric": [1.234567, 2.345678], "label": ["A", "B"]})
    summary = app_mod._summarise_run_df(df)

    assert list(summary.columns) == ["metric", "label"]
    assert summary["metric"].tolist() == [1.2346, 2.3457]
    assert summary["label"].tolist() == ["A", "B"]


def test_summarise_multi_handles_missing_sections(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app_mod = _load_app(monkeypatch)

    results = [
        {
            "period": ("2020-01", "2020-06", "2020-07", "2020-12"),
            "out_ew_stats": {"sharpe": 1.23456, "cagr": 0.05678},
            "out_user_stats": {},
        },
        {"period": None, "out_ew_stats": object(), "out_user_stats": object()},
    ]

    summary = app_mod._summarise_multi(results)

    assert list(summary.columns) == [
        "in_start",
        "in_end",
        "out_start",
        "out_end",
        "ew_sharpe",
        "user_sharpe",
        "ew_cagr",
        "user_cagr",
    ]
    assert summary.loc[0, "ew_sharpe"] == pytest.approx(1.2346)
    assert summary.loc[0, "user_sharpe"] != summary.loc[0, "user_sharpe"]  # NaN
    assert summary.loc[1, "in_start"] == ""
    assert summary.loc[1, "ew_cagr"] != summary.loc[1, "ew_cagr"]  # NaN


def test_expected_columns_handles_various_specs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app_mod = _load_app(monkeypatch)

    assert app_mod._expected_columns(3) == 3
    assert app_mod._expected_columns(["a", "b"]) == 2
    assert app_mod._expected_columns(None) == 1


def test_normalize_columns_supplies_placeholders(monkeypatch: pytest.MonkeyPatch) -> None:
    app_mod = _load_app(monkeypatch)

    sentinel = object()
    app_mod.st.empty = lambda: sentinel  # type: ignore[attr-defined]

    cols = app_mod._normalize_columns(None, 3)

    assert cols == [sentinel, sentinel, sentinel]

    # When explicit columns exceed the requested count they should be truncated.
    trimmed = app_mod._normalize_columns([1, 2, 3], 2)
    assert trimmed == [1, 2]


def test_columns_wraps_streamlit_columns(monkeypatch: pytest.MonkeyPatch) -> None:
    app_mod = _load_app(monkeypatch)

    calls: list[Any] = []

    def fake_columns(spec: Any) -> list[str]:
        calls.append(spec)
        return ["first", "second"]

    app_mod.st.columns = fake_columns  # type: ignore[assignment]

    result = app_mod._columns([1, 2, 3])

    assert calls == [[1, 2, 3]]
    assert result == ["first", "second", "second"]


def test_is_streamlit_mock_identifies_mocks(monkeypatch: pytest.MonkeyPatch) -> None:
    app_mod = _load_app(monkeypatch)

    assert app_mod._is_streamlit_mock(Mock()) is True
    assert app_mod._is_streamlit_mock(object()) is False


def test_magicmock_streamlit_bootstrap_installs_null_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app_mod, stub = _load_app_with_magicmock(monkeypatch)

    try:
        assert app_mod._STREAMLIT_IS_MOCK is True
        # Without ``streamlit.empty`` the helper should allocate ``_NullContext`` instances.
        placeholders = app_mod._normalize_columns(None, 2)
        assert all(isinstance(p, app_mod._NullContext) for p in placeholders)

        # UI callbacks should be replaced with inert stubs when patched with ``MagicMock``.
        assert app_mod.st.button("any") is False
        assert isinstance(app_mod.st.session_state, app_mod._SessionState)
    finally:
        sys.modules.pop("trend_portfolio_app.app", None)


def test_apply_session_state_expands_session_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    app_mod = _load_app(monkeypatch)

    state = app_mod.st.session_state
    state.clear()
    state.update(
        {
            "data.csv_path": "override.csv",
            "portfolio.policy": "aggressive",
            "multi_period.window._months": "6",
            "metrics.alpha": 0.75,
            "unrelated": "ignored",
            "custom.key": "skipped",  # Prefix missing from allow list
        }
    )

    cfg: dict[str, Any] = {"data": {"csv_path": "initial.csv"}}

    app_mod._apply_session_state(cfg)

    assert cfg["data"]["csv_path"] == "override.csv"
    assert cfg["portfolio"]["policy"] == "aggressive"
    assert cfg["multi_period"]["window"]["length"] == 126
    assert cfg["metrics"]["alpha"] == 0.75
    assert "custom" not in cfg
    assert "unrelated" not in cfg


def test_render_sidebar_resets_and_serialises(monkeypatch: pytest.MonkeyPatch) -> None:
    app_mod = _load_app(monkeypatch)

    defaults = {"data": {"csv_path": "reset.csv"}, "portfolio": {"policy": "reset"}}
    monkeypatch.setattr(app_mod, "_read_defaults", lambda: json.loads(json.dumps(defaults)))

    button_calls: list[str] = []

    def fake_button(label: str, *_, **__) -> bool:
        button_calls.append(label)
        return label == "Reset to defaults"

    app_mod.st.button = fake_button  # type: ignore[assignment]

    seen_values: list[str] = []

    def fake_text_input(
        label: str, *, key: str, value: str, help: str
    ) -> str:  # pragma: no cover - signature guard
        seen_values.append(value)
        assert key == "data.csv_path"
        assert "manager returns" in help
        return "user.csv"

    app_mod.st.text_input = fake_text_input  # type: ignore[assignment]

    downloads: list[tuple[str, bytes, str, str]] = []

    def fake_download(
        label: str, *, data: bytes, file_name: str, mime: str
    ) -> None:  # pragma: no cover - signature guard
        downloads.append((label, data, file_name, mime))

    app_mod.st.download_button = fake_download  # type: ignore[assignment]

    cfg: dict[str, Any] = {"data": {"csv_path": "initial.csv"}}

    app_mod._render_sidebar(cfg)

    assert button_calls == ["Reset to defaults"]
    assert app_mod.st.session_state.config_dict["portfolio"]["policy"] == "reset"
    assert app_mod.st.session_state.config_dict["data"]["csv_path"] == "user.csv"
    assert seen_values == ["reset.csv"]
    assert cfg["data"]["csv_path"] == "user.csv"

    assert len(downloads) == 1
    label, payload, file_name, mime = downloads[0]
    assert label == "Download YAML"
    assert file_name == "config.yml"
    assert mime == "text/yaml"
    dumped = app_mod.yaml.safe_load(payload.decode("utf-8"))
    assert dumped["data"]["csv_path"] == "user.csv"


def test_render_run_section_executes_single_period(monkeypatch: pytest.MonkeyPatch) -> None:
    app_mod = _load_app(monkeypatch)

    def fake_button(label: str, *_, **__) -> bool:
        return label == "Run Single Period"

    app_mod.st.button = fake_button  # type: ignore[assignment]

    successes: list[str] = []
    app_mod.st.success = successes.append  # type: ignore[assignment]

    tables: list[pd.DataFrame] = []
    app_mod.st.dataframe = lambda df, **__: tables.append(df)  # type: ignore[assignment]

    downloads: list[tuple[str, bytes, str, str]] = []

    def fake_download(
        label: str, *, data: bytes, file_name: str, mime: str
    ) -> None:  # pragma: no cover - signature guard
        downloads.append((label, data, file_name, mime))

    app_mod.st.download_button = fake_download  # type: ignore[assignment]

    app_mod.st.session_state.clear()
    app_mod.st.session_state.update({"data.csv_path": "from-state.csv"})

    cfg: dict[str, Any] = {"data": {"csv_path": "initial.csv"}}

    summary = pd.DataFrame({"value": [1, 2]})
    monkeypatch.setattr(app_mod, "_summarise_run_df", lambda _df: summary)

    run_calls: list[Any] = []
    monkeypatch.setattr(app_mod.pipeline, "run", lambda cfg_obj: run_calls.append(cfg_obj) or summary)

    built_cfg_objects: list[Any] = []

    def fake_build(d: dict[str, Any]) -> dict[str, Any]:
        cfg_obj = {"__cfg__": dict(d)}
        built_cfg_objects.append(cfg_obj)
        return cfg_obj

    monkeypatch.setattr(app_mod, "_build_cfg", fake_build)

    app_mod._render_run_section(cfg)

    assert built_cfg_objects
    assert run_calls == built_cfg_objects
    assert cfg["data"]["csv_path"] == "from-state.csv"
    assert successes == ["Completed. 2 rows."]
    assert tables == [summary]
    assert len(downloads) == 1
    label, payload, file_name, mime = downloads[0]
    assert label == "Download CSV"
    assert file_name == "single_period_summary.csv"
    assert mime == "text/csv"
    rows = payload.decode("utf-8").strip().splitlines()
    assert rows[0] == "value"
    assert rows[1:] == ["1", "2"]


def test_render_run_section_executes_multi_period(monkeypatch: pytest.MonkeyPatch) -> None:
    app_mod = _load_app(monkeypatch)

    def fake_button(label: str, *_, **__) -> bool:
        return label == "Run Multi-Period"

    app_mod.st.button = fake_button  # type: ignore[assignment]

    successes: list[str] = []
    app_mod.st.success = successes.append  # type: ignore[assignment]

    tables: list[pd.DataFrame] = []
    app_mod.st.dataframe = lambda df, **__: tables.append(df)  # type: ignore[assignment]

    downloads: list[tuple[str, bytes, str, str]] = []

    def fake_download(
        label: str, *, data: bytes, file_name: str, mime: str
    ) -> None:  # pragma: no cover - signature guard
        downloads.append((label, data, file_name, mime))

    app_mod.st.download_button = fake_download  # type: ignore[assignment]

    app_mod.st.session_state.clear()
    app_mod.st.session_state.update({"data.csv_path": "from-state.csv"})

    cfg: dict[str, Any] = {"data": {"csv_path": "initial.csv"}}

    summary = pd.DataFrame({"ew_sharpe": [1.0]})
    monkeypatch.setattr(app_mod, "_summarise_multi", lambda _results: summary)

    payload = [
        {"period": ("2020", "2020", "2021", "2021"), "out_ew_stats": {"sharpe": 1.0}},
        {"period": ("2021", "2021", "2022", "2022"), "out_ew_stats": {"sharpe": 1.1}},
    ]

    monkeypatch.setattr(app_mod, "run_multi", lambda _cfg: payload)

    built_cfg_objects: list[dict[str, Any]] = []

    def fake_build(d: dict[str, Any]) -> dict[str, Any]:
        cfg_obj = dict(d)
        built_cfg_objects.append(cfg_obj)
        return cfg_obj

    monkeypatch.setattr(app_mod, "_build_cfg", fake_build)

    app_mod._render_run_section(cfg)

    assert cfg["data"]["csv_path"] == "from-state.csv"
    assert built_cfg_objects and built_cfg_objects[0]["data"]["csv_path"] == "from-state.csv"
    assert successes == ["Completed. Periods: 2"]
    assert tables == [summary]
    assert len(downloads) == 2
    csv_label, csv_payload, csv_name, csv_mime = downloads[0]
    assert csv_label == "Download periods CSV"
    assert csv_name == "multi_period_summary.csv"
    assert csv_mime == "text/csv"
    csv_rows = csv_payload.decode("utf-8").strip().splitlines()
    assert csv_rows[0] == "ew_sharpe"
    assert csv_rows[1:] == ["1.0"]

    raw_label, raw_payload, raw_name, raw_mime = downloads[1]
    assert raw_label == "Download raw JSON"
    assert raw_name == "multi_period_raw.json"
    assert raw_mime == "application/json"
    decoded_payload = json.loads(raw_payload.decode("utf-8"))
    expected = []
    for item in payload:
        expected.append({**item, "period": list(item["period"])})
    assert decoded_payload == expected
