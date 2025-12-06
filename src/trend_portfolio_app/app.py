from __future__ import annotations

import gc
import importlib
import json
import os
import sys
import types
from collections.abc import Mapping, Sequence
from importlib import import_module
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Literal, cast

import pandas as pd
import streamlit as st
import yaml

import trend_analysis as _trend_pkg
from trend_analysis.config import DEFAULTS as DEFAULT_CFG_PATH
from trend_analysis.config import Config, validate_trend_config
from trend_analysis.data import identify_risk_free_fund
from trend_analysis.diagnostics import coerce_pipeline_result
from trend_analysis.logging_setup import setup_logging
from trend_analysis.multi_period import run_from_config as run_multi
from trend_analysis.multi_period.loaders import detect_index_columns
from utils.paths import proj_path

_STREAMLIT_LOG_ENV = "TREND_STREAMLIT_LOG_PATH"
_PIPELINE_SIMPLE_ENV = "TREND_PIPELINE_PROXY_SIMPLE"
_STREAMLIT_LOG_PATH: Path | None = None
_SIMPLE_PIPELINE_CACHE: types.ModuleType | None = None


def _ensure_streamlit_logging() -> Path | None:
    disable = os.environ.get("TREND_DISABLE_PERF_LOGS", "").strip().lower()
    if disable in {"1", "true", "yes"}:
        return None
    existing = os.environ.get(_STREAMLIT_LOG_ENV)
    if existing:
        return Path(existing)
    log_path = setup_logging(app_name="app")
    os.environ[_STREAMLIT_LOG_ENV] = str(log_path)
    return log_path


_STREAMLIT_LOG_PATH = _ensure_streamlit_logging()

_PIPELINE_DEBUG: list[tuple[str, bool, int, int, int]] = []


def _resolve_pipeline(*, fresh: bool = False, simple: bool = False) -> Any:
    """Return the preferred ``trend_analysis.pipeline`` module.

    When ``simple`` is set the module cache entry is dropped so we import a
    clean copy without attempting to honour patched module objects or GC
    scanning. Otherwise this consults the live module cache so tests that swap
    ``sys.modules['trend_analysis.pipeline']`` continue to work. When the
    cached module differs from the originally imported instance (for example if
    another test reloaded the package) we prefer whichever version exposes
    patched attributes. This mirrors the previous eager-import behaviour while
    allowing lazy resolution.
    """

    global _SIMPLE_PIPELINE_CACHE

    # When simple mode is disabled, drop any cached module to avoid reusing a
    # direct-imported pipeline after switching back to the GC-scanning path.
    if not simple:
        _SIMPLE_PIPELINE_CACHE = None

    if simple:
        if _SIMPLE_PIPELINE_CACHE is not None:
            return _SIMPLE_PIPELINE_CACHE

        sys.modules.pop("trend_analysis.pipeline", None)
        module = import_module("trend_analysis.pipeline")
        _SIMPLE_PIPELINE_CACHE = module
        return module

    module = import_module("trend_analysis.pipeline")

    if fresh:
        try:
            module = importlib.reload(module)
        except Exception:
            # Best-effort reload; fall back to the already imported module if
            # reload fails (for example when the module is a stub without a
            # loader during tests).
            pass

    return module


def _pipeline_proxy_simple() -> bool:
    """Return ``True`` when the simple proxy mode is enabled via env flag."""

    flag = os.environ.get(_PIPELINE_SIMPLE_ENV, "")
    return flag.strip().lower() in {"1", "true", "yes", "on"}


class _PipelineProxy:
    """Lazy proxy that favours patched pipeline attributes when reloaded."""

    def __getattr__(self, name: str) -> Any:
        global _SIMPLE_PIPELINE_CACHE
        missing = object()

        if _pipeline_proxy_simple():
            module = _resolve_pipeline(fresh=True, simple=True)
            attr = getattr(module, name, missing)

            if name == "run":
                pkg_module = getattr(_trend_pkg, "pipeline", None)
                pkg_attr = (
                    getattr(pkg_module, name, missing)
                    if pkg_module is not None
                    else missing
                )
                _PIPELINE_DEBUG.append(
                    (
                        name,
                        True,
                        id(module),
                        id(pkg_module) if pkg_module is not None else 0,
                        id(pkg_attr) if pkg_attr is not missing else 0,
                    )
                )

            if attr is not missing:
                return attr

            pkg_module = getattr(_trend_pkg, "pipeline", None)
            pkg_attr = (
                getattr(pkg_module, name, missing)
                if pkg_module is not None
                else missing
            )
            if pkg_attr is not missing:
                return pkg_attr

            raise AttributeError(name)

        # Simple mode is disabled; ensure any cached simple-mode module is
        # cleared so switching back to the default path prefers patched
        # GC-scanned modules.
        _SIMPLE_PIPELINE_CACHE = None

        # Prefer attributes from the live module object in sys.modules. Tests
        # commonly monkeypatch `trend_analysis.pipeline` in-place, so returning
        # the attribute from the cached module ensures those patches are
        # honoured. Only if the attribute is missing from the cached module do
        # we fall back to the package-level `trend_analysis.pipeline` attr.
        module = sys.modules.get("trend_analysis.pipeline")
        if module is None:
            module = _resolve_pipeline()

        attr = getattr(module, name, missing)

        # Instrumentation for tests — keep a lightweight trace when 'run' is
        # resolved so debugging info is available if needed.
        if name == "run":
            pkg_module = getattr(_trend_pkg, "pipeline", None)
            pkg_attr = (
                getattr(pkg_module, name, missing)
                if pkg_module is not None
                else missing
            )
            _PIPELINE_DEBUG.append(
                (
                    name,
                    False,
                    id(module),
                    id(pkg_module) if pkg_module is not None else 0,
                    id(pkg_attr) if pkg_attr is not missing else 0,
                )
            )

        # If the live module exposes the attribute, we still check for any
        # other in-memory module objects named "trend_analysis.pipeline" that
        # might have been monkeypatched (tests sometimes patch an earlier
        # imported module instance that no longer lives in sys.modules). If
        # we find a different attribute on one of those module objects, prefer
        # that patched attribute. Otherwise return the current module attr.
        if attr is not missing:
            # Search GC for other module instances named 'trend_analysis.pipeline'
            try:
                for obj in gc.get_objects():
                    if not isinstance(obj, types.ModuleType):
                        continue
                    if getattr(obj, "__name__", None) != "trend_analysis.pipeline":
                        continue
                    cand = getattr(obj, name, missing)
                    if cand is not missing and cand is not attr:
                        return cand
            except Exception:
                # GC inspection is best-effort; fall back to the module attr.
                pass

            return attr

        # Otherwise attempt to resolve via the package attribute as a fallback.
        pkg_module = getattr(_trend_pkg, "pipeline", None)
        pkg_attr = (
            getattr(pkg_module, name, missing) if pkg_module is not None else missing
        )

        if pkg_attr is not missing:
            return pkg_attr

        raise AttributeError(name)


pipeline = _PipelineProxy()

# Instrumentation globals (mutated during _render_app) used by tests.
page_config_calls: list[bool] = []
titles: list[str] = []
# Re-export for tests that access globals directly after import
__all__ = [
    "page_config_calls",
    "titles",
]

if TYPE_CHECKING:  # pragma: no cover - type-only alias for static checkers
    from trend_analysis.config.models import ConfigProtocol as ConfigType
else:  # pragma: no cover - runtime fallback when pydantic models are optional
    from typing import Any as ConfigType


# ---------------------------------------------------------------------------
# Helper primitives used across the Streamlit app


class _NullContext:
    """Minimal context manager used when Streamlit is mocked in tests."""

    def __enter__(self) -> "_NullContext":  # pragma: no cover - trivial
        return self

    def __exit__(self, *_exc: Any) -> Literal[False]:  # pragma: no cover - trivial
        return False


def _is_mock_streamlit(module: Any) -> bool:
    """Best‑effort detection for a real ``unittest.mock.MagicMock`` instance.

    The previous implementation relied solely on ``type(module).__name__ ==
    'MagicMock'`` which allowed tests (or any code) to suppress the application
    bootstrap simply by reassigning ``__name__`` on an arbitrary stub class.
    That made the import side‑effect (page configuration) disappear and caused
    the ``test_render_app_executes_with_dummy_streamlit`` test to fail because
    the app never ran.

    We tighten the heuristic to only treat genuine ``unittest.mock`` MagicMock
    objects as mocks. A lightweight subclass that merely renames its
    ``__name__`` no longer short‑circuits the render path, restoring the
    expected behaviour under test.
    """

    cls = type(module)
    return cls.__module__ == "unittest.mock" and cls.__name__ == "MagicMock"


def _read_defaults() -> Dict[str, Any]:
    """Load the default YAML configuration bundled with the analysis
    package."""

    data = yaml.safe_load(Path(DEFAULT_CFG_PATH).read_text(encoding="utf-8"))
    if not isinstance(data, dict):  # pragma: no cover - defensive
        raise TypeError("Default configuration must be a mapping")

    defaults: Dict[str, Any] = dict(data)
    defaults.setdefault("data", {})
    defaults.setdefault("portfolio", {})

    demo_csv = Path("demo/demo_returns.csv")
    if demo_csv.exists():
        defaults["data"].setdefault("csv_path", str(demo_csv))
    defaults["portfolio"].setdefault("policy", "")

    return defaults


@st.cache_data(show_spinner=False)
def _analyze_csv_columns(csv_path: str) -> Dict[str, Any]:
    """Load CSV and analyze columns for risk-free and benchmark candidates.

    Returns a dict with:
    - columns: list of all column names
    - numeric_columns: list of numeric column names (excluding Date)
    - risk_free_candidate: auto-detected risk-free column (or None)
    - benchmark_candidates: list of detected benchmark/index columns
    - error: error message if loading failed (or None)
    """
    result: Dict[str, Any] = {
        "columns": [],
        "numeric_columns": [],
        "risk_free_candidate": None,
        "benchmark_candidates": [],
        "error": None,
    }

    if not csv_path or not csv_path.strip():
        return result

    # Restrict CSV loading to files under the allowed root directory ("demo")
    safe_root = Path("demo").resolve()
    user_input_path = Path(csv_path.strip())
    # Reject absolute paths up front
    if user_input_path.is_absolute():
        result["error"] = f"Access denied: path must be relative ({csv_path})"
        return result
    try:
        # Join user input to safe root, then resolve
        candidate_path = (safe_root / user_input_path).resolve()
        # Ensure candidate_path is a child/descendant of safe_root
        candidate_path.relative_to(safe_root)
    except Exception as exc:
        result["error"] = f"Access denied or invalid path: {csv_path} ({exc})"
        return result
    if not candidate_path.exists():
        result["error"] = f"File not found: {csv_path}"
        return result

    try:
        df = pd.read_csv(candidate_path)
        result["columns"] = list(df.columns)

        # Get numeric columns (excluding Date-like columns)
        date_cols = {"date", "Date", "DATE", "timestamp", "Timestamp"}

        # First check for already-numeric columns
        numeric_cols = [
            c
            for c in df.select_dtypes(include=["number"]).columns
            if c not in date_cols
        ]

        # If no numeric columns found, try to detect percentage-formatted columns
        # (e.g., "0.37%", "-2.72%") which are common in financial data
        if not numeric_cols:
            for col in df.columns:
                if col in date_cols:
                    continue
                # Check if column values look like percentages or numbers
                sample = df[col].dropna().head(10).astype(str)
                if sample.empty:
                    continue
                # Check if values contain % or look numeric
                looks_numeric = (
                    sample.str.replace(r"[%,\s]", "", regex=True)
                    .str.match(r"^-?\d*\.?\d+$")
                    .any()
                )
                if looks_numeric:
                    numeric_cols.append(col)

        result["numeric_columns"] = numeric_cols

        # Detect benchmark/index columns from all non-date columns
        all_data_cols = [c for c in df.columns if c not in date_cols]
        result["benchmark_candidates"] = detect_index_columns(all_data_cols)

        # Detect risk-free candidate (lowest volatility column)
        # For percentage data, we need to convert first
        if numeric_cols:
            try:
                # Try to convert percentage strings to floats for analysis
                df_numeric = df.copy()
                for col in numeric_cols:
                    if df_numeric[col].dtype == object:
                        # Convert percentage strings like "0.37%" to 0.0037
                        df_numeric[col] = (
                            df_numeric[col]
                            .astype(str)
                            .str.replace("%", "", regex=False)
                            .str.replace(",", "", regex=False)
                            .apply(pd.to_numeric, errors="coerce")
                        )
                        # If values were percentages, they're now in decimal form
                        # (e.g., 0.37 from "0.37%")

                rf = identify_risk_free_fund(df_numeric)
                if rf and rf in numeric_cols:
                    result["risk_free_candidate"] = rf
            except Exception as e:
                import logging

                logging.getLogger(__name__).debug(f"Risk-free detection failed: {e}")

    except Exception as exc:
        result["error"] = str(exc)

    return result


def _to_yaml(d: Dict[str, Any]) -> str:
    """Serialise a mapping to YAML while preserving insertion order."""

    dumped: str = yaml.safe_dump(d, sort_keys=False, allow_unicode=True)
    return dumped


def _merge_update(base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge ``updates`` into a shallow copy of ``base``."""

    merged: Dict[str, Any] = dict(base)
    for key, value in updates.items():
        base_value = merged.get(key)
        if isinstance(base_value, dict) and isinstance(value, Mapping):
            merged[key] = _merge_update(cast(Dict[str, Any], base_value), dict(value))
        else:
            merged[key] = value
    return merged


def _expected_columns(spec: Any) -> int:
    if isinstance(spec, int):
        return max(0, spec)
    if isinstance(spec, Sequence):
        return len(spec)
    return 1


def _normalize_columns(cols: Any, expected: int) -> List[Any]:
    if isinstance(cols, (list, tuple)):
        normalised = list(cols)
    elif cols is None:
        normalised = []
    else:
        normalised = [cols]

    if not normalised:
        placeholder_factory = getattr(st, "empty", None)
        if callable(placeholder_factory):
            placeholder = placeholder_factory()
        else:
            placeholder = _NullContext()
        normalised = [placeholder]

    if len(normalised) < expected:
        filler = normalised[-1]
        normalised.extend([filler] * (expected - len(normalised)))

    return normalised[:expected]


def _columns(spec: Any) -> List[Any]:
    expected = max(1, _expected_columns(spec))
    return _normalize_columns(st.columns(spec), expected)


def _build_cfg(d: Dict[str, Any]) -> ConfigType:
    """Instantiate the flexible ``Config`` object used by the pipeline."""
    validate_trend_config(d, base_path=proj_path())
    return Config(**d)


def _summarise_run_df(df: pd.DataFrame | None) -> pd.DataFrame:
    """Round numeric columns for presentation after a single-period run."""

    if df is None or df.empty:
        return pd.DataFrame()

    disp = df.copy()
    for column in disp.columns:
        if pd.api.types.is_numeric_dtype(disp[column]):
            disp[column] = pd.to_numeric(disp[column], errors="coerce").round(4)
    return disp


def _build_summary_from_result(result: Mapping[str, Any] | None) -> pd.DataFrame:
    """Construct a summary DataFrame from ``pipeline.run_full`` results."""

    if not result:
        return pd.DataFrame()

    out_stats = result.get("out_sample_stats")
    if not isinstance(out_stats, Mapping) or not out_stats:
        return pd.DataFrame()

    rows: Dict[str, Dict[str, Any]] = {}
    for key, value in out_stats.items():
        if hasattr(value, "__dict__"):
            rows[key] = dict(vars(value))
        elif isinstance(value, Mapping):
            rows[key] = dict(value)

    summary = pd.DataFrame(rows).T
    for label, ir_map in cast(
        Mapping[str, Mapping[str, float]], result.get("benchmark_ir", {})
    ).items():
        series = pd.Series(
            {
                asset: score
                for asset, score in ir_map.items()
                if asset not in {"equal_weight", "user_weight"}
            }
        )
        summary[f"ir_{label}"] = series

    return summary


def _summarise_multi(results: List[Dict[str, Any]]) -> pd.DataFrame:
    """Create a tidy summary table from multi-period back-test results."""

    rows: List[Dict[str, Any]] = []

    def _period_value(period: Any, index: int) -> Any:
        if period is None:
            return ""
        if isinstance(period, (list, tuple)):
            return period[index] if len(period) > index else ""
        try:
            seq = list(period)
        except (TypeError, ValueError):
            return ""
        return seq[index] if len(seq) > index else ""

    def _coerce_metric(container: Any, key: str) -> float:
        if container is None:
            return float("nan")
        value: Any
        if isinstance(container, Mapping):
            value = container.get(key)
        else:
            value = getattr(container, key, None)
        try:
            return float(value)
        except (TypeError, ValueError):
            return float("nan")

    for result in results:
        period = result.get("period")
        out_ew = result.get("out_ew_stats") or {}
        out_user = result.get("out_user_stats") or {}

        rows.append(
            {
                "in_start": _period_value(period, 0),
                "in_end": _period_value(period, 1),
                "out_start": _period_value(period, 2),
                "out_end": _period_value(period, 3),
                "ew_sharpe": _coerce_metric(out_ew, "sharpe"),
                "user_sharpe": _coerce_metric(out_user, "sharpe"),
                "ew_cagr": _coerce_metric(out_ew, "cagr"),
                "user_cagr": _coerce_metric(out_user, "cagr"),
            }
        )

    df = pd.DataFrame(rows)
    if not df.empty:
        for col in ["ew_sharpe", "user_sharpe", "ew_cagr", "user_cagr"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").round(4)
    return df


# ---------------------------------------------------------------------------
# Streamlit UI wiring


def _apply_session_state(cfg_dict: Dict[str, Any]) -> None:
    """Fold dotted session-state keys back into the nested configuration."""

    def _set_nested(target: Dict[str, Any], dotted: str, value: Any) -> None:
        parts = dotted.split(".")
        cursor: Dict[str, Any] = target
        for key in parts[:-1]:
            next_val = cursor.get(key)
            if not isinstance(next_val, dict):
                next_val = {}
                cursor[key] = next_val
            cursor = cast(Dict[str, Any], next_val)
        cursor[parts[-1]] = value

    prefixes = (
        "data.",
        "sample_split.",
        "portfolio.",
        "metrics.",
        "multi_period.",
        "preprocessing.",
        "vol_adjust.",
    )

    for raw_key, raw_value in list(st.session_state.items()):
        if not isinstance(raw_key, str) or "." not in raw_key:
            continue
        if not raw_key.startswith(prefixes):
            continue
        if raw_key.endswith("._months"):
            try:
                months = int(raw_value)
            except (TypeError, ValueError):
                continue
            days = int(months * 21)
            base_key = raw_key.rsplit(".", 1)[0]
            _set_nested(cfg_dict, f"{base_key}.length", days)
            continue
        _set_nested(cfg_dict, raw_key, raw_value)

    # Ensure the canonical csv_path key mirrors the sidebar input.
    csv_key = "data.csv_path"
    if csv_key in st.session_state:
        _set_nested(cfg_dict, csv_key, st.session_state[csv_key])


def _render_sidebar(cfg_dict: Dict[str, Any]) -> None:
    st.header("Configuration")

    if st.button("Reset to defaults", use_container_width=True):
        st.session_state.config_dict = _read_defaults()
        cfg_dict.update(st.session_state.config_dict)

    csv_value = st.text_input(
        "CSV path",
        key="data.csv_path",
        value=str(cfg_dict.get("data", {}).get("csv_path", "")),
        help="Path to a CSV with manager returns",
    )
    cfg_dict.setdefault("data", {})["csv_path"] = csv_value

    # Analyze CSV columns and show configuration options
    if csv_value and csv_value.strip():
        analysis = _analyze_csv_columns(csv_value)

        if analysis["error"]:
            st.warning(f"⚠️ {analysis['error']}")
        elif analysis["columns"]:
            # We have columns - show configuration even if not all are numeric
            st.divider()
            st.subheader("Data Configuration")

            # Show column count
            num_cols = len(analysis["numeric_columns"])
            total_cols = len(analysis["columns"]) - 1  # Exclude Date
            if num_cols > 0:
                st.caption(f"📊 {num_cols} data columns detected")
            else:
                st.caption(
                    f"📊 {total_cols} columns detected (will be converted on load)"
                )

            # Get columns for selection - use numeric if available, else all non-date
            selectable_cols = analysis["numeric_columns"]
            if not selectable_cols:
                date_cols = {"date", "Date", "DATE", "timestamp", "Timestamp"}
                selectable_cols = [c for c in analysis["columns"] if c not in date_cols]

            # Risk-free column selection
            rf_options = ["(None - auto-detect)", "(None - skip)"] + selectable_cols
            current_rf = cfg_dict.get("data", {}).get("risk_free_column", "")

            # Determine default index
            if current_rf and current_rf in selectable_cols:
                rf_default_idx = rf_options.index(current_rf)
            elif analysis["risk_free_candidate"]:
                # Show the auto-detected candidate
                rf_default_idx = 0  # Auto-detect
            else:
                rf_default_idx = 0

            rf_selection = st.selectbox(
                "Risk-free column",
                rf_options,
                index=rf_default_idx,
                help="Column to use as risk-free rate. 'Auto-detect' finds the lowest volatility column.",
            )

            # Update config based on selection
            if rf_selection == "(None - auto-detect)":
                cfg_dict.setdefault("data", {})["risk_free_column"] = None
                cfg_dict.setdefault("data", {})["allow_risk_free_fallback"] = True
                if analysis["risk_free_candidate"]:
                    st.caption(f"💡 Auto-detected: {analysis['risk_free_candidate']}")
            elif rf_selection == "(None - skip)":
                cfg_dict.setdefault("data", {})["risk_free_column"] = None
                cfg_dict.setdefault("data", {})["allow_risk_free_fallback"] = True
            else:
                cfg_dict.setdefault("data", {})["risk_free_column"] = rf_selection
                cfg_dict.setdefault("data", {})["allow_risk_free_fallback"] = False

            # Benchmark/Index columns (informational)
            if analysis["benchmark_candidates"]:
                st.caption(
                    f"📈 Detected benchmarks: {', '.join(analysis['benchmark_candidates'])}"
                )
            else:
                st.caption("ℹ️ No benchmark columns detected (SPX, TSX, INDEX, etc.)")

    st.divider()
    yaml_bytes = _to_yaml(cfg_dict).encode("utf-8")
    st.download_button(
        "Download YAML", data=yaml_bytes, file_name="config.yml", mime="text/yaml"
    )


def _render_run_section(cfg_dict: Dict[str, Any]) -> None:
    st.subheader("Execute analysis")
    col1, col2 = _columns(2)
    with col1:
        go_single = st.button("Run Single Period", type="primary")
    with col2:
        go_multi = st.button("Run Multi-Period", type="primary")

    cfg_obj: ConfigType | None = None
    if go_single or go_multi:
        _apply_session_state(cfg_dict)
        try:
            cfg_obj = _build_cfg(cfg_dict)
        except Exception as exc:  # pragma: no cover - defensive UI feedback
            st.error(f"Failed to build configuration: {exc}")
            return

    if go_single and cfg_obj is not None:
        with st.spinner("Running single-period analysis..."):
            try:
                summary_frame = pipeline.run(cfg_obj)
            except Exception:  # pragma: no cover - UI fallthrough
                summary_frame = None
            run_full_error: Exception | None = None
            full_result_payload: dict[str, Any] | None = None
            full_result_diag = None
            try:
                full_result = pipeline.run_full(cfg_obj)
            except (FileNotFoundError, KeyError) as exc:
                run_full_error = exc
                full_result = None
            except Exception as exc:  # pragma: no cover - defensive logging
                run_full_error = exc
                full_result = None
            if full_result is not None:
                full_result_payload, full_result_diag = coerce_pipeline_result(
                    full_result
                )

        summary = _summarise_run_df(
            summary_frame if isinstance(summary_frame, pd.DataFrame) else None
        )
        has_summary = isinstance(summary, pd.DataFrame)
        summary_rows = len(summary) if has_summary else 0

        if not full_result_payload and not has_summary:
            message = "Analysis failed for the configured period. Please check your data and configuration settings."
            if full_result_diag is not None:
                message = (
                    f"{message} ({full_result_diag.reason_code}: "
                    f"{full_result_diag.message})"
                )
            elif run_full_error is not None:
                message = f"{message} ({run_full_error})"
            st.warning(message)
        else:
            if summary.empty and full_result_payload:
                summary_raw = _build_summary_from_result(full_result_payload)
                summary = _summarise_run_df(summary_raw)
                summary_rows = len(summary)
            st.success(f"Completed. {summary_rows} rows.")
            if not summary.empty:
                st.dataframe(summary, use_container_width=True)
                csv_bytes = summary.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Download CSV",
                    data=csv_bytes,
                    file_name="single_period_summary.csv",
                    mime="text/csv",
                )

            if full_result_payload:
                risk_diag = full_result_payload.get("risk_diagnostics")
                if isinstance(risk_diag, Mapping) and risk_diag:
                    st.subheader("Risk diagnostics")
                    line_chart = getattr(st, "line_chart", None)
                    bar_chart = getattr(st, "bar_chart", None)

                    asset_vol = risk_diag.get("asset_volatility")
                    if isinstance(asset_vol, pd.DataFrame) and not asset_vol.empty:
                        st.caption("Realised asset volatility")
                        if callable(line_chart):
                            line_chart(asset_vol)
                        else:
                            st.dataframe(asset_vol)

                    port_vol = risk_diag.get("portfolio_volatility")
                    if isinstance(port_vol, pd.Series) and not port_vol.empty:
                        st.caption("Portfolio volatility")
                        port_df = port_vol.to_frame("portfolio_volatility")
                        if callable(line_chart):
                            line_chart(port_df)
                        else:
                            st.dataframe(port_df)

                    turnover_series = risk_diag.get("turnover")
                    if (
                        isinstance(turnover_series, pd.Series)
                        and not turnover_series.empty
                    ):
                        st.caption("Turnover per rebalance")
                        turnover_df = turnover_series.to_frame("turnover")
                        if callable(bar_chart):
                            bar_chart(turnover_df)
                        else:
                            st.dataframe(turnover_df)

                    turnover_value = risk_diag.get("turnover_value")
                    if isinstance(turnover_value, (float, int)):
                        st.caption(f"Turnover applied: {turnover_value:.4f}")
            elif run_full_error is not None or full_result_diag is not None:
                detail = (
                    f"{full_result_diag.reason_code}: {full_result_diag.message}"
                    if full_result_diag is not None
                    else str(run_full_error)
                )
                st.info(
                    "Partial results shown. Full diagnostics are unavailable because the detailed analysis failed: "
                    f"{detail}"
                )

    if go_multi and cfg_obj is not None:
        with st.spinner("Running multi-period analysis..."):
            results = run_multi(cfg_obj)
        summary = _summarise_multi(results)
        st.success(f"Completed. Periods: {len(results)}")
        if not summary.empty:
            st.dataframe(summary, use_container_width=True)
            csv_bytes = summary.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download periods CSV",
                data=csv_bytes,
                file_name="multi_period_summary.csv",
                mime="text/csv",
            )
        raw = json.dumps(results, default=str).encode("utf-8")
        st.download_button(
            "Download raw JSON",
            data=raw,
            file_name="multi_period_raw.json",
            mime="application/json",
        )


def _render_app() -> None:
    # Instrumentation lists used by tests (harmless in production)
    st.set_page_config(page_title="Trend Portfolio App", layout="wide")
    page_config_calls.append(True)
    st.title("Trend Portfolio App")
    titles.append("Trend Portfolio App")

    cfg_dict = st.session_state.setdefault("config_dict", _read_defaults())

    with st.sidebar:
        _render_sidebar(cfg_dict)

    _render_run_section(cfg_dict)


if not _is_mock_streamlit(st):
    _render_app()
