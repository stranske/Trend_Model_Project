from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, cast

import pandas as pd
import streamlit as st

import yaml  # type: ignore[import-untyped]
from trend_analysis import pipeline
from trend_analysis.config import DEFAULTS as DEFAULT_CFG_PATH
from trend_analysis.config import Config
from trend_analysis.multi_period import run as run_multi

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

    def __exit__(self, *_exc: Any) -> bool:  # pragma: no cover - trivial
        return False


def _is_mock_streamlit(module: Any) -> bool:
    """Detect the lightweight ``MagicMock`` used by helper unit tests."""

    return type(module).__name__ == "MagicMock"


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


def _to_yaml(d: Dict[str, Any]) -> str:
    """Serialise a mapping to YAML while preserving insertion order."""

    dumped = yaml.safe_dump(d, sort_keys=False, allow_unicode=True)
    return cast(str, dumped)


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


def _summarise_multi(results: List[Dict[str, Any]]) -> pd.DataFrame:
    """Create a tidy summary table from multi-period back-test results."""

    rows: List[Dict[str, Any]] = []

    def _period_value(period: Any, index: int) -> Any:
        if period is None:
            return ""
        if isinstance(period, (list, tuple)):
            return period[index] if len(period) > index else ""
        try:
            seq = list(period)  # type: ignore[arg-type]
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
            out_df = pipeline.run(cfg_obj)
        summary = _summarise_run_df(out_df)
        st.success(f"Completed. {len(summary)} rows.")
        if not summary.empty:
            st.dataframe(summary, use_container_width=True)
            csv_bytes = summary.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download CSV",
                data=csv_bytes,
                file_name="single_period_summary.csv",
                mime="text/csv",
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
    st.set_page_config(page_title="Trend Portfolio App", layout="wide")
    st.title("Trend Portfolio App")

    cfg_dict = st.session_state.setdefault("config_dict", _read_defaults())

    with st.sidebar:
        _render_sidebar(cfg_dict)

    _render_run_section(cfg_dict)


if not _is_mock_streamlit(st):
    _render_app()
