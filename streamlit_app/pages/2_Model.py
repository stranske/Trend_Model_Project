"""Model configuration page for the Streamlit application."""

from __future__ import annotations

import contextlib
import difflib
import hashlib
import html
import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import dataclass
from datetime import date, datetime
from time import monotonic, sleep
from typing import Any, Mapping

import streamlit as st
import yaml

from streamlit_app import state as app_state
from streamlit_app.components import analysis_runner, nl_operation_viewer
from streamlit_app.components.llm_settings import sanitize_api_key as _sanitize_api_key
from streamlit_app.components.progress_eta import (
    estimate_eta_seconds,
    progress_ratio_and_remaining,
    update_eta_seconds,
)
from trend_analysis.config.patch import apply_config_patch, diff_configs
from trend_analysis.llm import (
    ConfigPatchChain,
    LLMProviderConfig,
    build_config_patch_prompt,
    create_llm,
)
from trend_analysis.llm.schema import load_compact_schema

# Extended metric fields for ranking
METRIC_FIELDS = [
    ("Sharpe Ratio", "sharpe"),
    ("Annual Return", "return_ann"),
    ("Sortino Ratio", "sortino"),
    ("Info Ratio", "info_ratio"),
    ("Max Drawdown", "drawdown"),
    ("Volatility", "vol"),
]

# Available weighting schemes from the plugin registry
WEIGHTING_SCHEMES = [
    ("Equal Weight (1/N)", "equal"),
    ("Risk Parity (inverse vol)", "risk_parity"),
    ("Hierarchical Risk Parity", "hrp"),
    ("Equal Risk Contribution", "erc"),
    ("Robust Mean-Variance", "robust_mv"),
    ("Robust Risk Parity", "robust_risk_parity"),
]


# Config chat panel helpers
_CONFIG_HISTORY_KEY = "config_chat_history"
_MAX_CONFIG_HISTORY = 20
_CONFIG_PREVIEW_TIMINGS_KEY = "config_chat_preview_timings"
_MAX_CONFIG_PREVIEW_TIMINGS = 20
_CONFIG_CHAIN_STATE_KEY = "config_chat_chain_state"
_DEFAULT_CONFIG_CHAT_PROVIDER = "openai"
_DEFAULT_CONFIG_CHAT_MODEL = "gpt-4o-mini"
_CONFIG_CHAIN_CACHE_VERSION = "v1"
_CONFIG_CHAIN_METRICS_KEY = "config_chat_chain_metrics"
_CONFIG_CHAIN_STATS_KEY = "config_chat_chain_stats"
_CONFIG_CHAIN_KEY_FIELDS = (
    "provider",
    "model",
    "temperature",
)
_CONFIG_CHAIN_INVALIDATION_FIELDS = (
    "provider",
    "model",
    "base_url",
    "organization",
    "temperature",
)
_CONFIG_CHAIN_LLM_FIELDS = (
    "timeout",
    "max_retries",
    "extra_payload_hash",
    "api_key_fingerprint",
)
_LOGGER = logging.getLogger(__name__)
_CONFIG_CHAIN_CORE_FIELDS = _CONFIG_CHAIN_KEY_FIELDS
_CONFIG_CHAIN_REBUILD_FIELDS = _CONFIG_CHAIN_INVALIDATION_FIELDS
_CONFIG_CHAIN_RESET_FIELDS = _CONFIG_CHAIN_INVALIDATION_FIELDS
_LLM_PROVIDER_OVERRIDE_KEY = "llm_provider_override"
_LLM_MODEL_OVERRIDE_KEY = "llm_model_override"
_LLM_BASE_URL_OVERRIDE_KEY = "llm_base_url_override"
_LLM_ORG_OVERRIDE_KEY = "llm_org_override"
_LLM_TEMPERATURE_OVERRIDE_KEY = "llm_temperature_override"
_LLM_OVERRIDE_SNAPSHOT_KEY = "llm_override_snapshot"


def _get_chain_cache_state() -> dict[str, Any]:
    state = st.session_state.get(_CONFIG_CHAIN_STATE_KEY)
    if not isinstance(state, dict):
        state = {"entries": {}}
        st.session_state[_CONFIG_CHAIN_STATE_KEY] = state
    entries = state.get("entries")
    if not isinstance(entries, dict):
        state["entries"] = {}
    return state


def _chain_cache_signature(cache_key: Mapping[str, Any]) -> str:
    return json.dumps(dict(cache_key), sort_keys=True, default=str)


def _derive_cache_signature(
    provider: str,
    model: str,
    base_url: str | None,
    organization: str | None,
    temperature: float,
) -> str:
    normalized_provider = _normalize_cache_str(provider) or "openai"
    normalized_model = _normalize_cache_str(model) or _DEFAULT_CONFIG_CHAT_MODEL
    normalized_base_url = _normalize_cache_str(base_url)
    normalized_org = _normalize_cache_str(organization)
    normalized_temperature = _normalize_temperature(temperature)
    cache_key = {
        "cache_version": _CONFIG_CHAIN_CACHE_VERSION,
        "provider": normalized_provider,
        "model": normalized_model,
        "base_url": normalized_base_url,
        "organization": normalized_org,
        "temperature": normalized_temperature,
    }
    return _chain_cache_signature(cache_key)


def _chain_cache_signature_from_inputs(
    provider: str,
    model: str,
    base_url: str | None,
    organization: str | None,
    temperature: float,
) -> str:
    return _derive_cache_signature(provider, model, base_url, organization, temperature)


def _chain_resource_signature(
    chain_cache_key: Mapping[str, Any],
    llm_cache_key: Mapping[str, Any],
) -> str:
    return _chain_cache_signature({"chain": dict(chain_cache_key), "llm": dict(llm_cache_key)})


def _chain_cache_summary(
    cache_key: Mapping[str, Any],
    llm_cache_key: Mapping[str, Any] | None = None,
) -> str:
    provider = cache_key.get("provider") or "default"
    model = cache_key.get("model") or "default"
    temperature = cache_key.get("temperature")
    base_url = None
    organization = None
    if isinstance(llm_cache_key, Mapping):
        base_url = llm_cache_key.get("base_url")
        organization = llm_cache_key.get("organization")
    summary = f"{provider}:{model}@{_format_value(temperature)}"
    if base_url:
        summary = f"{summary} | base_url={base_url}"
    if organization:
        summary = f"{summary} | org={organization}"
    return summary


def _build_chain_cache_key(
    *,
    provider: str,
    model: str,
    temperature: float,
) -> dict[str, Any]:
    return {
        "cache_version": _CONFIG_CHAIN_CACHE_VERSION,
        "provider": provider,
        "model": model,
        "temperature": temperature,
    }


def _build_chain_cache_keys(
    *,
    provider: str,
    model: str,
    base_url: str | None,
    organization: str | None,
    temperature: float,
    timeout: float | None,
    max_retries: int | None,
    extra_payload_hash: str | None,
    api_key_fingerprint: str | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    chain_cache_key = _build_chain_cache_key(
        provider=provider,
        model=model,
        temperature=temperature,
    )
    llm_cache_key = _build_llm_cache_key(
        provider=provider,
        model=model,
        base_url=base_url,
        organization=organization,
        timeout=timeout,
        max_retries=max_retries,
        extra_payload_hash=extra_payload_hash,
        api_key_fingerprint=api_key_fingerprint,
    )
    return chain_cache_key, llm_cache_key


def _build_llm_cache_key(
    *,
    provider: str,
    model: str,
    base_url: str | None,
    organization: str | None,
    timeout: float | None,
    max_retries: int | None,
    extra_payload_hash: str | None,
    api_key_fingerprint: str | None,
) -> dict[str, Any]:
    return {
        "cache_version": _CONFIG_CHAIN_CACHE_VERSION,
        "provider": provider,
        "model": model,
        "base_url": base_url,
        "organization": organization,
        "timeout": timeout,
        "max_retries": max_retries,
        "extra_payload_hash": extra_payload_hash,
        "api_key_fingerprint": api_key_fingerprint,
    }

def _cache_key_changes(
    previous: Mapping[str, Any] | None,
    current: Mapping[str, Any],
    fields: tuple[str, ...],
) -> list[str]:
    if not isinstance(previous, Mapping):
        return []
    changed = []
    for field in fields:
        if previous.get(field) != current.get(field):
            changed.append(field)
    return changed


def _merge_cache_keys(
    cache_key: Mapping[str, Any] | None,
    llm_cache_key: Mapping[str, Any] | None,
) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    if isinstance(cache_key, Mapping):
        merged.update(cache_key)
    if isinstance(llm_cache_key, Mapping):
        merged.update(llm_cache_key)
    return merged


def _get_config_change_history() -> list[dict[str, Any]]:
    history = st.session_state.get(_CONFIG_HISTORY_KEY)
    if not isinstance(history, list):
        history = []
        st.session_state[_CONFIG_HISTORY_KEY] = history
    return history


def _record_config_change(preview: Mapping[str, Any]) -> None:
    before = preview.get("before")
    after = preview.get("after")
    if not isinstance(before, Mapping) or not isinstance(after, Mapping):
        return
    entry = {
        "timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "instruction": preview.get("instruction"),
        "summary": preview.get("summary"),
        "risk_flags": list(preview.get("risk_flags") or []),
        "before": deepcopy(dict(before)),
        "after": deepcopy(dict(after)),
        "diff": preview.get("diff"),
    }
    history = _get_config_change_history()
    history.append(entry)
    if len(history) > _MAX_CONFIG_HISTORY:
        del history[:-_MAX_CONFIG_HISTORY]


def _record_preview_timing(preview: Mapping[str, Any], total_seconds: float) -> None:
    timings = preview.get("timings")
    if not isinstance(timings, Mapping):
        return
    chain_key = timings.get("chain_cache_key")
    provider = None
    model = None
    temperature = None
    invalidation_fields: list[str] | None = None
    if isinstance(chain_key, Mapping):
        provider = chain_key.get("provider")
        model = chain_key.get("model")
        temperature = chain_key.get("temperature")
    invalidation_fields_raw = timings.get("chain_cache_invalidation_fields")
    if isinstance(invalidation_fields_raw, list):
        invalidation_fields = [str(field) for field in invalidation_fields_raw]
    llm_changed_fields_raw = timings.get("chain_llm_changed_fields")
    llm_changed_fields: list[str] | None = None
    if isinstance(llm_changed_fields_raw, list):
        llm_changed_fields = [str(field) for field in llm_changed_fields_raw]
    entry = {
        "timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "instruction": preview.get("instruction"),
        "provider": provider,
        "model": model,
        "temperature": temperature,
        "cache_signature": timings.get("chain_cache_signature"),
        "resource_signature": timings.get("chain_resource_signature"),
        "cache_summary": timings.get("chain_cache_summary"),
        "cache_miss_reason": timings.get("chain_cache_miss_reason"),
        "cache_invalidation_fields": invalidation_fields,
        "cache_llm_changed_fields": llm_changed_fields,
        "cache_settings_changed": timings.get("chain_settings_changed"),
        "cache_session_reset": timings.get("chain_cache_session_reset"),
        "chain_build_seconds": timings.get("chain_build_seconds"),
        "chain_lookup_seconds": timings.get("chain_lookup_seconds"),
        "chain_reused": timings.get("chain_reused"),
        "run_seconds": timings.get("run_seconds"),
        "total_seconds": total_seconds,
    }
    history = st.session_state.get(_CONFIG_PREVIEW_TIMINGS_KEY)
    if not isinstance(history, list):
        history = []
        st.session_state[_CONFIG_PREVIEW_TIMINGS_KEY] = history
    history.append(entry)
    if len(history) > _MAX_CONFIG_PREVIEW_TIMINGS:
        del history[:-_MAX_CONFIG_PREVIEW_TIMINGS]

    if _LOGGER.isEnabledFor(logging.INFO):
        _LOGGER.info(
            (
                "Config chat preview timing: reused=%s build=%.2fs run=%.2fs "
                "total=%.2fs cache=%s miss=%s invalidated_by=%s"
            ),
            "yes" if entry.get("chain_reused") else "no",
            float(entry.get("chain_build_seconds") or 0.0),
            float(entry.get("run_seconds") or 0.0),
            total_seconds,
            entry.get("cache_summary") or "unknown",
            entry.get("cache_miss_reason") or "none",
            ", ".join(entry.get("cache_invalidation_fields") or []) or "none",
        )
    st.session_state[_CONFIG_CHAIN_METRICS_KEY] = {
        "timestamp": entry.get("timestamp"),
        "provider": provider,
        "model": model,
        "temperature": temperature,
        "cache_signature": timings.get("chain_cache_signature"),
        "resource_signature": timings.get("chain_resource_signature"),
        "cache_summary": timings.get("chain_cache_summary"),
        "cache_miss_reason": timings.get("chain_cache_miss_reason"),
        "cache_invalidation_fields": invalidation_fields,
        "cache_llm_changed_fields": llm_changed_fields,
        "cache_settings_changed": timings.get("chain_settings_changed"),
        "cache_session_reset": timings.get("chain_cache_session_reset"),
        "chain_reused": timings.get("chain_reused"),
        "chain_build_seconds": timings.get("chain_build_seconds"),
        "chain_lookup_seconds": timings.get("chain_lookup_seconds"),
        "run_seconds": timings.get("run_seconds"),
        "total_seconds": total_seconds,
    }
    _LOGGER.info(
        "Config chat preview: provider=%s model=%s temp=%s reused=%s build_s=%s run_s=%s total_s=%s "
        "cache_miss=%s invalidated_by=%s",
        provider,
        model,
        temperature,
        entry.get("chain_reused"),
        entry.get("chain_build_seconds"),
        entry.get("run_seconds"),
        total_seconds,
        entry.get("cache_miss_reason"),
        invalidation_fields,
    )


def _record_chain_cache_stats(
    chain_meta: Mapping[str, Any],
    chain_build_seconds: float,
) -> None:
    stats = st.session_state.get(_CONFIG_CHAIN_STATS_KEY)
    if not isinstance(stats, Mapping):
        stats = {"hits": 0, "misses": 0}
    stats = dict(stats)
    reused = bool(chain_meta.get("chain_reused"))
    if reused:
        stats["hits"] = int(stats.get("hits", 0)) + 1
    else:
        stats["misses"] = int(stats.get("misses", 0)) + 1
    stats["last_build_seconds"] = chain_build_seconds
    stats["last_reused"] = reused
    stats["last_miss_reason"] = chain_meta.get("chain_cache_miss_reason")
    stats["last_invalidation_fields"] = chain_meta.get("chain_cache_invalidation_fields")
    stats["last_signature"] = chain_meta.get("chain_cache_signature")
    stats["timestamp"] = datetime.utcnow().isoformat(timespec="seconds") + "Z"
    st.session_state[_CONFIG_CHAIN_STATS_KEY] = stats


def _format_seconds(value: Any) -> str:
    if value is None:
        return "—"
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "—"
    return f"{numeric:.2f}s"


def _preview_timing_summary(
    timings: Mapping[str, Any] | None,
    total_seconds: float | None = None,
) -> str:
    if not isinstance(timings, Mapping):
        return "Preview timing unavailable."
    cache_hit = "hit" if timings.get("chain_reused") else "miss"
    build = _format_seconds(timings.get("chain_build_seconds"))
    lookup = _format_seconds(timings.get("chain_lookup_seconds"))
    run = _format_seconds(timings.get("run_seconds"))
    total_value = total_seconds if total_seconds is not None else timings.get("total_seconds")
    total = _format_seconds(total_value)
    return (
        "Preview timing: "
        f"cache {cache_hit} | build {build} | lookup {lookup} | run {run} | total {total}"
    )


def _render_preview_timing_history() -> None:
    st.markdown("**Preview timings**")
    history = st.session_state.get(_CONFIG_PREVIEW_TIMINGS_KEY)
    if not isinstance(history, list) or not history:
        st.info("No preview timing data yet.")
        return

    rows: list[dict[str, str]] = []
    for entry in reversed(history):
        if not isinstance(entry, Mapping):
            continue
        cache_sig = entry.get("cache_signature")
        cache_sig_label = str(cache_sig)[:8] if cache_sig else "—"
        resource_sig = entry.get("resource_signature")
        resource_sig_label = str(resource_sig)[:8] if resource_sig else "—"
        cache_summary = entry.get("cache_summary") or "—"
        cache_reason = entry.get("cache_miss_reason")
        cache_reason_label = str(cache_reason) if cache_reason else "—"
        invalidation_fields = entry.get("cache_invalidation_fields")
        invalidation_label = (
            ", ".join(invalidation_fields) if isinstance(invalidation_fields, list) else "—"
        )
        llm_changed_fields = entry.get("cache_llm_changed_fields")
        llm_changed_label = (
            ", ".join(llm_changed_fields) if isinstance(llm_changed_fields, list) else "—"
        )
        settings_changed = entry.get("cache_settings_changed")
        settings_changed_label = (
            "Yes" if settings_changed is True else "No" if settings_changed is False else "—"
        )
        cache_reset = entry.get("cache_session_reset")
        cache_reset_label = "Yes" if cache_reset else "No"
        rows.append(
            {
                "Timestamp": str(entry.get("timestamp") or "Unknown time"),
                "Instruction": str(entry.get("instruction") or "Preview"),
                "Provider": str(entry.get("provider") or "default"),
                "Model": str(entry.get("model") or "default"),
                "Temp": _format_value(entry.get("temperature")),
                "Settings changed": settings_changed_label,
                "Session cache reset": cache_reset_label,
                "Cache key": str(cache_summary),
                "Cache sig": cache_sig_label,
                "Resource sig": resource_sig_label,
                "Cache miss": cache_reason_label,
                "Cache invalidated by": invalidation_label,
                "LLM changed": llm_changed_label,
                "Chain build": _format_seconds(entry.get("chain_build_seconds")),
                "Chain lookup": _format_seconds(entry.get("chain_lookup_seconds")),
                "Chain reused": "Yes" if entry.get("chain_reused") else "No",
                "Run": _format_seconds(entry.get("run_seconds")),
                "Total": _format_seconds(entry.get("total_seconds")),
            }
        )
    if not rows:
        st.info("No preview timing data yet.")
        return
    st.dataframe(rows, use_container_width=True, hide_index=True)


def _render_last_preview_metrics() -> None:
    metrics = st.session_state.get(_CONFIG_CHAIN_METRICS_KEY)
    if not isinstance(metrics, Mapping):
        return
    cache_sig = metrics.get("cache_signature")
    cache_label = str(cache_sig)[:8] if cache_sig else "—"
    resource_sig = metrics.get("resource_signature")
    resource_label = str(resource_sig)[:8] if resource_sig else "—"
    cache_summary = metrics.get("cache_summary") or "—"
    chain_reused = "Yes" if metrics.get("chain_reused") else "No"
    cache_miss = metrics.get("cache_miss_reason")
    cache_miss_label = str(cache_miss) if cache_miss else "—"
    invalidation_fields = metrics.get("cache_invalidation_fields")
    invalidation_label = (
        ", ".join(invalidation_fields) if isinstance(invalidation_fields, list) else "—"
    )
    llm_changed_fields = metrics.get("cache_llm_changed_fields")
    llm_changed_label = (
        ", ".join(llm_changed_fields) if isinstance(llm_changed_fields, list) else "—"
    )
    settings_changed = metrics.get("cache_settings_changed")
    settings_changed_label = (
        "Yes" if settings_changed is True else "No" if settings_changed is False else "—"
    )
    cache_reset = metrics.get("cache_session_reset")
    cache_reset_label = "Yes" if cache_reset else "No"
    st.caption(
        "Last preview cache — "
        f"Key: {cache_summary} | "
        f"Sig: {cache_label} | "
        f"Resource sig: {resource_label} | "
        f"Chain reused: {chain_reused} | "
        f"Cache miss: {cache_miss_label} | "
        f"Invalidated by: {invalidation_label} | "
        f"LLM changed: {llm_changed_label} | "
        f"Settings changed: {settings_changed_label} | "
        f"Session reset: {cache_reset_label} | "
        f"Build: {_format_seconds(metrics.get('chain_build_seconds'))} | "
        f"Lookup: {_format_seconds(metrics.get('chain_lookup_seconds'))} | "
        f"Run: {_format_seconds(metrics.get('run_seconds'))} | "
        f"Total: {_format_seconds(metrics.get('total_seconds'))}"
    )
    stats = st.session_state.get(_CONFIG_CHAIN_STATS_KEY)
    if not isinstance(stats, Mapping):
        return
    hits = int(stats.get("hits", 0))
    misses = int(stats.get("misses", 0))
    total = hits + misses
    hit_rate = f"{(hits / total) * 100:.0f}%" if total else "—"
    last_invalidation = stats.get("last_invalidation_fields")
    last_invalidation_label = (
        ", ".join(last_invalidation) if isinstance(last_invalidation, list) else "—"
    )
    st.caption(
        "Cache stats — "
        f"Hits: {hits} | Misses: {misses} | Hit rate: {hit_rate} | "
        f"Last build: {_format_seconds(stats.get('last_build_seconds'))} | "
        f"Last invalidation: {last_invalidation_label}"
    )


def _format_percent(value: Any) -> str:
    if value is None:
        return "—"
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "—"
    return f"{numeric * 100:.1f}%"


def _format_value(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "Yes" if value else "No"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def _render_validation_error_styles() -> None:
    st.markdown(
        """
<style>
.validation-errors {
  border: 1px solid #fecaca;
  background: #fff1f2;
  color: #991b1b;
  padding: 0.75rem 1rem;
  border-radius: 6px;
}
.validation-errors ul {
  margin: 0.25rem 0 0 1.25rem;
}
</style>
""",
        unsafe_allow_html=True,
    )


def _render_validation_errors(errors: list[str]) -> None:
    if not errors:
        return
    _render_validation_error_styles()
    items = "".join(f"<li>{html.escape(err)}</li>" for err in errors)
    st.markdown(
        f'<div class="validation-errors"><strong>Validation errors</strong><ul>{items}</ul></div>',
        unsafe_allow_html=True,
    )


def _config_summary_sections(
    model_state: Mapping[str, Any],
) -> list[tuple[str, list[tuple[str, str]]]]:
    return [
        (
            "Overview",
            [
                ("Preset", _format_value(model_state.get("preset"))),
                ("Weighting", _format_value(model_state.get("weighting_scheme"))),
                ("Selection count", _format_value(model_state.get("selection_count"))),
            ],
        ),
        (
            "Time Windows",
            [
                (
                    "Lookback periods",
                    _format_value(model_state.get("lookback_periods")),
                ),
                (
                    "Evaluation periods",
                    _format_value(model_state.get("evaluation_periods")),
                ),
                ("Min history", _format_value(model_state.get("min_history_periods"))),
                ("Frequency", _format_value(model_state.get("multi_period_frequency"))),
            ],
        ),
        (
            "Risk + Constraints",
            [
                ("Risk target", _format_percent(model_state.get("risk_target"))),
                ("Max weight", _format_percent(model_state.get("max_weight"))),
                ("Min weight", _format_percent(model_state.get("min_weight"))),
                ("Max turnover", _format_percent(model_state.get("max_turnover"))),
            ],
        ),
        (
            "Signals",
            [
                ("Trend window", _format_value(model_state.get("trend_window"))),
                ("Trend lag", _format_value(model_state.get("trend_lag"))),
                ("Vol adjust", _format_value(model_state.get("vol_adjust_enabled"))),
            ],
        ),
    ]


def _render_config_summary(model_state: Mapping[str, Any] | None) -> None:
    if not model_state:
        st.info("No configuration loaded yet.")
        return

    for title, rows in _config_summary_sections(model_state):
        st.markdown(f"**{title}**")
        for label, value in rows:
            st.markdown(f"- {label}: {value}")


def _config_status_label(model_state: Mapping[str, Any] | None) -> tuple[str, str]:
    active_name = st.session_state.get("active_saved_model_name")
    if not model_state:
        return "Not loaded", "⚪"
    if not active_name:
        return "Unsaved", "🟡"
    snapshot = st.session_state.get("last_loaded_model_state")
    if isinstance(snapshot, Mapping):
        diffs = app_state.diff_model_states(snapshot, model_state)
        if diffs:
            return f"Modified (from '{active_name}')", "🟠"
        return f"In sync with '{active_name}'", "🟢"
    return f"Saved '{active_name}'", "🟢"


def _render_config_status_badge(model_state: Mapping[str, Any] | None) -> None:
    label, icon = _config_status_label(model_state)
    st.caption(f"{icon} Config status: {label}")


def _build_config_wrapper(model_state: Mapping[str, Any]) -> dict[str, Any]:
    wrapper: dict[str, Any] = {"model_state": dict(model_state)}
    for key in (
        "analysis_fund_columns",
        "fund_columns",
        "data_fingerprint",
        "data_loaded_key",
        "selected_benchmark",
        "selected_risk_free",
        "uploaded_filename",
    ):
        value = st.session_state.get(key)
        if value is not None:
            wrapper[key] = value
    return wrapper


_MODEL_WIDGET_KEYS = {
    "date_mode_radio",
    "sim_start_date",
    "sim_end_date",
    "preset_selector",
    "weighting_scheme_selector",
    "inclusion_approach_select",
    "buy_hold_initial_select",
    "rank_pct_primary",
    "mp_min_funds_input",
    "mp_max_funds_input",
    "benchmark_selector",
}


def _reset_model_widget_state() -> None:
    for key in _MODEL_WIDGET_KEYS:
        st.session_state.pop(key, None)
    for key in list(st.session_state.keys()):
        if isinstance(key, str) and key.startswith("metric_"):
            st.session_state.pop(key, None)


def _parse_date(value: Any) -> date | None:
    if isinstance(value, date):
        return value
    if isinstance(value, str) and value:
        try:
            return datetime.strptime(value[:10], "%Y-%m-%d").date()
        except ValueError:
            return None
    return None


def _sync_model_widgets_from_state(model_state: Mapping[str, Any]) -> None:
    """Sync widget session_state keys from model_state values.

    This ensures widgets with explicit keys reflect imported config values.
    """
    date_mode = model_state.get("date_mode")
    if date_mode in {"relative", "explicit"}:
        st.session_state["date_mode_radio"] = date_mode

    start_date = _parse_date(model_state.get("start_date"))
    if start_date is not None:
        st.session_state["sim_start_date"] = start_date

    end_date = _parse_date(model_state.get("end_date"))
    if end_date is not None:
        st.session_state["sim_end_date"] = end_date

    preset = model_state.get("preset")
    if isinstance(preset, str) and preset:
        st.session_state["preset_selector"] = preset

    weighting = model_state.get("weighting_scheme")
    if isinstance(weighting, str) and weighting:
        st.session_state["weighting_scheme_selector"] = weighting

    inclusion = model_state.get("inclusion_approach")
    if isinstance(inclusion, str) and inclusion:
        st.session_state["inclusion_approach_select"] = inclusion

    buy_hold = model_state.get("buy_hold_initial")
    if isinstance(buy_hold, str) and buy_hold:
        st.session_state["buy_hold_initial_select"] = buy_hold

    rank_pct = model_state.get("rank_pct")
    if isinstance(rank_pct, (int, float)):
        st.session_state["rank_pct_primary"] = int(float(rank_pct) * 100)

    mp_min = model_state.get("mp_min_funds")
    if isinstance(mp_min, (int, float)):
        st.session_state["mp_min_funds_input"] = int(mp_min)

    mp_max = model_state.get("mp_max_funds")
    if isinstance(mp_max, (int, float)):
        st.session_state["mp_max_funds_input"] = int(mp_max)

    info_benchmark = model_state.get("info_ratio_benchmark")
    if isinstance(info_benchmark, str) and info_benchmark:
        st.session_state["benchmark_selector"] = info_benchmark

    metric_weights = model_state.get("metric_weights")
    if isinstance(metric_weights, Mapping):
        for key, value in metric_weights.items():
            st.session_state[f"metric_{key}"] = value


def _apply_config_wrapper(wrapper: Mapping[str, Any]) -> None:
    model_state = wrapper.get("model_state")
    if isinstance(model_state, Mapping):
        st.session_state["model_state"] = dict(model_state)
        st.session_state["last_loaded_model_state"] = dict(model_state)
        _reset_model_widget_state()
        _sync_model_widgets_from_state(model_state)
    for key in (
        "analysis_fund_columns",
        "fund_columns",
        "data_fingerprint",
        "data_loaded_key",
        "selected_benchmark",
        "selected_risk_free",
        "uploaded_filename",
    ):
        if key in wrapper:
            st.session_state[key] = wrapper[key]


def _resolve_llm_provider_config() -> LLMProviderConfig:
    overrides = _resolve_llm_session_overrides()
    provider_override = overrides.get("provider")
    provider_name = (provider_override or os.environ.get("TREND_LLM_PROVIDER") or "openai").lower()
    supported = {"openai", "anthropic", "ollama"}
    if provider_name not in supported:
        raise ValueError(
            f"Unknown LLM provider '{provider_name}'. "
            f"Expected one of: {', '.join(sorted(supported))}."
        )
    api_key = _sanitize_api_key(os.environ.get("TS_STREAMLIT_API_KEY"))
    if not api_key:
        api_key = _sanitize_api_key(os.environ.get("OPENAI_API_KEY"))
    if not api_key:
        api_key = _sanitize_api_key(os.environ.get("TREND_LLM_API_KEY"))
    if not api_key and provider_name == "anthropic":
        api_key = _sanitize_api_key(os.environ.get("ANTHROPIC_API_KEY"))
    model = overrides.get("model") or os.environ.get("TREND_LLM_MODEL")
    base_url = overrides.get("base_url") or os.environ.get("TREND_LLM_BASE_URL")
    organization = overrides.get("organization") or os.environ.get("TREND_LLM_ORG")
    kwargs: dict[str, Any] = {"provider": provider_name}
    if model:
        kwargs["model"] = model
    if api_key:
        kwargs["api_key"] = api_key
    if base_url:
        kwargs["base_url"] = base_url
    if organization:
        kwargs["organization"] = organization
    return LLMProviderConfig(**kwargs)


def _resolve_llm_temperature() -> float:
    override = st.session_state.get(_LLM_TEMPERATURE_OVERRIDE_KEY)
    if override is not None:
        override_text = str(override).strip()
        if override_text:
            try:
                return float(override_text)
            except (TypeError, ValueError):
                pass
    raw = os.environ.get("TREND_LLM_TEMPERATURE")
    if raw is None:
        return 0.0
    try:
        return float(raw)
    except (TypeError, ValueError):
        return 0.0


def _hash_api_key(api_key: str | None) -> str | None:
    if not api_key:
        return None
    return hashlib.sha256(api_key.encode("utf-8")).hexdigest()


def _hash_text(value: str | None) -> str | None:
    if not value:
        return None
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]


def _serialize_extra(extra: Mapping[str, Any] | None) -> str:
    if not extra:
        return ""
    return json.dumps(dict(extra), sort_keys=True, default=str)


def _normalize_cache_str(value: str | None) -> str | None:
    if value is None:
        return None
    stripped = value.strip()
    return stripped or None


def _resolve_llm_session_overrides() -> dict[str, str | None]:
    provider = st.session_state.get(_LLM_PROVIDER_OVERRIDE_KEY)
    model = st.session_state.get(_LLM_MODEL_OVERRIDE_KEY)
    base_url = st.session_state.get(_LLM_BASE_URL_OVERRIDE_KEY)
    organization = st.session_state.get(_LLM_ORG_OVERRIDE_KEY)
    provider_override = _normalize_cache_str(str(provider)) if provider else None
    if provider_override:
        provider_override = provider_override.lower()
    return {
        "provider": provider_override,
        "model": _normalize_cache_str(str(model)) if model else None,
        "base_url": _normalize_cache_str(str(base_url)) if base_url else None,
        "organization": (_normalize_cache_str(str(organization)) if organization else None),
    }


def _current_chain_settings_snapshot(
    config: LLMProviderConfig | None = None,
    temperature: float | None = None,
) -> dict[str, Any]:
    if config is None:
        config = _resolve_llm_provider_config()
    if temperature is None:
        temperature = _resolve_llm_temperature()
    return {
        "provider": _normalize_cache_str(config.provider),
        "model": _normalize_cache_str(config.model),
        "base_url": _normalize_cache_str(config.base_url),
        "organization": _normalize_cache_str(config.organization),
        "temperature": _normalize_temperature(temperature),
    }


def _maybe_reset_config_chat_cache(snapshot: Mapping[str, Any]) -> list[str]:
    previous = st.session_state.get(_LLM_OVERRIDE_SNAPSHOT_KEY)
    normalized = dict(snapshot)
    if not isinstance(previous, Mapping):
        st.session_state[_LLM_OVERRIDE_SNAPSHOT_KEY] = normalized
        return []
    previous_normalized = {key: previous.get(key) for key in normalized}
    if previous_normalized == normalized:
        return []
    changed = [key for key in normalized if previous_normalized.get(key) != normalized.get(key)]
    st.session_state[_LLM_OVERRIDE_SNAPSHOT_KEY] = normalized
    st.session_state.pop("config_chat_preview", None)
    st.session_state.pop("config_chat_last_instruction", None)
    st.session_state.pop(_CONFIG_CHAIN_STATE_KEY, None)
    st.session_state.pop(_CONFIG_CHAIN_METRICS_KEY, None)
    st.session_state.pop(_CONFIG_CHAIN_STATS_KEY, None)
    _LOGGER.info(
        "Config chat cache reset due to settings change: %s -> %s",
        previous_normalized,
        normalized,
    )
    return changed


def _llm_required_env_vars(provider: str) -> list[str] | None:
    required = ["TS_STREAMLIT_API_KEY", "TREND_LLM_API_KEY"]
    if provider == "openai":
        required.append("OPENAI_API_KEY")
    elif provider == "anthropic":
        required.append("ANTHROPIC_API_KEY")
    elif provider == "ollama":
        pass
    else:
        _LOGGER.warning("Unknown LLM provider for env var requirements: %s", provider)
        return None
    return required


def _llm_env_var_present(name: str) -> bool:
    value = os.environ.get(name)
    if name in {
        "TS_STREAMLIT_API_KEY",
        "TREND_LLM_API_KEY",
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
    }:
        return bool(_sanitize_api_key(value))
    return bool(value)


def _llm_env_var_status(provider: str) -> dict[str, bool]:
    required = _llm_required_env_vars(provider)
    if not required:
        return {}
    return {name: _llm_env_var_present(name) for name in required}


def _render_llm_status_panel() -> None:
    provider_labels = {
        "openai": "OpenAI",
        "anthropic": "Anthropic",
        "ollama": "Ollama",
    }
    selected_provider = _normalize_cache_str(st.session_state.get("selected_provider"))
    selected_provider = (selected_provider or _DEFAULT_CONFIG_CHAT_PROVIDER).lower()
    selected_model = (
        _normalize_cache_str(st.session_state.get("selected_model")) or _DEFAULT_CONFIG_CHAT_MODEL
    )
    provider_label = provider_labels.get(selected_provider, selected_provider)
    st.info(f"Active provider: {provider_label}")
    st.info(f"Active model: {selected_model}")
    required_vars = _llm_required_env_vars(selected_provider)
    if required_vars is None:
        st.warning(f"Unknown provider: {selected_provider}. Update your LLM settings.")
        return
    if not required_vars:
        st.caption("Expected environment variables: None required.")
        return
    missing_vars = [name for name in required_vars if not _llm_env_var_present(name)]
    st.caption("Expected environment variables (values hidden):")
    for name in required_vars:
        icon = "✓" if name not in missing_vars else "✗"
        st.write(f"{icon} `{name}`")
    if missing_vars:
        missing_list = ", ".join(missing_vars)
        st.warning(
            f"Missing required environment variables for {provider_label}. " f"Set: {missing_list}."
        )


def _sync_llm_selection_from_overrides() -> None:
    provider_override = _normalize_cache_str(st.session_state.get(_LLM_PROVIDER_OVERRIDE_KEY))
    if provider_override:
        st.session_state["selected_provider"] = provider_override.lower()
    else:
        env_provider = _normalize_cache_str(os.environ.get("TREND_LLM_PROVIDER"))
        st.session_state["selected_provider"] = (
            env_provider or _DEFAULT_CONFIG_CHAT_PROVIDER
        ).lower()
    model_override = _normalize_cache_str(st.session_state.get(_LLM_MODEL_OVERRIDE_KEY))
    if model_override:
        st.session_state["selected_model"] = model_override
    else:
        env_model = _normalize_cache_str(os.environ.get("TREND_LLM_MODEL"))
        st.session_state["selected_model"] = env_model or _DEFAULT_CONFIG_CHAT_MODEL


def _render_llm_session_overrides_panel() -> None:
    with st.expander("LLM Settings (Session Only)", expanded=False):
        st.caption("Overrides apply only to this session and do not store or display API keys.")
        provider_options = [None, "openai", "anthropic", "ollama"]
        provider_labels = {
            None: "Use env default",
            "openai": "OpenAI",
            "anthropic": "Anthropic",
            "ollama": "Ollama",
        }
        current_override = st.session_state.get(_LLM_PROVIDER_OVERRIDE_KEY)
        if current_override not in provider_options:
            current_override = None
        st.selectbox(
            "Provider",
            provider_options,
            index=provider_options.index(current_override),
            key=_LLM_PROVIDER_OVERRIDE_KEY,
            format_func=lambda value: provider_labels.get(value, "Use env default"),
            help="Overrides TREND_LLM_PROVIDER for this session only.",
            on_change=_sync_llm_selection_from_overrides,
        )
        st.text_input(
            "Model (optional)",
            key=_LLM_MODEL_OVERRIDE_KEY,
            help="Overrides TREND_LLM_MODEL for this session only.",
            on_change=_sync_llm_selection_from_overrides,
        )
        st.text_input(
            "Base URL (optional)",
            key=_LLM_BASE_URL_OVERRIDE_KEY,
            help="Overrides TREND_LLM_BASE_URL for this session only.",
        )
        st.text_input(
            "Organization (optional)",
            key=_LLM_ORG_OVERRIDE_KEY,
            help="Overrides TREND_LLM_ORG for this session only.",
        )
        temp_override = st.text_input(
            "Temperature (optional)",
            key=_LLM_TEMPERATURE_OVERRIDE_KEY,
            help="Overrides TREND_LLM_TEMPERATURE for this session only.",
        )
        if temp_override:
            try:
                float(temp_override)
            except (TypeError, ValueError):
                st.warning("Temperature override must be a number; using env default.")
        _sync_llm_selection_from_overrides()
        _maybe_reset_config_chat_cache(_current_chain_settings_snapshot())


def _normalize_temperature(value: float) -> float:
    try:
        return round(float(value), 4)
    except (TypeError, ValueError):
        return 0.0


def _hash_cache_key(value: Mapping[str, Any]) -> str:
    return _chain_cache_signature(value)


@dataclass(frozen=True, slots=True)
class _ApiKeySecret:
    value: str | None
    fingerprint: str | None

    def __eq__(self, other: object) -> bool:
        if isinstance(other, _ApiKeySecret):
            return (self.value, self.fingerprint) == (other.value, other.fingerprint)
        if isinstance(other, str):
            return self.value == other
        if other is None:
            return self.value is None
        return False


def _hash_api_key_secret(secret: _ApiKeySecret) -> str:
    if secret.fingerprint:
        return secret.fingerprint
    return "no-key"


@st.cache_resource(show_spinner=False)
def _cached_compact_schema() -> dict[str, Any]:
    return load_compact_schema()


@st.cache_resource(
    show_spinner=False,
    hash_funcs={dict: _hash_cache_key, _ApiKeySecret: _hash_api_key_secret},
)
def _cached_llm_client(
    cache_key: Mapping[str, Any],
    api_key_secret: _ApiKeySecret | None,
    extra_payload: str,
) -> Any:
    api_key = api_key_secret.value if api_key_secret is not None else None
    config = LLMProviderConfig(
        provider=str(cache_key.get("provider")),
        model=str(cache_key.get("model")),
        api_key=api_key,
        base_url=cache_key.get("base_url"),
        organization=cache_key.get("organization"),
        timeout=cache_key.get("timeout"),
        max_retries=cache_key.get("max_retries"),
        extra=json.loads(extra_payload) if extra_payload else {},
    )
    return create_llm(config)


@st.cache_resource(
    show_spinner=False,
    hash_funcs={dict: _hash_cache_key, _ApiKeySecret: _hash_api_key_secret},
)
def _cached_config_patch_chain(
    chain_cache_key: Mapping[str, Any],
    llm_cache_key: Mapping[str, Any],
    api_key_secret: _ApiKeySecret | None,
    extra_payload: str,
) -> ConfigPatchChain:
    llm = _cached_llm_client(
        llm_cache_key,
        api_key_secret,
        extra_payload,
    )
    schema = _cached_compact_schema()
    return ConfigPatchChain.from_env(
        llm=llm,
        schema=schema,
        prompt_builder=build_config_patch_prompt,
        temperature=float(chain_cache_key.get("temperature") or 0.0),
        model=str(chain_cache_key.get("model")),
    )


def _build_chain_config(config: LLMProviderConfig) -> dict[str, Any]:
    extra_payload = _serialize_extra(config.extra)
    return {
        "timeout": config.timeout,
        "max_retries": config.max_retries,
        "extra_payload": extra_payload,
        "extra_payload_hash": _hash_text(extra_payload),
        "api_key": config.api_key,
        "api_key_fingerprint": _hash_api_key(config.api_key),
    }


def _build_chain_cache_context(
    config: LLMProviderConfig | None = None,
    temperature: float | None = None,
) -> dict[str, Any]:
    if config is None:
        config = _resolve_llm_provider_config()
    if temperature is None:
        temperature = _resolve_llm_temperature()
    chain_config = _build_chain_config(config)
    provider = _normalize_cache_str(config.provider) or "openai"
    temperature = _normalize_temperature(temperature)
    base_url = _normalize_cache_str(config.base_url)
    organization = _normalize_cache_str(config.organization)
    normalized_model = _normalize_cache_str(config.model)
    resolved_model = normalized_model or _DEFAULT_CONFIG_CHAT_MODEL
    cache_key, llm_cache_key = _build_chain_cache_keys(
        provider=provider,
        model=resolved_model,
        base_url=base_url,
        organization=organization,
        temperature=temperature,
        timeout=chain_config["timeout"],
        max_retries=chain_config["max_retries"],
        extra_payload_hash=chain_config["extra_payload_hash"],
        api_key_fingerprint=chain_config["api_key_fingerprint"],
    )
    return {
        "provider": provider,
        "temperature": temperature,
        "base_url": base_url,
        "organization": organization,
        "resolved_model": resolved_model,
        "api_key": chain_config["api_key"],
        "api_key_fingerprint": chain_config["api_key_fingerprint"],
        "extra_payload": chain_config["extra_payload"],
        "llm_cache_key": llm_cache_key,
        "cache_key": cache_key,
    }


def _build_nl_chain(
    provider: str | None = None,
    model: str | None = None,
    base_url: str | None = None,
    organization: str | None = None,
    temperature: float | None = None,
    cache_signature: str | None = None,
    timeout: float | None = None,
    max_retries: int | None = None,
    api_key_secret: "_ApiKeySecret" | None = None,
    extra_payload: str | None = None,
) -> tuple[ConfigPatchChain, dict[str, Any]]:
    config = _resolve_llm_provider_config()
    resolved_provider = _normalize_cache_str(provider) or _normalize_cache_str(config.provider) or "openai"
    resolved_model = _normalize_cache_str(model) or _normalize_cache_str(config.model) or _DEFAULT_CONFIG_CHAT_MODEL
    resolved_base_url = (
        _normalize_cache_str(base_url)
        if base_url is not None
        else _normalize_cache_str(config.base_url)
    )
    resolved_org = (
        _normalize_cache_str(organization)
        if organization is not None
        else _normalize_cache_str(config.organization)
    )
    resolved_temp = _normalize_temperature(
        temperature if temperature is not None else _resolve_llm_temperature()
    )
    resolved_timeout = timeout if timeout is not None else config.timeout
    resolved_max_retries = max_retries if max_retries is not None else config.max_retries
    resolved_payload = extra_payload if extra_payload is not None else _serialize_extra(config.extra)
    resolved_payload_hash = _hash_text(resolved_payload)
    resolved_api_key_secret = api_key_secret or _ApiKeySecret(
        config.api_key,
        _hash_api_key(config.api_key),
    )
    reset_fields = _maybe_reset_config_chat_cache(
        _current_chain_settings_snapshot(config, resolved_temp)
    )
    llm_cache_key = _build_llm_cache_key(
        provider=resolved_provider,
        model=resolved_model,
        base_url=resolved_base_url,
        organization=resolved_org,
        timeout=resolved_timeout,
        max_retries=resolved_max_retries,
        extra_payload_hash=resolved_payload_hash,
        api_key_fingerprint=resolved_api_key_secret.fingerprint,
    )
    cache_key = _build_chain_cache_key(
        provider=resolved_provider,
        model=resolved_model,
        temperature=resolved_temp,
    )
    st.session_state["selected_provider"] = resolved_provider
    st.session_state["selected_model"] = resolved_model
    cache_state = _get_chain_cache_state()
    signature = cache_signature or _chain_cache_signature(cache_key)
    resource_signature = _chain_resource_signature(cache_key, llm_cache_key)
    previous_signature = cache_state.get("last_signature")
    previous_resource_signature = cache_state.get("last_resource_signature")
    previous_cache_key = cache_state.get("last_cache_key")
    previous_llm_cache_key = cache_state.get("last_llm_cache_key")
    previous_invalidation_key = None
    if isinstance(previous_cache_key, Mapping) or isinstance(previous_llm_cache_key, Mapping):
        previous_invalidation_key = _merge_cache_keys(previous_cache_key, previous_llm_cache_key)
    current_invalidation_key = _merge_cache_keys(cache_key, llm_cache_key)
    changed_fields = _cache_key_changes(previous_cache_key, cache_key, _CONFIG_CHAIN_CORE_FIELDS)
    llm_changed_fields = _cache_key_changes(
        previous_llm_cache_key,
        llm_cache_key,
        _CONFIG_CHAIN_LLM_FIELDS,
    )
    invalidation_fields = _cache_key_changes(
        previous_invalidation_key,
        current_invalidation_key,
        _CONFIG_CHAIN_REBUILD_FIELDS,
    )
    settings_changed = bool(
        (previous_signature and previous_signature != signature)
        or invalidation_fields
        or reset_fields
    )
    resource_changed = bool(
        previous_resource_signature and previous_resource_signature != resource_signature
    )
    session_reset = False
    if invalidation_fields:
        st.session_state.pop("config_chat_preview", None)
        st.session_state.pop("config_chat_last_instruction", None)
        cache_state["last_invalidation_fields"] = list(invalidation_fields)
        session_reset = True
    elif reset_fields:
        invalidation_fields = list(reset_fields)
        cache_state["last_invalidation_fields"] = list(invalidation_fields)
        session_reset = True
    api_key_secret = resolved_api_key_secret
    lookup_start = monotonic()
    chain = _cached_config_patch_chain(
        cache_key,
        llm_cache_key,
        api_key_secret,
        resolved_payload,
    )
    lookup_seconds = monotonic() - lookup_start
    entries = cache_state["entries"]
    cached_chain_id = entries.get(signature)
    reused = cached_chain_id == id(chain)
    cache_miss_reason = None
    if not reused:
        if settings_changed:
            if invalidation_fields:
                cache_miss_reason = f"settings_changed: {', '.join(invalidation_fields)}"
            elif changed_fields:
                cache_miss_reason = f"settings_changed: {', '.join(changed_fields)}"
            else:
                cache_miss_reason = "settings_changed"
        elif llm_changed_fields:
            cache_miss_reason = f"llm_settings_changed: {', '.join(llm_changed_fields)}"
        elif resource_changed:
            cache_miss_reason = "llm_settings_changed"
        else:
            cache_miss_reason = "first_build"
    entries[signature] = id(chain)
    cache_state["last_signature"] = signature
    cache_state["last_resource_signature"] = resource_signature
    cache_state["last_cache_key"] = cache_key
    cache_state["last_llm_cache_key"] = llm_cache_key
    cache_state["last_chain_id"] = id(chain)
    st.session_state["config_chat_chain_key"] = cache_key
    st.session_state["config_chat_chain_signature"] = signature
    if _LOGGER.isEnabledFor(logging.INFO):
        _LOGGER.info(
            "Config chat chain: reused=%s cache=%s lookup_s=%.3f miss=%s",
            "yes" if reused else "no",
            _chain_cache_summary(cache_key, llm_cache_key),
            lookup_seconds,
            cache_miss_reason or "none",
        )
    return chain, {
        "chain_reused": reused,
        "chain_cache_key": cache_key,
        "chain_cache_signature": signature,
        "chain_resource_signature": resource_signature,
        "chain_cache_summary": _chain_cache_summary(cache_key, llm_cache_key),
        "chain_cache_miss_reason": cache_miss_reason,
        "chain_cache_invalidation_fields": invalidation_fields,
        "chain_settings_changed": settings_changed,
        "chain_llm_changed_fields": llm_changed_fields,
        "chain_cache_session_reset": session_reset,
        "chain_lookup_seconds": lookup_seconds,
    }


def _get_nl_chain() -> tuple[ConfigPatchChain, dict[str, Any]]:
    return _build_nl_chain()


def _generate_config_preview(
    model_state: Mapping[str, Any],
    instruction: str,
) -> dict[str, Any]:
    chain_start = monotonic()
    chain, chain_meta = _get_nl_chain()
    chain_build_seconds = monotonic() - chain_start
    _record_chain_cache_stats(chain_meta, chain_build_seconds)
    run_start = monotonic()
    patch = chain.run(current_config=dict(model_state), instruction=instruction)
    run_seconds = monotonic() - run_start
    before = deepcopy(dict(model_state))
    after = apply_config_patch(before, patch)
    diff_text = diff_configs(before, after)
    return {
        "instruction": instruction,
        "before": before,
        "after": after,
        "diff": diff_text,
        "summary": patch.summary,
        "risk_flags": [flag.value for flag in patch.risk_flags],
        "needs_review": patch.needs_review,
        "patch": patch.model_dump(),
        "timings": {
            "chain_build_seconds": chain_build_seconds,
            "chain_reused": chain_meta.get("chain_reused"),
            "chain_cache_key": chain_meta.get("chain_cache_key"),
            "chain_cache_signature": chain_meta.get("chain_cache_signature"),
            "chain_resource_signature": chain_meta.get("chain_resource_signature"),
            "chain_cache_summary": chain_meta.get("chain_cache_summary"),
            "chain_cache_miss_reason": chain_meta.get("chain_cache_miss_reason"),
            "chain_cache_invalidation_fields": chain_meta.get("chain_cache_invalidation_fields"),
            "chain_settings_changed": chain_meta.get("chain_settings_changed"),
            "chain_llm_changed_fields": chain_meta.get("chain_llm_changed_fields"),
            "chain_cache_session_reset": chain_meta.get("chain_cache_session_reset"),
            "chain_lookup_seconds": chain_meta.get("chain_lookup_seconds"),
            "run_seconds": run_seconds,
        },
    }


def _estimate_llm_seconds() -> float:
    stored = st.session_state.get("config_chat_llm_seconds")
    return estimate_eta_seconds(stored)


def _record_llm_seconds(duration: float) -> None:
    stored = st.session_state.get("config_chat_llm_seconds")
    updated = update_eta_seconds(stored, duration)
    if updated is None:
        return
    st.session_state["config_chat_llm_seconds"] = updated


def _generate_preview_with_progress(
    model_state: Mapping[str, Any],
    instruction: str,
) -> dict[str, Any]:
    estimate = _estimate_llm_seconds()
    progress_slot = st.empty()
    progress_bar = progress_slot.progress(0.0, text="Preparing preview...")
    start = monotonic()
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(_generate_config_preview, model_state, instruction)
        while not future.done():
            elapsed = monotonic() - start
            ratio, remaining = progress_ratio_and_remaining(elapsed, estimate)
            progress_bar.progress(
                ratio,
                text=f"Generating preview... ~{int(round(remaining))}s remaining",
            )
            sleep(0.25)
    duration = monotonic() - start
    _record_llm_seconds(duration)
    progress_bar.progress(1.0, text="Preview ready.")
    progress_slot.empty()
    result = future.result()
    timings = result.get("timings")
    if isinstance(timings, Mapping):
        timings["total_seconds"] = duration
    _record_preview_timing(result, duration)
    st.session_state["config_chat_last_preview_summary"] = _preview_timing_summary(
        timings if isinstance(timings, Mapping) else None,
        total_seconds=duration,
    )
    return result


def _current_run_key(model_state: dict[str, Any], benchmark: str | None) -> str:
    fingerprint = st.session_state.get("data_fingerprint", "unknown")
    model_blob = json.dumps(model_state, sort_keys=True, default=str)
    bench = benchmark or "__none__"
    applied_funds = st.session_state.get("analysis_fund_columns")
    if not isinstance(applied_funds, list):
        applied_funds = st.session_state.get("fund_columns")
    if not isinstance(applied_funds, list):
        applied_funds = []

    selected_rf = st.session_state.get("selected_risk_free")
    selected_rf_key = selected_rf or "__none__"
    info_ratio_benchmark = (
        model_state.get("info_ratio_benchmark") if isinstance(model_state, dict) else None
    )
    regime_proxy = None
    if bool(model_state.get("regime_enabled", False)):
        regime_proxy = model_state.get("regime_proxy")
    prohibited = {selected_rf, benchmark, info_ratio_benchmark, regime_proxy} - {None}
    sanitized_funds = [c for c in applied_funds if c not in prohibited]

    funds_blob = json.dumps(list(sanitized_funds), sort_keys=False, default=str)
    funds_hash = hashlib.sha256(funds_blob.encode("utf-8")).hexdigest()[:12]
    return f"{fingerprint}:{bench}:{selected_rf_key}:{funds_hash}:{model_blob}"


def _apply_preview_state(
    preview: Mapping[str, Any],
    *,
    run_analysis: bool = False,
) -> None:
    after = preview.get("after")
    if not isinstance(after, Mapping):
        st.warning("Preview is missing updated configuration data.")
        return

    _record_config_change(preview)
    st.session_state["model_state"] = deepcopy(dict(after))
    analysis_runner.clear_cached_analysis()
    app_state.clear_analysis_results()

    if not run_analysis:
        st.success("Applied config changes to this session.")
        return

    df, _ = app_state.get_uploaded_data()
    if df is None:
        st.error("Load data before running analysis.")
        return

    benchmark = st.session_state.get("selected_benchmark")
    selected_rf = st.session_state.get("selected_risk_free")
    effective_model_state = dict(st.session_state.get("model_state", {}))
    if selected_rf:
        effective_model_state["risk_free_column"] = selected_rf

    with st.spinner("Running analysis..."):
        try:
            result = analysis_runner.run_analysis(
                df,
                effective_model_state,
                benchmark,
                data_hash=st.session_state.get("data_hash"),
            )
        except Exception as exc:
            st.error(f"Analysis failed: {exc}")
            st.session_state["analysis_result"] = None
            st.session_state["analysis_result_key"] = None
            st.session_state["analysis_error"] = {
                "message": "Analysis failed.",
                "detail": str(exc),
            }
            return

    st.session_state["analysis_result"] = result
    st.session_state["analysis_result_key"] = _current_run_key(
        st.session_state.get("model_state", {}),
        benchmark,
    )
    st.session_state.pop("analysis_error", None)
    st.success("Applied config changes and ran analysis.")


def _requires_risky_confirmation(preview: Mapping[str, Any]) -> bool:
    risk_flags = preview.get("risk_flags")
    needs_review = preview.get("needs_review")
    return bool(risk_flags) or bool(needs_review)


def _queue_risky_apply(preview: Mapping[str, Any], *, run_analysis: bool) -> None:
    st.session_state["config_chat_pending_apply"] = {
        "preview": preview,
        "run_analysis": run_analysis,
    }


def _render_risky_change_dialog() -> None:
    pending = st.session_state.get("config_chat_pending_apply")
    if not isinstance(pending, Mapping):
        return
    preview = pending.get("preview")
    if not isinstance(preview, Mapping):
        st.session_state.pop("config_chat_pending_apply", None)
        return
    risk_flags = preview.get("risk_flags") or []
    needs_review = bool(preview.get("needs_review"))
    if not risk_flags:
        if not needs_review:
            st.session_state.pop("config_chat_pending_apply", None)
            return

    dialog = getattr(st, "dialog", None)
    if dialog is None:
        st.error("RISKY CHANGE: confirmation dialog unavailable.")
        return

    with dialog("Confirm risky change"):
        st.warning("This change modifies sensitive configuration settings.")
        if risk_flags:
            st.caption(f"Flags: {', '.join(risk_flags)}")
        if needs_review:
            st.caption("Unknown config keys detected; please review before applying.")
        confirm = st.button("Apply anyway", type="primary")
        cancel = st.button("Cancel", type="secondary")
        if confirm:
            st.session_state.pop("config_chat_pending_apply", None)
            _apply_preview_state(preview, run_analysis=bool(pending.get("run_analysis")))
        elif cancel:
            st.session_state.pop("config_chat_pending_apply", None)


def _revert_last_config_change() -> None:
    history = _get_config_change_history()
    if not history:
        st.warning("No prior config change to revert.")
        return
    entry = history.pop()
    previous = entry.get("before") if isinstance(entry, Mapping) else None
    if not isinstance(previous, Mapping):
        st.error("Revert failed: missing previous configuration.")
        return
    st.session_state["model_state"] = deepcopy(dict(previous))
    st.session_state.pop("config_chat_preview", None)
    analysis_runner.clear_cached_analysis()
    app_state.clear_analysis_results()
    st.success("Reverted to the previous configuration.")


def _render_diff_preview_styles() -> None:
    st.markdown(
        """
<style>
.config-diff {
  border: 1px solid #e2e8f0;
  border-radius: 6px;
  background: #ffffff;
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, "Liberation Mono", monospace;
  font-size: 0.85rem;
  line-height: 1.4;
  overflow-x: auto;
}
.config-diff .diff-line {
  padding: 2px 8px;
  white-space: pre;
}
.config-diff .diff-add { background: #e6ffed; color: #14532d; }
.config-diff .diff-remove { background: #ffeef0; color: #7f1d1d; }
.config-diff .diff-header { background: #f8fafc; color: #0f172a; font-weight: 600; }
.config-diff .diff-hunk { background: #eff6ff; color: #1d4ed8; }
.config-diff .diff-context { color: #111827; }
.config-diff-table table.diff {
  width: 100%;
  border-collapse: collapse;
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, "Liberation Mono", monospace;
  font-size: 0.82rem;
  line-height: 1.35;
}
.config-diff-table table.diff th {
  background: #f8fafc;
  color: #0f172a;
  text-align: left;
  padding: 4px 6px;
  border: 1px solid #e2e8f0;
}
.config-diff-table table.diff td {
  padding: 2px 6px;
  vertical-align: top;
  white-space: pre;
  border: 1px solid #e2e8f0;
}
.config-diff-table .diff_add { background: #e6ffed; color: #14532d; }
.config-diff-table .diff_sub { background: #ffeef0; color: #7f1d1d; }
.config-diff-table .diff_chg { background: #fff7ed; color: #9a3412; }
</style>
""",
        unsafe_allow_html=True,
    )


def _diff_text_to_html(diff_text: str) -> str:
    lines = diff_text.splitlines()
    html_lines: list[str] = []
    for line in lines:
        if line.startswith(("+++ ", "--- ")):
            css_class = "diff-header"
        elif line.startswith("@@"):
            css_class = "diff-hunk"
        elif line.startswith("+"):
            css_class = "diff-add"
        elif line.startswith("-"):
            css_class = "diff-remove"
        else:
            css_class = "diff-context"
        safe_line = html.escape(line)
        html_lines.append(f'<div class="diff-line {css_class}">{safe_line}</div>')
    return '<div class="config-diff">' + "".join(html_lines) + "</div>"


def _render_unified_diff(diff_text: str) -> None:
    if not diff_text.strip():
        st.info("No differences found.")
        return
    _render_diff_preview_styles()
    st.markdown(_diff_text_to_html(diff_text), unsafe_allow_html=True)


def _render_side_by_side_diff(before: Mapping[str, Any], after: Mapping[str, Any]) -> None:
    before_yaml = yaml.safe_dump(dict(before), sort_keys=False, default_flow_style=False)
    after_yaml = yaml.safe_dump(dict(after), sort_keys=False, default_flow_style=False)
    differ = difflib.HtmlDiff(tabsize=2, wrapcolumn=80)
    diff_table = differ.make_table(
        before_yaml.splitlines(),
        after_yaml.splitlines(),
        fromdesc="Before",
        todesc="After",
        context=True,
        numlines=3,
    )
    _render_diff_preview_styles()
    st.markdown(
        f'<div class="config-diff config-diff-table">{diff_table}</div>',
        unsafe_allow_html=True,
    )
    with st.expander("Raw YAML", expanded=False):
        col_before, col_after = st.columns(2)
        with col_before:
            st.caption("Before")
            st.code(before_yaml, language="yaml")
        with col_after:
            st.caption("After")
            st.code(after_yaml, language="yaml")


def _render_config_diff_preview(model_state: Mapping[str, Any] | None) -> None:
    st.markdown("---")
    st.markdown("**Diff preview**")
    preview = st.session_state.get("config_chat_preview")
    if not isinstance(preview, Mapping):
        st.info("No preview available yet. Send an instruction to generate a diff.")
        return

    before = preview.get("before")
    if not isinstance(before, Mapping):
        before = model_state or {}
    after = preview.get("after")
    if not isinstance(after, Mapping):
        st.warning("Preview data is incomplete. Try generating a new diff.")
        return
    diff_text = preview.get("diff")
    if not isinstance(diff_text, str):
        diff_text = diff_configs(dict(before), dict(after))

    timings = preview.get("timings")
    if isinstance(timings, Mapping):
        chain_reused = "Yes" if timings.get("chain_reused") else "No"
        cache_signature = timings.get("chain_cache_signature")
        cache_signature_label = str(cache_signature)[:8] if cache_signature else "—"
        settings_changed = timings.get("chain_settings_changed")
        settings_changed_label = (
            "Yes" if settings_changed is True else "No" if settings_changed is False else "—"
        )
        cache_miss_reason = timings.get("chain_cache_miss_reason")
        cache_miss_label = f" | Cache miss: {cache_miss_reason}" if cache_miss_reason else ""
        invalidation_fields = timings.get("chain_cache_invalidation_fields")
        invalidation_label = (
            f" | Invalidated by: {', '.join(invalidation_fields)}"
            if isinstance(invalidation_fields, list) and invalidation_fields
            else ""
        )
        st.caption(
            "Preview timing — "
            f"Chain reused: {chain_reused} | "
            f"Cache sig: {cache_signature_label} | "
            f"Settings changed: {settings_changed_label} | "
            f"Chain build: {_format_seconds(timings.get('chain_build_seconds'))} | "
            f"Chain lookup: {_format_seconds(timings.get('chain_lookup_seconds'))} | "
            f"Run: {_format_seconds(timings.get('run_seconds'))} | "
            f"Total: {_format_seconds(timings.get('total_seconds'))}"
            f"{cache_miss_label}{invalidation_label}"
        )

    tabs = st.tabs(["Unified diff", "Side-by-side"])
    with tabs[0]:
        _render_unified_diff(diff_text)
    with tabs[1]:
        _render_side_by_side_diff(before, after)


def _render_config_change_history() -> None:
    history = _get_config_change_history()
    st.markdown("**Change history**")
    if not history:
        st.info("No configuration changes applied yet.")
        return

    for entry in reversed(history):
        if not isinstance(entry, Mapping):
            continue
        timestamp = entry.get("timestamp") or "Unknown time"
        instruction = entry.get("instruction") or "Config change"
        label = f"{timestamp} • {instruction}"
        with st.expander(label, expanded=False):
            summary = entry.get("summary")
            if summary:
                st.caption(summary)
            risk_flags = entry.get("risk_flags")
            if risk_flags:
                st.caption(f"Risk flags: {', '.join(risk_flags)}")
            before = entry.get("before")
            after = entry.get("after")
            if not isinstance(before, Mapping) or not isinstance(after, Mapping):
                st.warning("History entry is missing configuration data.")
                continue
            diff_text = entry.get("diff")
            if not isinstance(diff_text, str):
                diff_text = diff_configs(dict(before), dict(after))
            tabs = st.tabs(["Unified diff", "Side-by-side"])
            with tabs[0]:
                _render_unified_diff(diff_text)
            with tabs[1]:
                _render_side_by_side_diff(before, after)


def _render_config_chat_contents(model_state: Mapping[str, Any] | None) -> None:
    st.caption("Describe the configuration change you want to try.")
    _render_last_preview_metrics()
    last_summary = st.session_state.get("config_chat_last_preview_summary")
    if isinstance(last_summary, str) and last_summary:
        st.caption(last_summary)
    try:
        cache_context = _build_chain_cache_context()
    except Exception as exc:
        st.caption(f"Cache key unavailable: {exc}")
    else:
        cache_key = cache_context.get("cache_key")
        llm_cache_key = cache_context.get("llm_cache_key")
        cache_summary = (
            _chain_cache_summary(cache_key, llm_cache_key)
            if isinstance(cache_key, Mapping)
            else "—"
        )
        cache_signature = (
            _chain_cache_signature(cache_key)[:8] if isinstance(cache_key, Mapping) else "—"
        )
        st.caption(f"Current cache key: {cache_summary} | Sig: {cache_signature}")
    instruction = st.text_area(
        "Instruction",
        key="config_chat_instruction",
        height=120,
        placeholder="e.g. Increase lookback to 24 months and reduce max weight to 10%",
    )
    send_clicked = st.button("Send", key="config_chat_send", use_container_width=True)
    if send_clicked:
        trimmed = instruction.strip()
        if not trimmed:
            st.warning("Enter an instruction before sending.")
        else:
            st.session_state["config_chat_last_instruction"] = trimmed
            st.success("Instruction captured. Preview coming next.")
    preview = st.session_state.get("config_chat_preview")
    has_preview = isinstance(preview, Mapping) and isinstance(preview.get("after"), Mapping)
    action_cols = st.columns(4)
    with action_cols[0]:
        preview_clicked = st.button(
            "Preview",
            key="config_chat_preview_btn",
            use_container_width=True,
            disabled=not instruction.strip(),
        )
    with action_cols[1]:
        apply_clicked = st.button(
            "Apply",
            key="config_chat_apply_btn",
            use_container_width=True,
            disabled=not has_preview,
        )
    with action_cols[2]:
        apply_run_clicked = st.button(
            "Apply + Run",
            key="config_chat_apply_run_btn",
            use_container_width=True,
            type="primary",
            disabled=not has_preview,
        )
    with action_cols[3]:
        history = _get_config_change_history()
        revert_clicked = st.button(
            "Revert",
            key="config_chat_revert_btn",
            use_container_width=True,
            disabled=len(history) == 0,
        )

    if preview_clicked:
        trimmed = instruction.strip()
        if not trimmed:
            st.warning("Enter an instruction before previewing.")
        elif model_state is None:
            st.error("No configuration is loaded to preview against.")
        else:
            try:
                preview_payload = _generate_preview_with_progress(model_state, trimmed)
            except Exception as exc:
                st.error(f"Preview failed: {exc}")
            else:
                st.session_state["config_chat_preview"] = preview_payload
                st.session_state["config_chat_last_instruction"] = trimmed
                st.success("Preview ready. Review the diff below.")

    if apply_clicked and has_preview:
        if _requires_risky_confirmation(preview):
            _queue_risky_apply(preview, run_analysis=False)
        else:
            _apply_preview_state(preview, run_analysis=False)

    if apply_run_clicked and has_preview:
        if _requires_risky_confirmation(preview):
            _queue_risky_apply(preview, run_analysis=True)
        else:
            _apply_preview_state(preview, run_analysis=True)

    if revert_clicked:
        _revert_last_config_change()
    _render_risky_change_dialog()
    st.markdown("---")
    st.markdown("**Current configuration summary**")
    _render_config_summary(model_state)
    _render_config_diff_preview(model_state)
    st.markdown("---")
    _render_config_change_history()
    st.markdown("---")
    with st.expander("Preview timing log", expanded=False):
        _render_preview_timing_history()
    st.markdown("---")
    with st.expander("NL operation log", expanded=False):
        nl_operation_viewer.render_nl_operation_viewer()


def render_config_chat_panel(
    *,
    location: str = "sidebar",
    model_state: Mapping[str, Any] | None = None,
) -> None:
    """Render the Config Chat panel for natural-language config tweaks."""

    if location == "sidebar":
        sidebar_ctx = st.sidebar
        if not (hasattr(sidebar_ctx, "__enter__") and hasattr(sidebar_ctx, "__exit__")):
            sidebar_ctx = contextlib.nullcontext()
        with sidebar_ctx:
            _render_llm_status_panel()
            _render_llm_session_overrides_panel()
            with st.expander("💬 Config Chat", expanded=False):
                _render_config_chat_contents(model_state)
        return

    with st.expander("💬 Config Chat", expanded=False):
        _render_config_chat_contents(model_state)


# Preset configurations with default parameter values
PRESET_CONFIGS = {
    "Baseline": {
        "lookback_periods": 3,
        "min_history_periods": 3,
        "evaluation_periods": 1,
        "selection_count": 10,
        "weighting_scheme": "equal",
        "metric_weights": {
            "sharpe": 1.0,
            "return_ann": 1.0,
            "sortino": 0.0,
            "info_ratio": 0.0,
            "drawdown": 0.5,
            "vol": 0.0,
        },
        "risk_target": 0.10,
        # Date mode: "relative" (use lookback/eval windows) or "explicit" (user-specified dates)
        "date_mode": "relative",
        "start_date": None,
        "end_date": None,
        # Risk settings
        "rf_override_enabled": False,
        "rf_rate_annual": 0.0,
        "vol_floor": 0.015,
        "warmup_periods": 0,
        # Volatility adjustment details (Phase 10)
        "vol_adjust_enabled": True,
        "vol_window_length": 63,
        "vol_window_decay": "ewma",
        "vol_ewma_lambda": 0.94,
        # Advanced settings
        "max_weight": 0.20,
        "min_weight": 0.05,
        "cooldown_periods": 1,
        "rebalance_freq": "M",
        "max_turnover": 1.0,
        "transaction_cost_bps": 0,
        # Fund holding rules (Phase 3)
        "min_tenure_periods": 3,
        "max_changes_per_period": 0,  # 0 = unlimited
        "max_active_positions": 0,  # 0 = unlimited (uses selection_count)
        # Portfolio signal parameters (Phase 4)
        "trend_window": 63,
        "trend_lag": 1,
        "trend_min_periods": None,
        "trend_zscore": False,
        "trend_vol_adjust": False,
        "trend_vol_target": None,
        # Regime analysis (Phase 6)
        "regime_enabled": False,
        "regime_proxy": "SPX",
        # Robustness & Expert settings (Phase 7)
        "shrinkage_enabled": True,
        "shrinkage_method": "ledoit_wolf",
        "random_seed": 42,
        # Robustness fallbacks (Phase 14)
        "condition_threshold": 1.0e12,
        "safe_mode": "hrp",
        # Constraints (Phase 15)
        "long_only": True,
        # Entry/Exit thresholds (Phase 5)
        "z_entry_soft": 1.0,
        "z_exit_soft": -1.0,
        "soft_strikes": 2,
        "entry_soft_strikes": 1,
        "min_weight_strikes": 2,
        "sticky_add_periods": 1,
        "sticky_drop_periods": 1,
        "ci_level": 0.0,
        # Multi-period & Selection settings (Phase 8)
        "multi_period_enabled": True,
        "multi_period_frequency": "A",
        "inclusion_approach": "threshold",
        "slippage_bps": 0,
        "bottom_k": 0,
        # Selection approach details (Phase 9)
        "rank_pct": 0.10,
        # Multi-period bounds (Phase 12)
        "mp_min_funds": 10,
        "mp_max_funds": 25,
        # Hard thresholds (Phase 13)
        "z_entry_hard": None,
        "z_exit_hard": None,
    },
    "Conservative": {
        "lookback_periods": 5,
        "min_history_periods": 5,
        "evaluation_periods": 1,
        "selection_count": 8,
        "weighting_scheme": "risk_parity",
        "metric_weights": {
            "sharpe": 1.0,
            "return_ann": 0.5,
            "sortino": 1.0,
            "info_ratio": 0.0,
            "drawdown": 1.5,
            "vol": 1.0,
        },
        "risk_target": 0.08,
        # Date mode
        "date_mode": "relative",
        "start_date": None,
        "end_date": None,
        # Risk settings - lower floor for more conservative scaling
        "rf_rate_annual": 0.0,
        "vol_floor": 0.02,
        "warmup_periods": 6,
        # Advanced settings - more restrictive
        "max_weight": 0.15,
        "min_weight": 0.05,
        "cooldown_periods": 2,
        "rebalance_freq": "Q",
        "max_turnover": 0.50,
        "transaction_cost_bps": 10,
        # Fund holding rules - conservative: higher tenure, limited changes
        "min_tenure_periods": 6,
        "max_changes_per_period": 2,
        "max_active_positions": 10,
        # Portfolio signal parameters - longer window for stability
        "trend_window": 126,
        "trend_lag": 1,
        "trend_min_periods": None,
        "trend_zscore": True,
        "trend_vol_adjust": False,
        "trend_vol_target": None,
        # Regime analysis - enabled for defensive positioning
        "regime_enabled": True,
        "regime_proxy": "SPX",
        # Robustness & Expert settings - more conservative
        "shrinkage_enabled": True,
        "shrinkage_method": "ledoit_wolf",
        "random_seed": 42,
        # Robustness fallbacks (Phase 14) - conservative: stricter threshold
        "condition_threshold": 1.0e10,
        "safe_mode": "risk_parity",
        # Constraints (Phase 15)
        "long_only": True,
        # Entry/Exit thresholds - conservative: stricter entry, lenient exit
        "z_entry_soft": 1.5,
        "z_exit_soft": -1.0,
        "soft_strikes": 3,
        "entry_soft_strikes": 2,
        "min_weight_strikes": 2,
        "sticky_add_periods": 2,
        "sticky_drop_periods": 1,
        "ci_level": 0.0,
        # Multi-period & Selection settings (Phase 8) - conservative: longer periods
        "multi_period_enabled": True,
        "multi_period_frequency": "A",
        "inclusion_approach": "threshold",
        "slippage_bps": 5,
        "bottom_k": 0,
        # Selection approach details (Phase 9)
        "rank_pct": 0.10,
        # Multi-period bounds (Phase 12) - conservative: narrower range
        "mp_min_funds": 8,
        "mp_max_funds": 15,
        # Hard thresholds (Phase 13) - conservative: enabled, stricter
        "z_entry_hard": 2.5,
        "z_exit_hard": -2.5,
    },
    "Aggressive": {
        "lookback_periods": 2,
        "min_history_periods": 2,
        "evaluation_periods": 1,
        "selection_count": 15,
        "weighting_scheme": "hrp",
        "metric_weights": {
            "sharpe": 0.5,
            "return_ann": 2.0,
            "sortino": 0.5,
            "info_ratio": 0.0,
            "drawdown": 0.0,
            "vol": 0.0,
        },
        "risk_target": 0.15,
        # Date mode
        "date_mode": "relative",
        "start_date": None,
        "end_date": None,
        # Risk settings - lower floor, no warmup for faster response
        "rf_rate_annual": 0.0,
        "vol_floor": 0.01,
        "warmup_periods": 0,
        # Advanced settings - less restrictive
        "max_weight": 0.25,
        "min_weight": 0.05,
        "cooldown_periods": 0,
        "rebalance_freq": "M",
        "max_turnover": 1.0,
        "transaction_cost_bps": 0,
        # Fund holding rules - aggressive: minimal constraints
        "min_tenure_periods": 1,
        "max_changes_per_period": 0,  # unlimited
        "max_active_positions": 0,  # unlimited
        # Portfolio signal parameters - shorter window for responsiveness
        "trend_window": 42,
        "trend_lag": 1,
        "trend_min_periods": None,
        "trend_zscore": False,
        "trend_vol_adjust": True,
        "trend_vol_target": 0.10,
        # Regime analysis - disabled for pure momentum
        "regime_enabled": False,
        "regime_proxy": "SPX",
        # Robustness & Expert settings - more flexibility
        "shrinkage_enabled": True,
        "shrinkage_method": "ledoit_wolf",
        "random_seed": 42,
        # Robustness fallbacks (Phase 14) - aggressive: higher tolerance
        "condition_threshold": 1.0e14,
        "safe_mode": "hrp",
        # Constraints (Phase 15)
        "long_only": True,
        # Entry/Exit thresholds - aggressive: lenient entry, quick exit
        "z_entry_soft": 0.5,
        "z_exit_soft": -0.5,
        "soft_strikes": 1,
        "entry_soft_strikes": 1,
        "min_weight_strikes": 2,
        "sticky_add_periods": 1,
        "sticky_drop_periods": 1,
        "ci_level": 0.0,
        # Multi-period & Selection settings (Phase 8) - aggressive: shorter periods
        "multi_period_enabled": True,
        "multi_period_frequency": "Q",
        "inclusion_approach": "threshold",
        "slippage_bps": 0,
        "bottom_k": 0,
        # Selection approach details (Phase 9)
        "rank_pct": 0.15,  # more aggressive percentage
        # Multi-period bounds (Phase 12) - aggressive: wider range
        "mp_min_funds": 15,
        "mp_max_funds": 40,
        # Hard thresholds (Phase 13) - aggressive: disabled
        "z_entry_hard": None,
        "z_exit_hard": None,
    },
    "Custom": None,  # Custom means keep current values
}

# Common index/benchmark column names
BENCHMARK_COLUMNS = ["SPX", "TSX", "MSCI", "ACWI", "EAFE", "EM", "AGG", "BND"]

# Help text for configuration parameters (brief tooltips)
HELP_TEXT = {
    "preset": (
        "Pre-configured settings optimized for different investment styles. Changing preset "
        "auto-populates all parameters."
    ),
    "lookback": "Months of history used to calculate fund metrics (Sharpe, returns, etc.) for ranking.",
    "min_history": "Minimum months of data required for a fund to be considered for selection.",
    "evaluation": "Out-of-sample period (months) to measure portfolio performance after selection.",
    "selection": "Number of top-ranked funds to include in the portfolio.",
    "weighting": "How to allocate capital across selected funds. See Help page for details.",
    "sharpe_weight": "Importance of risk-adjusted returns in fund ranking.",
    "return_weight": "Importance of absolute returns in fund ranking.",
    "sortino_weight": "Importance of downside risk-adjusted returns in fund ranking.",
    "info_ratio_weight": "Importance of benchmark-relative risk-adjusted returns.",
    "drawdown_weight": "Importance of limiting drawdowns in fund ranking.",
    "vol_weight": "Importance of low volatility in fund ranking (lower vol = higher rank).",
    "risk_target": "Target portfolio volatility. Weights are scaled to achieve this level.",
    "info_ratio_benchmark": "Benchmark for calculating Information Ratio. Select an index or fund column.",
    # Date range settings
    "date_mode": "Choose whether to use relative lookback windows or explicit start/end dates.",
    "start_date": "Simulation start date. Data before this date will be excluded.",
    "end_date": "Simulation end date. Data after this date will be excluded.",
    # Risk settings
    "rf_override": (
        "Override the risk-free rate from data with a constant value. ⚠️ Using a constant "
        "rate reduces accuracy vs. time-varying rates."
    ),
    "rf_rate": (
        "Constant annual risk-free rate fallback. Only used when override is enabled and "
        "no RF column is in the data."
    ),
    "vol_floor": "Minimum volatility floor for scaling. Prevents extreme weights on low-vol assets.",
    "warmup_periods": (
        "Initial periods where returns are zeroed out to allow volatility estimates to "
        "stabilize before calculating performance metrics."
    ),
    # Phase 10: Volatility adjustment details
    "vol_adjust_enabled": "Enable volatility adjustment to scale returns to target vol.",
    "vol_window_length": "Rolling window for volatility estimation (periods). ~63 = 3 months.",
    "vol_window_decay": "EWMA weights recent data more; Simple uses equal weights.",
    "vol_ewma_lambda": "EWMA decay factor. Higher = longer memory. 0.94 is RiskMetrics standard.",
    # Advanced settings
    "max_weight": "Maximum allocation to any single fund. Prevents concentration risk.",
    "min_weight": "Minimum allocation per fund. Used as a weight floor and for underweight exit detection.",
    "cooldown_periods": "After a fund is removed, it cannot be re-added for this many periods.",
    "rebalance_freq": "How often to rebalance the portfolio weights.",
    "max_turnover": "Maximum portfolio turnover allowed per rebalance (1.0 = 100%).",
    "transaction_cost_bps": "Transaction cost in basis points (0.01% = 1 bp) applied per trade.",
    # Phase 3: Fund holding rules
    "min_tenure": "Minimum periods a fund must be held before it can be removed.",
    "max_changes": "Maximum number of fund additions/removals per rebalance. 0 = unlimited.",
    "max_active": "Maximum active positions in portfolio. 0 = use selection count.",
    # Phase 4: Trend signal parameters
    "trend_window": "Rolling window size for computing trend signals (in periods).",
    "trend_lag": "Number of periods to lag the signal (minimum 1 for causality).",
    "trend_min_periods": "Minimum observations required in rolling window. Blank = use window.",
    "trend_zscore": "Cross-sectionally standardize signals at each time step.",
    "trend_vol_adjust": "Scale signals by volatility to normalize across assets.",
    "trend_vol_target": "Target volatility for vol-adjusted signals.",
    # Phase 6: Regime analysis
    "regime_enabled": "Enable regime detection to adjust behavior in risk-on/risk-off environments.",
    "regime_proxy": "Market index used to detect risk-on/risk-off regimes.",
    # Phase 7: Robustness & Expert settings
    "shrinkage_enabled": "Apply covariance matrix shrinkage to improve stability.",
    "shrinkage_method": "Shrinkage method: Ledoit-Wolf or Oracle Approximating Shrinkage.",
    "random_seed": "Random seed for reproducibility. Change for different random selections.",
    # Phase 5: Entry/Exit thresholds
    "z_entry_soft": "Z-score threshold for fund entry consideration. Higher = stricter entry.",
    "z_exit_soft": "Z-score threshold for fund exit consideration. Lower = stricter exit.",
    "soft_strikes": "Consecutive periods below exit threshold before removing a fund.",
    "entry_soft_strikes": "Consecutive periods above entry threshold before adding a fund.",
    "min_weight_strikes": (
        "Underweight exit: consecutive periods a fund's natural weight stays below the "
        "minimum weight before it is replaced. 0 = disable."
    ),
    "sticky_add_periods": "Periods a fund must rank highly before being added to portfolio.",
    "sticky_drop_periods": "Periods a fund must rank poorly before being removed from portfolio.",
    "ci_level": "Confidence interval level for reporting only (0 = disabled, 0.9 = 90% CI).",
    # Phase 8: Multi-period & Selection settings
    "multi_period_enabled": "Enable rolling multi-period walk-forward analysis.",
    "multi_period_frequency": "Period frequency: Monthly (M), Quarterly (Q), or Annual (A).",
    "lookback_periods": "Number of periods for in-sample (training) window.",
    "evaluation_periods": "Number of periods for out-of-sample (testing) window.",
    "inclusion_approach": (
        "How to select funds: Top N, Top Percentage, Z-score Threshold, Random, or Buy & Hold."
    ),
    "buy_hold_initial": "Initial selection method for Buy & Hold mode.",
    "slippage_bps": "Additional slippage cost in basis points (market impact).",
    "bottom_k": "Number of bottom-ranked funds to always exclude (0 = none).",
    # Phase 9: Selection approach details
    "rank_pct": "Percentage of funds to include (0.10 = top 10%). Used with Top Percentage approach.",
    # Phase 12: Multi-period bounds
    "mp_min_funds": "Minimum number of funds to hold in multi-period analysis.",
    "mp_max_funds": "Maximum number of funds to hold in multi-period analysis.",
    # Phase 13: Hard entry/exit thresholds
    "z_entry_hard": "Hard entry: Z-score for immediate addition (bypasses strikes).",
    "z_exit_hard": "Hard exit: Z-score for immediate removal (bypasses strikes).",
    # Phase 14: Robustness fallbacks
    "condition_threshold": "Maximum acceptable condition number for covariance matrix.",
    "safe_mode": "Fallback weighting method when matrix is ill-conditioned.",
    # Phase 15: Constraints
    "long_only": (
        "Enforce long-only positions (no short selling). Built-in schemes (equal, "
        "score-prop, risk parity, HRP, ERC, robust_* defaults) are already non-negative "
        "unless you explicitly allow shorts (e.g., robust_mv with min_weight < 0). "
        "This matters when custom/manual weights or plugin engines allow shorts."
    ),
}


def _normalize_weights(weights: dict[str, float]) -> dict[str, float]:
    """Normalize metric weights to sum to 1.0."""
    total = sum(float(w or 0) for w in weights.values())
    if total <= 0:
        return weights
    return {k: round(v / total, 4) for k, v in weights.items()}


def _get_benchmark_columns(df) -> list[str]:
    """Identify potential benchmark columns in the dataset."""
    if df is None:
        return []
    all_cols = [str(c) for c in df.columns if str(c).upper() not in ("DATE", "INDEX")]
    # Prioritize known benchmark names, then include all columns as options
    benchmark_priority = []
    other_cols = []
    for col in all_cols:
        if col.upper() in [b.upper() for b in BENCHMARK_COLUMNS]:
            benchmark_priority.append(col)
        else:
            other_cols.append(col)
    return benchmark_priority + other_cols


def _validate_model(values: Mapping[str, Any], column_count: int) -> list[str]:
    errors: list[str] = []
    lookback = values.get("lookback_periods", 3)
    min_history = values.get("min_history_periods", lookback)
    if min_history > lookback:
        errors.append("Minimum history cannot exceed the lookback window.")
    selection = values.get("selection_count", 10)
    if column_count and selection > column_count:
        errors.append(
            f"Selection count ({selection}) cannot exceed available assets ({column_count})."
        )
    weights = values.get("metric_weights", {})
    if not any(float(w or 0) > 0 for w in weights.values()):
        errors.append("Provide at least one positive metric weight.")
    # Validate benchmark is set if info_ratio weight > 0
    if float(weights.get("info_ratio", 0)) > 0:
        if not values.get("info_ratio_benchmark"):
            errors.append("Select a benchmark for Information Ratio metric.")
    return errors


def _initial_model_state() -> dict[str, Any]:
    """Return default model state based on Baseline preset."""
    baseline = PRESET_CONFIGS["Baseline"]
    return {
        "preset": "Baseline",
        "lookback_periods": baseline["lookback_periods"],
        "min_history_periods": baseline["min_history_periods"],
        "evaluation_periods": baseline["evaluation_periods"],
        "selection_count": baseline["selection_count"],
        "weighting_scheme": baseline["weighting_scheme"],
        "metric_weights": baseline["metric_weights"].copy(),
        "risk_target": baseline["risk_target"],
        "info_ratio_benchmark": "",  # Empty until user selects
        # Date settings
        "date_mode": baseline["date_mode"],
        "start_date": baseline["start_date"],
        "end_date": baseline["end_date"],
        # Risk settings
        "rf_rate_annual": baseline["rf_rate_annual"],
        "vol_floor": baseline["vol_floor"],
        "warmup_periods": baseline["warmup_periods"],
        # Advanced settings
        "max_weight": baseline["max_weight"],
        "min_weight": baseline.get("min_weight", 0.05),
        "cooldown_periods": baseline["cooldown_periods"],
        "rebalance_freq": baseline["rebalance_freq"],
        "max_turnover": baseline["max_turnover"],
        "transaction_cost_bps": baseline["transaction_cost_bps"],
        # Fund holding rules (Phase 3)
        "min_tenure_periods": baseline["min_tenure_periods"],
        "max_changes_per_period": baseline["max_changes_per_period"],
        "max_active_positions": baseline["max_active_positions"],
        # Portfolio signal parameters (Phase 4)
        "trend_window": baseline["trend_window"],
        "trend_lag": baseline["trend_lag"],
        "trend_min_periods": baseline["trend_min_periods"],
        "trend_zscore": baseline["trend_zscore"],
        "trend_vol_adjust": baseline["trend_vol_adjust"],
        "trend_vol_target": baseline["trend_vol_target"],
        # Regime analysis (Phase 6)
        "regime_enabled": baseline["regime_enabled"],
        "regime_proxy": baseline["regime_proxy"],
        # Robustness & Expert settings (Phase 7)
        "shrinkage_enabled": baseline["shrinkage_enabled"],
        "shrinkage_method": baseline["shrinkage_method"],
        "random_seed": baseline["random_seed"],
        # Robustness fallbacks (Phase 14)
        "condition_threshold": baseline["condition_threshold"],
        "safe_mode": baseline["safe_mode"],
        # Constraints (Phase 15)
        "long_only": baseline["long_only"],
        # Entry/Exit thresholds (Phase 5)
        "z_entry_soft": baseline["z_entry_soft"],
        "z_exit_soft": baseline["z_exit_soft"],
        "soft_strikes": baseline["soft_strikes"],
        "entry_soft_strikes": baseline["entry_soft_strikes"],
        "min_weight_strikes": baseline.get("min_weight_strikes", 2),
        "sticky_add_periods": baseline["sticky_add_periods"],
        "sticky_drop_periods": baseline["sticky_drop_periods"],
        "ci_level": baseline["ci_level"],
        # Multi-period & Selection settings (Phase 8)
        "multi_period_enabled": baseline["multi_period_enabled"],
        "multi_period_frequency": baseline["multi_period_frequency"],
        "inclusion_approach": baseline["inclusion_approach"],
        "slippage_bps": baseline["slippage_bps"],
        "bottom_k": baseline["bottom_k"],
        # Selection approach details (Phase 9)
        "rank_pct": baseline["rank_pct"],
        # Multi-period bounds (Phase 12)
        "mp_min_funds": baseline["mp_min_funds"],
        "mp_max_funds": baseline["mp_max_funds"],
        # Hard thresholds (Phase 13)
        "z_entry_hard": baseline["z_entry_hard"],
        "z_exit_hard": baseline["z_exit_hard"],
    }


# Detailed descriptions for weighting schemes (shown in expander)
WEIGHTING_DESCRIPTIONS = {
    "equal": """
**Equal Weight (1/N)** allocates the same percentage to each selected fund.

- **Pros:** Simple, transparent, robust to estimation error
- **Cons:** Ignores risk characteristics; high-vol funds contribute more risk
- **Best for:** Most users; when you don't want to make assumptions about fund behavior
""",
    "risk_parity": """
**Risk Parity** allocates weights inversely proportional to each fund's volatility.
Higher-volatility funds receive lower weights, so each contributes roughly equal risk.

- **Pros:** Balances risk across assets; reduces concentration in volatile funds
- **Cons:** Ignores correlations; may over-allocate to low-vol assets
- **Best for:** Portfolios with assets of varying volatilities
""",
    "hrp": """
**Hierarchical Risk Parity (HRP)** uses machine learning clustering to build a
diversified allocation based on correlation structure.

- **Pros:** Accounts for correlations; more stable than mean-variance
- **Cons:** More complex; requires sufficient data for correlation estimation
- **Best for:** Complex portfolios with many correlated assets
""",
    "erc": """
**Equal Risk Contribution (ERC)** optimizes weights so each fund contributes
exactly the same marginal risk to the portfolio.

- **Pros:** Formal risk targeting; theoretically optimal risk allocation
- **Cons:** Requires optimization; sensitive to covariance estimation
- **Best for:** Formal risk management with specific risk targets
""",
    "robust_mv": """
**Robust Mean-Variance** uses shrinkage estimation to stabilize the covariance
matrix, reducing sensitivity to estimation error.

- **Pros:** More stable than classical MVO; resistant to extreme weights
- **Cons:** Still assumes you trust return forecasts
- **Best for:** When you have return forecasts but want protection from estimation error
""",
    "robust_risk_parity": """
**Robust Risk Parity** combines risk parity allocation with shrinkage estimation
for the covariance matrix.

- **Pros:** Benefits of risk parity with improved covariance estimation
- **Cons:** More complex; requires tuning shrinkage parameters
- **Best for:** Large portfolios with estimation uncertainty
""",
}


def render_model_page() -> None:
    app_state.initialize_session_state()
    model_state = st.session_state.setdefault("model_state", _initial_model_state())
    render_config_chat_panel(model_state=model_state)
    st.title("Model Configuration")

    # Clarify this is for custom analysis
    st.info(
        "💡 This page is for **custom analysis** with your own data. "
        "For quick demos with preset configurations, use the **Run Demo** button on the Home page."
    )

    _render_config_status_badge(model_state)

    # Help link - use st.page_link for proper navigation
    st.markdown(
        "📖 Use the **Help** page in the sidebar for detailed explanations of all parameters."
    )

    df, meta = app_state.get_uploaded_data()
    if df is None:
        st.error("Load data on the Data page before configuring the model.")
        return

    # Display dataset summary with name
    st.markdown("---")
    st.subheader("📊 Dataset Summary")

    # Get dataset name from session state or meta
    dataset_name = st.session_state.get("uploaded_filename", "Demo Dataset")
    if meta and hasattr(meta, "source_name"):
        dataset_name = meta.source_name
    st.markdown(f"**Dataset:** {dataset_name}")

    col_info1, col_info2, col_info3, col_info4 = st.columns(4)
    with col_info1:
        selected_rf = st.session_state.get("selected_risk_free")
        selected_bench = st.session_state.get("selected_benchmark")
        system_cols = {selected_rf, selected_bench, "Date", "DATE"} - {None}

        applied_funds = st.session_state.get("analysis_fund_columns")
        if not isinstance(applied_funds, list):
            applied_funds = st.session_state.get("fund_columns")

        if isinstance(applied_funds, list) and applied_funds:
            fund_cols = [c for c in applied_funds if c in df.columns and c not in system_cols]
        else:
            # Fallback: count only fund columns (exclude benchmarks/indices)
            fund_cols = [
                c
                for c in df.columns
                if c not in system_cols
                and c.upper()
                not in (
                    "SPX",
                    "TSX",
                    "RF",
                    "CASH",
                    "TBILL",
                    "TBILLS",
                    "T-BILL",
                )
            ]
        st.metric("Funds", len(fund_cols))
    with col_info2:
        st.metric("Time Periods", len(df))
    with col_info3:
        if hasattr(df.index, "min") and hasattr(df.index, "max"):
            start_date = df.index.min()
            st.metric(
                "Start Date",
                (
                    start_date.strftime("%Y-%m")
                    if hasattr(start_date, "strftime")
                    else str(start_date)[:7]
                ),
            )
    with col_info4:
        if hasattr(df.index, "max"):
            end_date = df.index.max()
            st.metric(
                "End Date",
                (
                    end_date.strftime("%Y-%m")
                    if hasattr(end_date, "strftime")
                    else str(end_date)[:7]
                ),
            )

    # Get data date boundaries for validation
    data_start = df.index.min() if hasattr(df.index, "min") else None
    data_end = df.index.max() if hasattr(df.index, "max") else None

    # Show fund names in an expander
    with st.expander("View fund names"):
        st.write(", ".join(fund_cols[:50]))
        if len(fund_cols) > 50:
            st.caption(f"...and {len(fund_cols) - 50} more")

    st.markdown("---")

    saved_model_states = app_state.get_saved_model_states()
    saved_names = sorted(saved_model_states)

    st.subheader("💾 Saved Configurations")
    with st.expander("Save, load, and manage model configurations", expanded=False):
        save_col, manage_col = st.columns(2)

        with save_col:
            st.markdown("**Save current settings**")
            with st.form("save_model_state_form"):
                default_name = st.session_state.get("active_saved_model_name", "")
                save_name = st.text_input(
                    "Configuration name",
                    value=default_name,
                    help="Provide a name to save the current model settings.",
                )
                overwrite_required = save_name.strip() in saved_model_states
                overwrite_confirmed = st.checkbox(
                    "Confirm overwrite",
                    value=False,
                    disabled=not overwrite_required,
                    help=(
                        "Required because a configuration with this name already exists."
                        if overwrite_required
                        else "Disabled until a duplicate name is entered."
                    ),
                )
                save_clicked = st.form_submit_button("Save Current Settings", type="primary")

            if save_clicked:
                trimmed = save_name.strip()
                if not trimmed:
                    st.error("Enter a name to save your configuration.")
                elif overwrite_required and not overwrite_confirmed:
                    st.warning("This name already exists. Check 'Confirm overwrite' to replace it.")
                else:
                    app_state.save_model_state(trimmed, st.session_state["model_state"])
                    app_state.save_config_wrapper(
                        trimmed, _build_config_wrapper(st.session_state["model_state"])
                    )
                    st.session_state["active_saved_model_name"] = trimmed
                    st.session_state["last_loaded_model_state"] = dict(
                        st.session_state["model_state"]
                    )
                    st.success(f"Saved configuration '{trimmed}'.")
                    st.rerun()

        with manage_col:
            st.markdown("**Load or manage saved configurations**")
            if not saved_names:
                st.info("No saved configurations yet. Save one to enable loading and export.")
            else:
                selected_index = 0
                active_saved_name = st.session_state.get("active_saved_model_name")
                if active_saved_name in saved_names:
                    selected_index = saved_names.index(active_saved_name)
                selected_saved = st.selectbox(
                    "Saved configurations",
                    saved_names,
                    index=selected_index,
                    key="saved_configuration_selector",
                )

                if st.button("Load selected configuration", key="load_saved_config_button"):
                    wrapper = app_state.load_saved_config_wrapper(selected_saved)
                    if isinstance(wrapper, Mapping):
                        _apply_config_wrapper(wrapper)
                    else:
                        loaded_state = app_state.load_saved_model_state(selected_saved)
                        st.session_state["model_state"] = loaded_state
                        st.session_state["last_loaded_model_state"] = dict(loaded_state)
                        _reset_model_widget_state()
                        _sync_model_widgets_from_state(loaded_state)
                    st.session_state["active_saved_model_name"] = selected_saved
                    analysis_runner.clear_cached_analysis()
                    app_state.clear_analysis_results()
                    st.success(
                        f"Loaded configuration '{selected_saved}'. The form has been updated."
                    )
                    st.rerun()

                with st.form("rename_saved_config_form"):
                    rename_target = st.text_input(
                        "Rename selected configuration",
                        value=selected_saved,
                        key="rename_saved_config_input",
                    )
                    rename_clicked = st.form_submit_button("Rename configuration")

                if rename_clicked:
                    try:
                        app_state.rename_saved_model_state(selected_saved, rename_target)
                    except (KeyError, ValueError) as exc:
                        st.error(str(exc))
                    else:
                        st.session_state["active_saved_model_name"] = rename_target.strip()
                        st.success(f"Renamed configuration to '{rename_target.strip()}'.")
                        st.rerun()

                if st.button(
                    "Delete selected configuration",
                    key="delete_saved_config_button",
                    type="secondary",
                ):
                    app_state.delete_saved_model_state(selected_saved)
                    if st.session_state.get("active_saved_model_name") == selected_saved:
                        st.session_state.pop("active_saved_model_name", None)
                    st.success(f"Deleted configuration '{selected_saved}'.")
                    st.rerun()

        st.markdown("---")

        # Quick download of current configuration (without saving)
        st.markdown("**Download Current Configuration**")
        current_wrapper = _build_config_wrapper(st.session_state["model_state"])
        current_payload = json.dumps(current_wrapper, indent=2, sort_keys=True, default=str)
        # Use active saved name if set, else uploaded filename, else "config"
        config_name = (
            st.session_state.get("active_saved_model_name")
            or st.session_state.get("uploaded_filename")
            or "config"
        )
        st.download_button(
            "📥 Download Configuration (JSON)",
            data=current_payload.encode("utf-8"),
            file_name=f"{config_name}_parameters.json",
            mime="application/json",
            help="Download all current parameters including model settings, fund selections, and benchmark choices.",
        )
        with st.expander("Preview current configuration", expanded=False):
            if hasattr(st, "json"):
                st.json(current_wrapper)
            else:
                st.write(current_wrapper)

        st.markdown("---")
        export_col, import_col = st.columns(2)
        with export_col:
            st.markdown("**Export saved configuration**")
            if saved_names:
                export_index = 0
                if st.session_state.get("active_saved_model_name") in saved_names:
                    export_index = saved_names.index(st.session_state["active_saved_model_name"])
                export_target = st.selectbox(
                    "Choose configuration to export",
                    saved_names,
                    index=export_index,
                    key="export_config_selector",
                )
                export_payload = app_state.export_model_state(export_target)
                st.download_button(
                    "📥 Download Saved Configuration",
                    data=export_payload.encode("utf-8"),
                    file_name=f"{export_target}_parameters.json",
                    mime="application/json",
                    key="download_saved_config_button",
                )
                with st.expander("Preview saved configuration", expanded=False):
                    st.text_area(
                        "Exported JSON",
                        value=export_payload,
                        height=160,
                        key="exported_config_payload",
                        help="Copy this JSON to share or reuse the configuration.",
                    )
            else:
                st.info("Save a configuration to enable export.")

        with import_col:
            st.markdown("**Import configuration from JSON**")
            import_name = st.text_input("Name for imported configuration", key="import_config_name")
            uploaded_config = st.file_uploader(
                "Upload JSON file",
                type=["json"],
                key="import_config_file",
                help="Optional: upload a JSON file instead of pasting.",
            )
            if uploaded_config is not None:
                try:
                    raw_value = uploaded_config.getvalue()
                    if isinstance(raw_value, bytes):
                        st.session_state["import_config_payload"] = raw_value.decode("utf-8-sig")
                    else:
                        st.session_state["import_config_payload"] = str(raw_value)
                except UnicodeDecodeError:
                    st.error(
                        "Unable to decode uploaded file as UTF-8. "
                        "Please upload a UTF-8 encoded JSON file."
                    )
                except (AttributeError, TypeError):
                    st.error("Unable to read uploaded JSON file due to an unexpected file format.")
                except OSError as exc:
                    st.error(f"Unable to read uploaded file: {exc}")
            import_payload = st.text_area("Paste JSON to import", key="import_config_payload")
            if st.button("Import JSON configuration", key="import_config_button"):
                if not import_payload.strip():
                    st.error("Paste a JSON payload to import a configuration.")
                else:
                    try:
                        imported_state = app_state.import_model_state(import_name, import_payload)
                    except ValueError as exc:
                        st.error(str(exc))
                    else:
                        st.session_state["active_saved_model_name"] = import_name.strip()
                        wrapper = app_state.load_saved_config_wrapper(import_name.strip())
                        if isinstance(wrapper, Mapping):
                            _apply_config_wrapper(wrapper)
                        else:
                            st.session_state["model_state"] = imported_state
                            st.session_state["last_loaded_model_state"] = dict(imported_state)
                            _reset_model_widget_state()
                            _sync_model_widgets_from_state(imported_state)
                        analysis_runner.clear_cached_analysis()
                        app_state.clear_analysis_results()
                        st.success(
                            f"Imported configuration '{import_name.strip()}'. The form has been updated."
                        )
                        st.rerun()

        st.markdown("---")
        st.markdown("**Compare saved configurations**")
        if len(saved_names) < 2:
            st.info("Save at least two configurations to compare differences.")
        else:
            compare_col_a, compare_col_b = st.columns(2)
            with compare_col_a:
                config_a_name = st.selectbox(
                    "Configuration A",
                    saved_names,
                    index=0,
                    key="compare_config_a",
                )
            with compare_col_b:
                default_b = 1 if len(saved_names) > 1 else 0
                config_b_name = st.selectbox(
                    "Configuration B",
                    saved_names,
                    index=default_b,
                    key="compare_config_b",
                )

            if config_a_name == config_b_name:
                st.warning("Select two different configurations to compare.")
            else:
                diffs = app_state.diff_model_states(
                    saved_model_states[config_a_name],
                    saved_model_states[config_b_name],
                )
                if not diffs:
                    st.success("No differences found. The selected configurations match.")
                else:
                    diff_rows = []
                    for entry in diffs:
                        diff_rows.append(
                            {
                                "Setting": entry.path,
                                "Config A": (
                                    json.dumps(entry.left, sort_keys=True, default=str)
                                    if entry.left is not None
                                    else "—"
                                ),
                                "Config B": (
                                    json.dumps(entry.right, sort_keys=True, default=str)
                                    if entry.right is not None
                                    else "—"
                                ),
                                "Change": entry.change_type
                                + (" (type changed)" if entry.type_changed else ""),
                            }
                        )

                    st.dataframe(diff_rows, use_container_width=True, hide_index=True)

                    diff_text = app_state.format_model_state_diff(
                        diffs, label_a=config_a_name, label_b=config_b_name
                    )
                    st.caption("Copyable diff output:")
                    st.code(diff_text, language="text")

    # =============================================
    # SIMULATION PERIOD SETTINGS (outside form for immediate feedback)
    # =============================================
    st.subheader("📅 Simulation Period")
    st.caption("Define the time range for your simulation.")

    date_mode_options = ["relative", "explicit"]
    date_mode_labels = {
        "relative": "Relative (use lookback windows)",
        "explicit": "Explicit (specify start/end dates)",
    }
    current_date_mode = model_state.get("date_mode", "relative")

    date_mode = st.radio(
        "Date Mode",
        options=date_mode_options,
        format_func=lambda x: date_mode_labels.get(x, x),
        index=(
            date_mode_options.index(current_date_mode)
            if current_date_mode in date_mode_options
            else 0
        ),
        help=HELP_TEXT["date_mode"],
        horizontal=True,
        key="date_mode_radio",
    )

    # Update model state if date mode changed
    if date_mode != current_date_mode:
        st.session_state["model_state"]["date_mode"] = date_mode

    # Show date pickers when in explicit mode
    if date_mode == "explicit":
        # Convert data boundaries to date objects for the date picker
        if data_start is not None and hasattr(data_start, "date"):
            min_date = data_start.date()
        else:
            min_date = None

        if data_end is not None and hasattr(data_end, "date"):
            max_date = data_end.date()
        else:
            max_date = None

        # Show valid data range prominently
        if min_date and max_date:
            st.info(
                f"📅 **Available data range:** {min_date.strftime('%b %d, %Y')} to {max_date.strftime('%b %d, %Y')}"
            )

        date_col1, date_col2 = st.columns(2)
        # Get current values from model state
        current_start = model_state.get("start_date")
        current_end = model_state.get("end_date")

        # Track if dates were auto-corrected
        start_was_corrected = False
        end_was_corrected = False
        original_start_str = None
        original_end_str = None

        # Convert to date objects if they're strings
        if isinstance(current_start, str) and current_start:
            try:
                import datetime

                original_start_str = current_start
                current_start = datetime.datetime.strptime(
                    current_start[:7] + "-01", "%Y-%m-%d"
                ).date()
            except (ValueError, TypeError):
                current_start = min_date
        elif current_start is None:
            current_start = min_date

        # Ensure current_start is within valid range
        if current_start is not None and min_date is not None:
            if current_start < min_date:
                start_was_corrected = True
                current_start = min_date
        if current_start is not None and max_date is not None:
            if current_start > max_date:
                start_was_corrected = True
                current_start = max_date

        if isinstance(current_end, str) and current_end:
            try:
                import datetime

                original_end_str = current_end
                current_end = datetime.datetime.strptime(current_end[:7] + "-01", "%Y-%m-%d").date()
            except (ValueError, TypeError):
                current_end = max_date
        elif current_end is None:
            current_end = max_date

        # Ensure current_end is within valid range
        if current_end is not None and min_date is not None:
            if current_end < min_date:
                end_was_corrected = True
                current_end = min_date
        if current_end is not None and max_date is not None:
            if current_end > max_date:
                end_was_corrected = True
                current_end = max_date

        with date_col1:
            sim_start_date = st.date_input(
                "Simulation Start Date",
                value=current_start,
                min_value=min_date,
                max_value=max_date,
                help=HELP_TEXT["start_date"],
                key="sim_start_date",
            )
            # Show warning if date was auto-corrected
            if start_was_corrected and original_start_str:
                st.caption(f"⚠️ Adjusted from {original_start_str[:10]} to nearest available date")
            # Update model state
            if sim_start_date:
                st.session_state["model_state"]["start_date"] = sim_start_date.strftime("%Y-%m-%d")

        with date_col2:
            sim_end_date = st.date_input(
                "Simulation End Date",
                value=current_end,
                min_value=min_date,
                max_value=max_date,
                help=HELP_TEXT["end_date"],
                key="sim_end_date",
            )
            # Show warning if date was auto-corrected
            if end_was_corrected and original_end_str:
                st.caption(f"⚠️ Adjusted from {original_end_str[:10]} to nearest available date")
            # Update model state
            if sim_end_date:
                st.session_state["model_state"]["end_date"] = sim_end_date.strftime("%Y-%m-%d")

        # Validate date range
        if sim_start_date and sim_end_date and sim_start_date > sim_end_date:
            st.error("Start date must be before end date.")
        else:
            # Show selected period info
            if sim_start_date and sim_end_date:
                months_span = (sim_end_date.year - sim_start_date.year) * 12 + (
                    sim_end_date.month - sim_start_date.month
                )
                if months_span > 600:  # 50 years * 12 months
                    st.warning("Date range exceeds 50 years - please verify your selection.")
                else:
                    st.info(
                        (
                            f"📊 Selected period: {sim_start_date.strftime('%Y-%m')} to "
                            f"{sim_end_date.strftime('%Y-%m')} ({months_span} months)"
                        )
                    )
    else:
        st.info(
            "📊 Using relative date mode: simulation dates will be computed from lookback and evaluation windows."
        )

    st.markdown("---")

    # Get benchmark column options for Info Ratio
    benchmark_options = _get_benchmark_columns(df)

    # Preset selection with auto-population (outside form for instant feedback)
    preset_options = ["Baseline", "Conservative", "Aggressive", "Custom"]
    current_preset = model_state.get("preset", "Baseline")
    try:
        preset_index = preset_options.index(current_preset)
    except ValueError:
        preset_index = 0

    # Preset selector (outside form for immediate updates)
    new_preset = st.selectbox(
        "📋 Preset Configuration",
        preset_options,
        index=preset_index,
        help=HELP_TEXT["preset"],
        key="preset_selector",
    )

    # Auto-populate when preset changes (except Custom)
    if new_preset != current_preset and new_preset != "Custom":
        preset_config = PRESET_CONFIGS.get(new_preset)
        if preset_config:
            st.session_state["model_state"] = {
                "preset": new_preset,
                "lookback_periods": preset_config["lookback_periods"],
                "min_history_periods": preset_config["min_history_periods"],
                "evaluation_periods": preset_config["evaluation_periods"],
                "selection_count": preset_config["selection_count"],
                "weighting_scheme": preset_config["weighting_scheme"],
                "metric_weights": preset_config["metric_weights"].copy(),
                "risk_target": preset_config["risk_target"],
                "info_ratio_benchmark": model_state.get("info_ratio_benchmark", ""),
                # Date settings
                "date_mode": preset_config["date_mode"],
                "start_date": preset_config["start_date"],
                "end_date": preset_config["end_date"],
                # Risk settings
                "rf_rate_annual": preset_config["rf_rate_annual"],
                "vol_floor": preset_config["vol_floor"],
                "warmup_periods": preset_config["warmup_periods"],
                # Advanced settings
                "max_weight": preset_config["max_weight"],
                "min_weight": preset_config.get("min_weight", 0.05),
                "cooldown_periods": preset_config["cooldown_periods"],
                "rebalance_freq": preset_config["rebalance_freq"],
                "max_turnover": preset_config["max_turnover"],
                "transaction_cost_bps": preset_config["transaction_cost_bps"],
                # Fund holding rules (Phase 3)
                "min_tenure_periods": preset_config["min_tenure_periods"],
                "max_changes_per_period": preset_config["max_changes_per_period"],
                "max_active_positions": preset_config["max_active_positions"],
                # Portfolio signal parameters (Phase 4)
                "trend_window": preset_config["trend_window"],
                "trend_lag": preset_config["trend_lag"],
                "trend_min_periods": preset_config["trend_min_periods"],
                "trend_zscore": preset_config["trend_zscore"],
                "trend_vol_adjust": preset_config["trend_vol_adjust"],
                "trend_vol_target": preset_config["trend_vol_target"],
                # Regime analysis (Phase 6)
                "regime_enabled": preset_config["regime_enabled"],
                "regime_proxy": preset_config["regime_proxy"],
                # Robustness & Expert settings (Phase 7)
                "shrinkage_enabled": preset_config["shrinkage_enabled"],
                "shrinkage_method": preset_config["shrinkage_method"],
                "random_seed": preset_config["random_seed"],
                # Entry/Exit thresholds (Phase 5)
                "z_entry_soft": preset_config["z_entry_soft"],
                "z_exit_soft": preset_config["z_exit_soft"],
                "soft_strikes": preset_config["soft_strikes"],
                "entry_soft_strikes": preset_config["entry_soft_strikes"],
                "min_weight_strikes": preset_config.get("min_weight_strikes", 2),
                "sticky_add_periods": preset_config["sticky_add_periods"],
                "sticky_drop_periods": preset_config["sticky_drop_periods"],
                "ci_level": preset_config["ci_level"],
            }
            st.rerun()

    # Weighting scheme selector (outside form for dynamic description updates)
    st.markdown("---")
    st.subheader("📊 Weighting Scheme")
    weighting_labels = [label for label, _ in WEIGHTING_SCHEMES]
    weighting_values = [value for _, value in WEIGHTING_SCHEMES]
    current_weighting = model_state.get("weighting_scheme", "equal")
    try:
        weighting_index = weighting_values.index(current_weighting)
    except ValueError:
        weighting_index = 0

    weighting_value = st.selectbox(
        "Select Weighting Scheme",
        options=weighting_values,
        format_func=lambda x: weighting_labels[weighting_values.index(x)],
        index=weighting_index,
        help=HELP_TEXT["weighting"],
        key="weighting_scheme_selector",
    )

    # Show description for selected weighting scheme (updates dynamically)
    with st.expander("ℹ️ About this weighting scheme", expanded=False):
        st.markdown(WEIGHTING_DESCRIPTIONS.get(weighting_value, "No description available."))

    # Update model_state if weighting changed
    if weighting_value != current_weighting:
        st.session_state["model_state"]["weighting_scheme"] = weighting_value

    with st.form("model_settings", clear_on_submit=False):
        # =====================================================================
        # Section 0: Analysis Mode (Primary Choice)
        # =====================================================================
        st.subheader("🎯 Analysis Mode")
        st.caption(
            "Choose between single-period (one-time selection) or multi-period "
            "(rolling walk-forward analysis with rebalancing)."
        )

        multi_period_enabled = st.checkbox(
            "Enable Multi-Period Walk-Forward Analysis",
            value=bool(model_state.get("multi_period_enabled", True)),
            help=HELP_TEXT["multi_period_enabled"],
        )
        if multi_period_enabled:
            st.success(
                "✅ Funds will be re-evaluated at each period. Entry/exit rules and rebalancing apply."
            )

        # Fund Selection Approach - determines how funds are chosen
        st.markdown("**Fund Selection Approach**")
        approach_c1, approach_c2 = st.columns(2)
        with approach_c1:
            inclusion_approaches = [
                "top_n",
                "top_pct",
                "threshold",
                "random",
                "buy_and_hold",
            ]
            inclusion_labels = {
                "top_n": "Top N Funds (Ranking)",
                "top_pct": "Top Percentage (Ranking)",
                "threshold": "Z-Score Threshold",
                "random": "Random Selection",
                "buy_and_hold": "Buy & Hold",
            }
            current_inclusion = model_state.get("inclusion_approach", "threshold")
            inclusion_approach = st.selectbox(
                "Selection Method",
                options=inclusion_approaches,
                format_func=lambda x: inclusion_labels.get(x, x),
                index=(
                    inclusion_approaches.index(current_inclusion)
                    if current_inclusion in inclusion_approaches
                    else 2  # Default to "threshold" (index 2)
                ),
                help=HELP_TEXT["inclusion_approach"],
                key="inclusion_approach_select",
            )

        # Indicate whether this is ranking-based, threshold-based, random, or buy_and_hold
        is_ranking_mode = inclusion_approach in ["top_n", "top_pct"]
        is_random_mode = inclusion_approach == "random"
        is_top_n_mode = inclusion_approach == "top_n"
        is_top_pct_mode = inclusion_approach == "top_pct"
        is_buy_and_hold_mode = inclusion_approach == "buy_and_hold"

        # Buy & Hold initial selection method
        buy_hold_initial = "top_n"  # Default
        with approach_c2:
            if is_buy_and_hold_mode:
                # Show initial selection method for buy & hold
                buy_hold_options = ["top_n", "top_pct", "threshold", "random"]
                buy_hold_labels = {
                    "top_n": "Top N (Ranking)",
                    "top_pct": "Top Percentage",
                    "threshold": "Z-Score Threshold",
                    "random": "Random",
                }
                current_buy_hold = model_state.get("buy_hold_initial", "top_n")
                buy_hold_initial = st.selectbox(
                    "Initial Selection Method",
                    options=buy_hold_options,
                    format_func=lambda x: buy_hold_labels.get(x, x),
                    index=(
                        buy_hold_options.index(current_buy_hold)
                        if current_buy_hold in buy_hold_options
                        else 0
                    ),
                    help=HELP_TEXT.get(
                        "buy_hold_initial",
                        "How to select funds initially. Funds are held until they cease to exist.",
                    ),
                    key="buy_hold_initial_select",
                )
                st.caption(
                    f"🔒 **Buy & Hold**: Select funds using {buy_hold_labels[buy_hold_initial]}, "
                    "then hold until fund data disappears. Replacements use same method."
                )
            elif is_top_pct_mode:
                # Show percentage selector directly for top_pct mode
                rank_pct_input = st.number_input(
                    "Top Percentage (%)",
                    min_value=1,
                    max_value=50,
                    value=int(float(model_state.get("rank_pct", 0.10)) * 100),
                    step=1,
                    help="Select top N% of funds by score (e.g., 10 = top 10%)",
                    key="rank_pct_primary",
                )
                st.caption(f"🏆 Select top {rank_pct_input}% of funds by score")
            elif is_random_mode:
                st.caption(
                    "🎲 **Random Mode**: Funds are randomly selected each period. "
                    "No in-sample ranking metrics used for selection."
                )
            elif is_ranking_mode:
                st.caption(
                    "🏆 **Ranking Mode**: Funds are ranked by score and the top performers "
                    "are selected. Entry/exit uses ranking stability."
                )
            else:
                st.caption(
                    "📊 **Threshold Mode**: Funds must exceed a z-score threshold to enter. "
                    "Entry/exit uses z-score thresholds."
                )

        # =====================================================================
        # Section 1: Fund Selection Settings
        # =====================================================================
        st.divider()
        st.subheader("📋 Fund Selection & Time Windows")
        st.caption("Configure time windows for fund evaluation and walk-forward analysis.")

        # Row 1: Frequency (sets the period unit for all time windows)
        # Note: This is inside the form, so labels won't update until form is submitted.
        # We use model_state to determine the current unit for display.
        multi_period_frequencies = ["M", "Q", "A"]
        freq_labels = {
            "M": "Monthly",
            "Q": "Quarterly",
            "A": "Annual",
        }
        freq_period_labels = {
            "M": "months",
            "Q": "quarters",
            "A": "years",
        }
        current_mp_freq = model_state.get("multi_period_frequency", "A")
        multi_period_frequency = st.selectbox(
            "Period Frequency",
            options=multi_period_frequencies,
            format_func=lambda x: freq_labels.get(x, x),
            index=(
                multi_period_frequencies.index(current_mp_freq)
                if current_mp_freq in multi_period_frequencies
                else 2
            ),
            help=HELP_TEXT["multi_period_frequency"],
        )
        # Use the saved frequency from model_state for label display (shows current saved value)
        # The new selection will take effect after save

        # Row 2: Time windows
        st.markdown("**Time Windows**")
        c1, c2, c3 = st.columns(3)
        with c1:
            lookback = st.number_input(
                "Lookback",
                min_value=1,
                max_value=20,
                value=int(model_state.get("lookback_periods", 3)),
                help=HELP_TEXT.get(
                    "lookback_periods",
                    "Number of periods for in-sample (training) window.",
                ),
            )
            st.caption("In-sample history for ranking")
        with c2:
            evaluation = st.number_input(
                "Evaluation",
                min_value=1,
                max_value=10,
                value=int(model_state.get("evaluation_periods", 1)),
                help=HELP_TEXT.get(
                    "evaluation_periods",
                    "Number of periods for out-of-sample (testing) window.",
                ),
            )
            st.caption("Out-of-sample test period")
        with c3:
            min_history = st.number_input(
                "Min History",
                min_value=1,
                max_value=20,
                value=int(
                    model_state.get("min_history_periods", model_state.get("lookback_periods", 3))
                ),
                help=HELP_TEXT.get(
                    "min_history",
                    "Minimum periods of data required for a fund to be considered.",
                ),
            )
            st.caption("Funds with less history excluded")

        # Show period summary based on selected frequency
        selected_unit = freq_period_labels.get(multi_period_frequency, "periods")
        st.caption(
            f"Strategy: {lookback} {selected_unit} training → {evaluation} {selected_unit} testing, "
            f"rebalanced {freq_labels[multi_period_frequency].lower()}."
        )

        # Section 2: Portfolio Settings
        st.divider()
        st.subheader("📈 Portfolio Settings")
        st.caption("Configure how the portfolio is constructed from selected funds.")

        # Used by multiple controls: multi-period toggles are stored in model_state
        mp_enabled_state = bool(model_state.get("multi_period_enabled", True))

        st.markdown("**Portfolio size (target / min / max)**")
        size_c1, size_c2, size_c3 = st.columns(3)
        with size_c1:
            selection = st.number_input(
                "Target Funds (Initial N)",
                min_value=1,
                max_value=len(fund_cols) if fund_cols else 100,
                value=min(
                    int(model_state.get("selection_count", 10)),
                    len(fund_cols) if fund_cols else 10,
                ),
                help=(
                    "Target number of funds to hold (initial selection size). "
                    "Other constraints (min/max) may expand or cap holdings."
                ),
            )
            st.caption("Target holdings (initial selection size)")

        # Disable min/max for top_n mode since fund count is fixed by selection_count
        min_max_disabled = not mp_enabled_state or is_top_n_mode
        min_max_help_suffix = (
            " (disabled for Top N mode - uses Target Funds)" if is_top_n_mode else ""
        )

        with size_c2:
            mp_min_funds = st.number_input(
                "Minimum Funds",
                min_value=0,
                max_value=len(fund_cols) if fund_cols else 100,
                value=int(model_state.get("mp_min_funds", 0)),
                help=HELP_TEXT["mp_min_funds"] + min_max_help_suffix,
                disabled=min_max_disabled,
                key="mp_min_funds_input",
            )
            if is_top_n_mode:
                st.caption("🔒 Disabled (Top N uses Target Funds)")
            else:
                st.caption("Floor (0 = disabled)")

        with size_c3:
            mp_max_funds = st.number_input(
                "Maximum Funds",
                min_value=0,
                max_value=len(fund_cols) if fund_cols else 100,
                value=int(model_state.get("mp_max_funds", 0)),
                help=HELP_TEXT["mp_max_funds"] + min_max_help_suffix,
                disabled=min_max_disabled,
                key="mp_max_funds_input",
            )
            if is_top_n_mode:
                st.caption("🔒 Disabled (Top N uses Target Funds)")
            else:
                st.caption("Cap (0 = disabled)")

        st.markdown("**Portfolio weight constraints**")
        w_c1, w_c2 = st.columns(2)
        with w_c1:
            min_weight_pct = st.number_input(
                "Portfolio Weight Minimum per Fund (%)",
                min_value=0.0,
                max_value=100.0,
                value=float(model_state.get("min_weight", 0.05)) * 100,
                step=0.5,
                format="%.2f",
                help=HELP_TEXT["min_weight"],
            )
            min_weight_decimal = min_weight_pct / 100.0

        with w_c2:
            max_weight = st.number_input(
                "Maximum Weight per Fund (%)",
                min_value=0.0,
                max_value=100.0,
                value=float(model_state.get("max_weight", 0.20)) * 100,
                step=1.0,
                format="%.2f",
                help=HELP_TEXT["max_weight"],
            )
            max_weight_decimal = max_weight / 100.0

        # Section 3: Metric Weights
        st.divider()
        st.subheader("⚖️ Metric Weights")
        if is_random_mode:
            st.info(
                "🎲 **Random Mode**: Metric weights are not used for selection in random mode. "
                "Funds are selected randomly. Metrics are still calculated for reporting purposes."
            )
        st.caption("Relative importance of each metric when ranking funds for selection.")

        metric_weights: dict[str, float] = {}
        # Create two rows for the 6 metrics
        help_keys = [
            "sharpe_weight",
            "return_weight",
            "sortino_weight",
            "info_ratio_weight",
            "drawdown_weight",
            "vol_weight",
        ]

        # First row: 3 metrics
        col1, col2, col3 = st.columns(3)
        cols_row1 = [col1, col2, col3]
        for i in range(min(3, len(METRIC_FIELDS))):
            label, code = METRIC_FIELDS[i]
            help_key = help_keys[i]
            with cols_row1[i]:
                metric_weights[code] = st.number_input(
                    label,
                    min_value=0.0,
                    value=float(model_state.get("metric_weights", {}).get(code, 1.0)),
                    step=0.1,
                    help=HELP_TEXT.get(help_key, "Weight for this metric in fund ranking."),
                    key=f"metric_{code}",
                )

        # Second row: remaining metrics
        if len(METRIC_FIELDS) > 3:
            col4, col5, col6 = st.columns(3)
            cols_row2 = [col4, col5, col6]
            for i in range(3, len(METRIC_FIELDS)):
                label, code = METRIC_FIELDS[i]
                help_key = help_keys[i] if i < len(help_keys) else "vol_weight"
                with cols_row2[i - 3]:
                    metric_weights[code] = st.number_input(
                        label,
                        min_value=0.0,
                        value=float(model_state.get("metric_weights", {}).get(code, 1.0)),
                        step=0.1,
                        help=HELP_TEXT.get(help_key, "Weight for this metric in fund ranking."),
                        key=f"metric_{code}",
                    )

        # Show weight sum
        weight_sum = sum(float(w or 0) for w in metric_weights.values())
        if weight_sum > 0:
            st.caption(
                f"📊 Total weight: {weight_sum:.2f} — Weights will be auto-normalized to sum to 1.0 during analysis."
            )
        else:
            st.warning("⚠️ Set at least one metric weight > 0.")

        # Benchmark selector for Info Ratio - always show when info_ratio weight > 0
        # Check both form value AND saved state for info_ratio weight
        info_ratio_weight = metric_weights.get("info_ratio", 0)
        saved_info_ratio_weight = model_state.get("metric_weights", {}).get("info_ratio", 0)
        show_benchmark_selector = info_ratio_weight > 0 or saved_info_ratio_weight > 0

        info_ratio_benchmark = model_state.get("info_ratio_benchmark", "")
        if show_benchmark_selector:
            st.divider()
            st.markdown("**📈 Information Ratio Benchmark**")
            st.caption(
                "Select the index or benchmark column to use for Information Ratio calculation."
            )
            current_benchmark = model_state.get("info_ratio_benchmark", "")
            benchmark_index = 0
            if current_benchmark and current_benchmark in benchmark_options:
                benchmark_index = (
                    benchmark_options.index(current_benchmark) + 1
                )  # +1 for "Select..." option

            benchmark_selection = st.selectbox(
                "Benchmark Column",
                options=["(Select a benchmark)"] + benchmark_options,
                index=benchmark_index,
                help=HELP_TEXT["info_ratio_benchmark"],
                key="benchmark_selector",
            )
            if benchmark_selection != "(Select a benchmark)":
                info_ratio_benchmark = benchmark_selection
        else:
            info_ratio_benchmark = ""

        # Section 4: Risk Settings
        st.divider()
        st.subheader("🎯 Risk Settings")

        risk_c1, risk_c2 = st.columns(2)
        with risk_c1:
            risk_target = st.number_input(
                "Target Portfolio Volatility",
                min_value=0.01,
                max_value=0.50,
                value=float(model_state.get("risk_target", 0.1)),
                step=0.01,
                format="%.2f",
                help=HELP_TEXT["risk_target"],
            )
            st.caption(f"Target: {risk_target:.0%} annualized vol")

        with risk_c2:
            rf_override_enabled = st.checkbox(
                "Override Risk-Free Rate",
                value=bool(model_state.get("rf_override_enabled", False)),
                help=HELP_TEXT["rf_override"],
            )
            # Use checkbox value directly - it's available immediately within form
            rf_rate_pct = st.number_input(
                "Constant RF Rate (%)",
                min_value=0.0,
                max_value=20.0,
                value=float(model_state.get("rf_rate_annual", 0.0)) * 100,
                step=0.25,
                format="%.2f",
                help=HELP_TEXT["rf_rate"],
                disabled=not rf_override_enabled,
            )
            rf_rate_annual = rf_rate_pct / 100.0 if rf_override_enabled else 0.0
            if rf_override_enabled:
                st.caption("⚠️ Using constant RF rate instead of data column.")
            else:
                st.caption("Enable override to enter a constant RF rate.")

        # Volatility floor and warmup
        vol_c1, vol_c2 = st.columns(2)
        with vol_c1:
            vol_floor_pct = st.number_input(
                "Volatility Floor (%)",
                min_value=0.0,
                max_value=10.0,
                value=float(model_state.get("vol_floor", 0.015)) * 100,
                step=0.1,
                format="%.2f",
                help=HELP_TEXT["vol_floor"],
            )
            # Convert to decimal for storage
            vol_floor = vol_floor_pct / 100.0

        with vol_c2:
            warmup_periods = st.number_input(
                "Warmup Periods",
                min_value=0,
                max_value=24,
                value=int(model_state.get("warmup_periods", 0)),
                help=HELP_TEXT["warmup_periods"],
            )

        # Volatility Adjustment Settings (Phase 10) - collapsible
        with st.expander("📊 Volatility Adjustment Details", expanded=False):
            st.caption(
                "Configure how volatility scaling is applied to returns. "
                "These settings control the rolling window for volatility estimation."
            )

            vol_adj_enabled = st.checkbox(
                "Enable Volatility Adjustment",
                value=bool(model_state.get("vol_adjust_enabled", True)),
                help=HELP_TEXT["vol_adjust_enabled"],
            )

            # Always show vol settings, disabled when checkbox is off (forms don't rerun)
            va_c1, va_c2, va_c3 = st.columns(3)
            with va_c1:
                vol_window_length = st.number_input(
                    "Vol Window (periods)",
                    min_value=10,
                    max_value=252,
                    value=int(model_state.get("vol_window_length", 63)),
                    help=HELP_TEXT["vol_window_length"],
                    disabled=not vol_adj_enabled,
                )
                if vol_adj_enabled:
                    st.caption(
                        f"~{vol_window_length // 21} months"
                        if vol_window_length >= 21
                        else f"{vol_window_length} days"
                    )
                else:
                    st.caption("Enable volatility adjustment to configure")

            with va_c2:
                decay_methods = ["ewma", "simple"]
                decay_labels = {
                    "ewma": "EWMA (Exponential)",
                    "simple": "Simple (Equal Weight)",
                }
                current_decay = model_state.get("vol_window_decay", "ewma")
                vol_window_decay = st.selectbox(
                    "Decay Method",
                    options=decay_methods,
                    format_func=lambda x: decay_labels.get(x, x),
                    index=(
                        decay_methods.index(current_decay) if current_decay in decay_methods else 0
                    ),
                    help=HELP_TEXT["vol_window_decay"],
                    disabled=not vol_adj_enabled,
                )

            with va_c3:
                vol_ewma_lambda = st.number_input(
                    "EWMA Lambda",
                    min_value=0.80,
                    max_value=0.99,
                    value=float(model_state.get("vol_ewma_lambda", 0.94)),
                    step=0.01,
                    format="%.2f",
                    help=HELP_TEXT["vol_ewma_lambda"],
                    disabled=(not vol_adj_enabled or vol_window_decay != "ewma"),
                )
                if vol_adj_enabled and vol_window_decay == "ewma":
                    half_life = round(-1 / (1 + 1e-9 - vol_ewma_lambda), 1)
                    st.caption(f"Half-life: ~{half_life:.0f} periods")

        # Store volatility adjustment parameters in model_state
        model_state["vol_adjust_enabled"] = vol_adj_enabled
        model_state["vol_window_length"] = vol_window_length
        model_state["vol_window_decay"] = vol_window_decay
        model_state["vol_ewma_lambda"] = vol_ewma_lambda

        # Section 5: Advanced Settings
        st.divider()
        st.subheader("⚙️ Advanced Settings")
        st.caption("Fine-tune fund addition/removal rules and transaction costs.")

        adv_c1, adv_c2 = st.columns(2)
        with adv_c1:
            cooldown_periods = st.number_input(
                "Cooldown Period",
                min_value=0,
                max_value=20,
                value=int(model_state.get("cooldown_periods", 1)),
                help=HELP_TEXT["cooldown_periods"],
            )
            max_turnover = st.number_input(
                "Maximum Turnover",
                min_value=0.0,
                max_value=2.0,
                value=float(model_state.get("max_turnover", 1.0)),
                step=0.1,
                format="%.1f",
                help=HELP_TEXT["max_turnover"],
            )

        with adv_c2:
            rebalance_options = ["M", "Q", "A"]
            rebalance_labels = {"M": "Monthly", "Q": "Quarterly", "A": "Annually"}
            current_rebal = model_state.get("rebalance_freq", "M")
            rebalance_freq = st.selectbox(
                "Rebalance Frequency",
                options=rebalance_options,
                format_func=lambda x: rebalance_labels.get(x, x),
                index=(
                    rebalance_options.index(current_rebal)
                    if current_rebal in rebalance_options
                    else 0
                ),
                help=HELP_TEXT["rebalance_freq"],
            )
            transaction_cost_bps = st.number_input(
                "Transaction Cost (bps)",
                min_value=0,
                max_value=100,
                value=int(model_state.get("transaction_cost_bps", 0)),
                help=HELP_TEXT["transaction_cost_bps"],
            )
            if transaction_cost_bps > 0:
                st.caption(f"Each trade incurs a {transaction_cost_bps} bp cost.")

        # Section 6: Fund Holding Rules (Phase 3)
        st.divider()
        st.subheader("🔒 Fund Holding Rules")
        st.caption("Control fund tenure and portfolio churn limits.")

        hold_c1, hold_c2 = st.columns(2)
        with hold_c1:
            min_tenure_periods = st.number_input(
                "Min Tenure (periods)",
                min_value=0,
                max_value=24,
                value=int(model_state.get("min_tenure_periods", 3)),
                help=HELP_TEXT["min_tenure"],
            )
            if min_tenure_periods > 0:
                st.caption(f"Funds held for at least {min_tenure_periods} periods")

        with hold_c2:
            max_changes_per_period = st.number_input(
                "Max Changes/Period",
                min_value=0,
                max_value=50,
                value=int(model_state.get("max_changes_per_period", 0)),
                help=HELP_TEXT["max_changes"],
            )
            if max_changes_per_period == 0:
                st.caption("Unlimited changes allowed")
            else:
                st.caption(f"Max {max_changes_per_period} adds/removes")

        # NOTE: Legacy "max_active_positions" UI control removed.
        # Portfolio sizing should be configured via:
        # - Target Funds (initial): selection_count
        # - Minimum Funds: mp_min_funds
        # - Maximum Funds: mp_max_funds
        max_active_positions = 0

        # Section 7: Trend Signal Settings - REMOVED FROM UI
        # These settings require daily returns to be meaningful.
        # With monthly returns, they would be inappropriate.
        # Using default values; see docs/TrendSignalSettings.md for documentation.
        trend_window = int(model_state.get("trend_window", 63))
        trend_lag = int(model_state.get("trend_lag", 1))
        trend_min_periods_out = model_state.get("trend_min_periods")
        trend_zscore = bool(model_state.get("trend_zscore", False))
        trend_vol_adjust = bool(model_state.get("trend_vol_adjust", False))
        trend_vol_target_out = model_state.get("trend_vol_target")

        # Section 8: Regime Analysis (Phase 6) - collapsible
        st.divider()
        with st.expander("🔄 Regime Analysis (Advanced)", expanded=False):
            st.caption(
                "Enable regime detection to adjust portfolio behavior based on "
                "market conditions (risk-on vs risk-off)."
            )

            reg_c1, reg_c2 = st.columns(2)
            with reg_c1:
                regime_enabled = st.checkbox(
                    "Enable Regime Detection",
                    value=bool(model_state.get("regime_enabled", False)),
                    help=HELP_TEXT["regime_enabled"],
                )

            with reg_c2:
                # Use benchmark columns as regime proxy options
                regime_proxy_options = ["SPX", "TSX", "MSCI", "ACWI"] + [
                    c for c in benchmark_options if c.upper() not in ["SPX", "TSX", "MSCI", "ACWI"]
                ][
                    :10
                ]  # Limit to 14 options
                current_regime_proxy = model_state.get("regime_proxy", "SPX")
                regime_proxy = st.selectbox(
                    "Regime Proxy Index",
                    options=regime_proxy_options,
                    index=(
                        regime_proxy_options.index(current_regime_proxy)
                        if current_regime_proxy in regime_proxy_options
                        else 0
                    ),
                    help=HELP_TEXT["regime_proxy"],
                    disabled=not regime_enabled,
                )

            if regime_enabled:
                st.info(
                    "📊 Regime detection will classify periods as risk-on or risk-off "
                    f"based on {regime_proxy} returns and volatility."
                )

        # Section 9: Expert Settings (Phase 7) - collapsible
        st.divider()
        with st.expander("🔧 Expert Settings", expanded=False):
            st.caption(
                "Advanced settings for covariance matrix handling, leverage limits, "
                "and reproducibility. Most users can leave these at defaults."
            )

            # Covariance shrinkage settings
            st.markdown("**Covariance Matrix Robustness**")
            exp_c1, exp_c2 = st.columns(2)
            with exp_c1:
                shrinkage_enabled = st.checkbox(
                    "Enable Shrinkage",
                    value=bool(model_state.get("shrinkage_enabled", True)),
                    help=HELP_TEXT["shrinkage_enabled"],
                )

            with exp_c2:
                shrinkage_methods = ["ledoit_wolf", "oas", "none"]
                shrinkage_labels = {
                    "ledoit_wolf": "Ledoit-Wolf",
                    "oas": "Oracle Approximating (OAS)",
                    "none": "None (raw)",
                }
                current_shrinkage = model_state.get("shrinkage_method", "ledoit_wolf")
                shrinkage_method = st.selectbox(
                    "Shrinkage Method",
                    options=shrinkage_methods,
                    format_func=lambda x: shrinkage_labels.get(x, x),
                    index=(
                        shrinkage_methods.index(current_shrinkage)
                        if current_shrinkage in shrinkage_methods
                        else 0
                    ),
                    help=HELP_TEXT["shrinkage_method"],
                    disabled=not shrinkage_enabled,
                )

            # Phase 14: Robustness fallbacks
            st.markdown("**Numerical Stability & Fallback**")
            rob_c1, rob_c2 = st.columns(2)
            with rob_c1:
                condition_threshold = st.number_input(
                    "Condition Number Threshold",
                    min_value=1.0e6,
                    max_value=1.0e15,
                    value=float(model_state.get("condition_threshold", 1.0e12)),
                    format="%.0e",
                    help=HELP_TEXT["condition_threshold"],
                )
            with rob_c2:
                safe_modes = ["hrp", "risk_parity", "equal"]
                safe_mode_labels = {
                    "hrp": "HRP (Hierarchical Risk Parity)",
                    "risk_parity": "Risk Parity",
                    "equal": "Equal Weight",
                }
                current_safe_mode = model_state.get("safe_mode", "hrp")
                safe_mode = st.selectbox(
                    "Fallback Method",
                    options=safe_modes,
                    format_func=lambda x: safe_mode_labels.get(x, x),
                    index=(
                        safe_modes.index(current_safe_mode)
                        if current_safe_mode in safe_modes
                        else 0
                    ),
                    help=HELP_TEXT["safe_mode"],
                )
            st.caption(
                f"If condition number exceeds {condition_threshold:.0e}, "
                f"fallback to {safe_mode_labels.get(safe_mode, safe_mode)}."
            )

            # Reproducibility
            st.markdown("**Reproducibility**")
            random_seed = st.number_input(
                "Random Seed",
                min_value=0,
                max_value=99999,
                value=int(model_state.get("random_seed", 42)),
                help=HELP_TEXT["random_seed"],
            )

            # Phase 15: Constraints
            st.markdown("**Constraints**")
            long_only = st.checkbox(
                "Long-Only Portfolio",
                value=bool(model_state.get("long_only", True)),
                help=HELP_TEXT["long_only"],
            )
            st.caption(
                "Built-in weighting schemes are long-only unless configured to allow "
                "shorts (e.g., robust_mv min_weight < 0). This setting also affects "
                "custom weights or plugin engines configured outside the UI."
            )
            if not long_only:
                st.warning(
                    "⚠️ Short positions enabled. Ensure your data and strategy "
                    "support short selling."
                )

        # =====================================================================
        # Section 10: Entry/Exit Rules (Phase 5) - conditional on selection mode
        # =====================================================================
        st.divider()
        with st.expander("🚪 Entry/Exit Rules", expanded=False):
            # Use model_state for disabled since form checkbox changes don't apply until save
            mp_enabled_state = bool(model_state.get("multi_period_enabled", True))

            if not mp_enabled_state:
                st.warning(
                    "⚠️ Entry/exit rules only apply in multi-period mode. "
                    "Enable multi-period above, save config, then configure these settings."
                )

            st.caption(
                "Configure how funds are added to and removed from the portfolio. "
                "These settings control manager hiring and firing decisions."
            )

            # RANKING STABILITY - applies to ranking modes (Top-N, Top-%)
            if is_ranking_mode:
                st.markdown("**Ranking Stability (for Top-N / Top-% modes)**")
                st.info(
                    "Stability settings prevent churning by requiring consistent ranking "
                    "before adding or removing a fund."
                )

                rank_c1, rank_c2 = st.columns(2)
                with rank_c1:
                    sticky_add_periods = st.number_input(
                        "Periods in Top-K Before Entry",
                        min_value=1,
                        max_value=12,
                        value=int(model_state.get("sticky_add_periods", 1)),
                        help=HELP_TEXT["sticky_add_periods"],
                        disabled=not mp_enabled_state,
                    )
                    st.caption(f"Fund must rank in top-K for {sticky_add_periods} period(s).")

                with rank_c2:
                    sticky_drop_periods = st.number_input(
                        "Periods Outside Top-K Before Exit",
                        min_value=1,
                        max_value=12,
                        value=int(model_state.get("sticky_drop_periods", 1)),
                        help=HELP_TEXT["sticky_drop_periods"],
                        disabled=not mp_enabled_state,
                    )
                    st.caption(f"Fund must fall out of top-K for {sticky_drop_periods} period(s).")
            elif is_random_mode:
                # Random mode: no ranking stability needed
                st.info(
                    "🎲 **Random Mode**: Funds are randomly selected each period. "
                    "Ranking stability settings do not apply."
                )
                sticky_add_periods = 1
                sticky_drop_periods = 1
            else:
                # Defaults for threshold mode
                sticky_add_periods = 1
                sticky_drop_periods = 1

            # Z-SCORE THRESHOLDS - apply to ALL modes (important for scoring)
            st.markdown("**Z-Score Thresholds**")
            st.info(
                "Z-scores measure fund performance vs peers. Soft thresholds require "
                "consecutive periods; hard thresholds trigger immediate action."
            )

            # Soft thresholds
            st.markdown("*Soft Thresholds (consecutive periods required)*")
            ee_c1, ee_c2 = st.columns(2)
            with ee_c1:
                z_entry_soft = st.number_input(
                    "Entry Threshold (Z-Score)",
                    min_value=-2.0,
                    max_value=3.0,
                    value=float(model_state.get("z_entry_soft", 1.0)),
                    step=0.25,
                    format="%.2f",
                    help=HELP_TEXT["z_entry_soft"],
                    disabled=not mp_enabled_state,
                )
                entry_soft_strikes = st.number_input(
                    "Entry Consecutive Periods",
                    min_value=1,
                    max_value=12,
                    value=int(model_state.get("entry_soft_strikes", 1)),
                    help="Fund must pass threshold for this many consecutive periods.",
                    disabled=not mp_enabled_state,
                )
                st.caption(
                    f"Score ≥ {z_entry_soft:.2f}σ for {entry_soft_strikes} period(s) to enter."
                )

            with ee_c2:
                z_exit_soft = st.number_input(
                    "Exit Threshold (Z-Score)",
                    min_value=-3.0,
                    max_value=1.0,
                    value=float(model_state.get("z_exit_soft", -1.0)),
                    step=0.25,
                    format="%.2f",
                    help=HELP_TEXT["z_exit_soft"],
                    disabled=not mp_enabled_state,
                )
                soft_strikes = st.number_input(
                    "Exit Consecutive Periods",
                    min_value=1,
                    max_value=12,
                    value=int(model_state.get("soft_strikes", 2)),
                    help="Fund must fail threshold for this many consecutive periods.",
                    disabled=not mp_enabled_state,
                )
                st.caption(f"Score ≤ {z_exit_soft:.2f}σ for {soft_strikes} period(s) to exit.")

            st.markdown("**Underweight Exit (Weight-based)**")
            min_weight_strikes = st.number_input(
                "Periods Underweight Before Forced Exit",
                min_value=0,
                max_value=12,
                value=int(model_state.get("min_weight_strikes", 2) or 2),
                help=HELP_TEXT["min_weight_strikes"],
                disabled=not mp_enabled_state,
            )
            st.caption(
                (
                    f"Same rule as engine log reason=low_weight_strikes; triggers after {min_weight_strikes} period(s)."
                    if min_weight_strikes > 0
                    else "Underweight exit is disabled."
                )
            )

            # Hard thresholds
            st.markdown("*Hard Thresholds (immediate action)*")
            hard_c1, hard_c2 = st.columns(2)
            with hard_c1:
                z_entry_hard_val = model_state.get("z_entry_hard")
                z_entry_hard_enabled = st.checkbox(
                    "Enable Hard Entry",
                    value=z_entry_hard_val is not None,
                    disabled=not mp_enabled_state,
                )
                z_entry_hard = st.number_input(
                    "Hard Entry Z-Score",
                    min_value=0.0,
                    max_value=5.0,
                    value=float(z_entry_hard_val if z_entry_hard_val is not None else 2.0),
                    step=0.25,
                    format="%.2f",
                    help=HELP_TEXT["z_entry_hard"],
                    disabled=not (mp_enabled_state and z_entry_hard_enabled),
                )
                if not z_entry_hard_enabled:
                    z_entry_hard = None
                st.caption(
                    f"Score ≥ {z_entry_hard or 2.0:.2f}σ enters instantly."
                    if z_entry_hard_enabled
                    else "Hard entry disabled."
                )

            with hard_c2:
                z_exit_hard_val = model_state.get("z_exit_hard")
                z_exit_hard_enabled = st.checkbox(
                    "Enable Hard Exit",
                    value=z_exit_hard_val is not None,
                    disabled=not mp_enabled_state,
                )
                z_exit_hard = st.number_input(
                    "Hard Exit Z-Score",
                    min_value=-5.0,
                    max_value=0.0,
                    value=float(z_exit_hard_val if z_exit_hard_val is not None else -2.0),
                    step=0.25,
                    format="%.2f",
                    help=HELP_TEXT["z_exit_hard"],
                    disabled=not (mp_enabled_state and z_exit_hard_enabled),
                )
                if not z_exit_hard_enabled:
                    z_exit_hard = None
                st.caption(
                    f"Score ≤ {z_exit_hard or -2.0:.2f}σ exits instantly."
                    if z_exit_hard_enabled
                    else "Hard exit disabled."
                )

            # Confidence Interval Reporting
            st.markdown("**Confidence Interval (Reporting Only)**")
            ci_level = st.slider(
                "Confidence Interval Level",
                min_value=0.0,
                max_value=0.99,
                value=float(model_state.get("ci_level", 0.0)),
                step=0.05,
                format="%.2f",
                help=HELP_TEXT["ci_level"],
                disabled=not mp_enabled_state,
            )
            if ci_level > 0:
                st.caption(
                    f"Reporting uses {ci_level * 100:.0f}% confidence; portfolio construction is unchanged."
                )
            else:
                st.caption("Confidence interval reporting is disabled.")

        # =====================================================================
        # Section 11: Advanced Selection Settings
        # =====================================================================
        st.divider()
        with st.expander("⚙️ Advanced Selection Settings", expanded=False):
            st.caption("Additional selection parameters for specific modes.")

            # Additional parameters for specific selection modes
            if inclusion_approach == "top_pct":
                # Use value from primary input (defined above) - convert from percentage to decimal
                rank_pct = rank_pct_input / 100.0
                st.info(
                    f"📊 Top Percentage is set to **{rank_pct_input}%** above. "
                    "Adjust it in the Fund Selection Approach section."
                )
            else:
                rank_pct = float(model_state.get("rank_pct", 0.10))

            if inclusion_approach == "threshold":
                st.info(
                    "💡 Z-Score Entry Threshold is configured in the **Entry/Exit Rules** "
                    "section below. The Entry Threshold (Z-Score) setting controls "
                    "which funds are selected."
                )

            st.markdown("**Additional Cost & Exclusion Settings**")
            cost_c1, cost_c2 = st.columns(2)
            with cost_c1:
                slippage_bps = st.number_input(
                    "Slippage (bps)",
                    min_value=0,
                    max_value=50,
                    value=int(model_state.get("slippage_bps", 0)),
                    help=HELP_TEXT["slippage_bps"],
                )
                if slippage_bps > 0:
                    st.caption(
                        f"Additional {slippage_bps} bps ({slippage_bps / 100:.2f}%) "
                        "market impact cost per trade."
                    )

            with cost_c2:
                bottom_k = st.number_input(
                    "Exclude Bottom K Funds",
                    min_value=0,
                    max_value=10,
                    value=int(model_state.get("bottom_k", 0)),
                    help=HELP_TEXT["bottom_k"],
                )
                if bottom_k > 0:
                    st.caption(f"Bottom {bottom_k} ranked funds will always be excluded.")

        # =====================================================================
        # Reporting Options
        # =====================================================================
        st.markdown("---")
        with st.expander("📊 Reporting Options", expanded=False):
            st.markdown("Configure what additional information to include in the Results page.")

            report_c1, report_c2 = st.columns(2)
            with report_c1:
                show_regime_analysis = st.checkbox(
                    "Regime Performance Breakdown",
                    value=bool(model_state.get("report_regime_analysis", False)),
                    help="Show portfolio performance across different market regimes.",
                )
                show_concentration = st.checkbox(
                    "Concentration Metrics",
                    value=bool(model_state.get("report_concentration", True)),
                    help="Show HHI and effective N for portfolio concentration.",
                )
                show_benchmark_comparison = st.checkbox(
                    "Benchmark Comparison Table",
                    value=bool(model_state.get("report_benchmark_comparison", True)),
                    help="Side-by-side comparison with selected benchmarks.",
                )

            with report_c2:
                show_factor_exposures = st.checkbox(
                    "Factor Exposures",
                    value=bool(model_state.get("report_factor_exposures", False)),
                    help="Show factor exposure analysis (requires factor data).",
                )
                show_attribution = st.checkbox(
                    "Volatility-Adjusted Attribution",
                    value=bool(model_state.get("report_attribution", False)),
                    help="Contribution to return by fund, adjusted for volatility.",
                )
                show_rolling_metrics = st.checkbox(
                    "Rolling Performance Metrics",
                    value=bool(model_state.get("report_rolling_metrics", True)),
                    help="Show rolling Sharpe, IR, and other metrics over time.",
                )

        submitted = st.form_submit_button("💾 Save Configuration", type="primary")

        if submitted:
            # Always set to Custom unless user explicitly selects Custom
            effective_preset = "Custom"

            candidate_state = {
                "preset": effective_preset,
                "lookback_periods": lookback,
                "min_history_periods": min_history,
                "evaluation_periods": evaluation,
                "multi_period_frequency": multi_period_frequency,
                "selection_count": selection,
                "weighting_scheme": weighting_value,
                "metric_weights": metric_weights,
                "risk_target": risk_target,
                "info_ratio_benchmark": info_ratio_benchmark,
                # Date settings (preserved from outside form)
                "date_mode": model_state.get("date_mode", "relative"),
                "start_date": model_state.get("start_date"),
                "end_date": model_state.get("end_date"),
                # Risk settings
                "rf_override_enabled": rf_override_enabled,
                "rf_rate_annual": rf_rate_annual,
                "vol_floor": vol_floor,
                "warmup_periods": warmup_periods,
                "vol_adjust_enabled": vol_adj_enabled,
                "vol_window_length": vol_window_length,
                "vol_window_decay": vol_window_decay,
                "vol_ewma_lambda": vol_ewma_lambda,
                # Advanced settings
                "max_weight": max_weight_decimal,
                "min_weight": min_weight_decimal,
                "cooldown_periods": cooldown_periods,
                "rebalance_freq": rebalance_freq,
                "max_turnover": max_turnover,
                "transaction_cost_bps": transaction_cost_bps,
                # Fund holding rules (Phase 3)
                "min_tenure_periods": min_tenure_periods,
                "max_changes_per_period": max_changes_per_period,
                "max_active_positions": max_active_positions,
                # Portfolio signal parameters (Phase 4)
                "trend_window": trend_window,
                "trend_lag": trend_lag,
                "trend_min_periods": trend_min_periods_out,
                "trend_zscore": trend_zscore,
                "trend_vol_adjust": trend_vol_adjust,
                "trend_vol_target": trend_vol_target_out,
                # Regime analysis (Phase 6)
                "regime_enabled": regime_enabled,
                "regime_proxy": regime_proxy,
                # Robustness & Expert settings (Phase 7)
                "shrinkage_enabled": shrinkage_enabled,
                "shrinkage_method": shrinkage_method,
                "random_seed": random_seed,
                # Robustness fallbacks (Phase 14)
                "condition_threshold": condition_threshold,
                "safe_mode": safe_mode,
                # Constraints (Phase 15)
                "long_only": long_only,
                # Entry/Exit thresholds (Phase 5)
                "z_entry_soft": z_entry_soft,
                "z_exit_soft": z_exit_soft,
                "soft_strikes": soft_strikes,
                "entry_soft_strikes": entry_soft_strikes,
                "min_weight_strikes": min_weight_strikes,
                "sticky_add_periods": sticky_add_periods,
                "sticky_drop_periods": sticky_drop_periods,
                "ci_level": ci_level,
                # Multi-period & Selection settings (Phase 8)
                "multi_period_enabled": multi_period_enabled,
                "inclusion_approach": inclusion_approach,
                "buy_hold_initial": buy_hold_initial,
                "slippage_bps": slippage_bps,
                "bottom_k": bottom_k,
                # Selection approach details (Phase 9)
                "rank_pct": rank_pct,
                # Multi-period bounds (Phase 12)
                "mp_min_funds": mp_min_funds,
                "mp_max_funds": mp_max_funds,
                # Hard thresholds (Phase 13)
                "z_entry_hard": z_entry_hard,
                "z_exit_hard": z_exit_hard,
                # Reporting options
                "report_regime_analysis": show_regime_analysis,
                "report_concentration": show_concentration,
                "report_benchmark_comparison": show_benchmark_comparison,
                "report_factor_exposures": show_factor_exposures,
                "report_attribution": show_attribution,
                "report_rolling_metrics": show_rolling_metrics,
            }
            errors = _validate_model(candidate_state, len(fund_cols) if fund_cols else 0)
            if errors:
                _render_validation_errors(errors)
            else:
                st.session_state["model_state"] = candidate_state
                analysis_runner.clear_cached_analysis()
                app_state.clear_analysis_results()
                st.success("✅ Model configuration saved. Go to Results to run analysis.")


def _should_auto_render() -> bool:
    """Return True when running inside an active Streamlit session."""
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
    except Exception:
        return False
    return get_script_run_ctx() is not None


if _should_auto_render():
    render_model_page()
