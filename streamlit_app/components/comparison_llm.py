"""LLM comparison helpers for the Streamlit app."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Mapping
from uuid import uuid4

import pandas as pd
import streamlit as st

from streamlit_app.components.llm_settings import (
    default_api_key,
    resolve_api_key_input,
    resolve_llm_provider_config,
    sanitize_api_key,
)
from trend_analysis.llm import (
    ResultClaimIssue,
    ResultSummaryChain,
    ResultSummaryResponse,
    build_comparison_prompt,
    compact_metric_catalog,
    create_llm,
    extract_metric_catalog,
    format_metric_catalog,
    postprocess_result_text,
    serialize_claim_issue,
)
from trend_analysis.llm.result_metrics import MetricEntry

DEFAULT_COMPARISON_QUESTIONS = """Compare the two simulations with emphasis on drivers of differences:
1. Which parameter changes most explain the performance and selection differences?
2. How did the changes affect selection, turnover, and risk metrics?
3. Were differences driven by in-sample/out-of-sample dynamics or constraints?
4. What parameter adjustments would you test next based on the observed deltas?"""
_CACHE_KEY = "comparison_llm_cache"


@dataclass(frozen=True)
class ComparisonExplanationResult:
    text: str
    trace_url: str | None
    claim_issues: list[ResultClaimIssue]
    metric_count: int
    created_at: str


def _cache_bucket() -> dict[str, ComparisonExplanationResult]:
    cache = st.session_state.get(_CACHE_KEY)
    if not isinstance(cache, dict):
        cache = {}
        st.session_state[_CACHE_KEY] = cache
    return cache


def _render_analysis_output(details: Mapping[str, Any], label: str) -> str:
    parts: list[str] = [f"Simulation {label} summary:"]
    summary = pd.DataFrame()
    try:
        from trend_analysis.export import summary_frame_from_result

        summary = summary_frame_from_result(details)
    except Exception:
        summary = pd.DataFrame()
    if not summary.empty:
        parts.append(summary.to_string(index=False))
    else:
        parts.append("Summary table unavailable.")
    sections = ", ".join(sorted(str(key) for key in details.keys()))
    if sections:
        parts.append(f"Available sections: {sections}")
    return "\n".join(parts)


def _prefix_metric_entries(
    entries: list[MetricEntry], prefix: str
) -> list[MetricEntry]:
    prefixed: list[MetricEntry] = []
    for entry in entries:
        prefixed.append(
            MetricEntry(
                path=f"{prefix}.{entry.path}",
                value=entry.value,
                source=f"{prefix}:{entry.source}",
            )
        )
    return prefixed


def _build_comparison_chain(
    provider: str | None,
    *,
    api_key: str | None,
    model: str | None,
    base_url: str | None,
    organization: str | None,
) -> ResultSummaryChain:
    config = resolve_llm_provider_config(
        provider,
        api_key=api_key,
        model=model,
        base_url=base_url,
        organization=organization,
    )
    llm = create_llm(config)
    return ResultSummaryChain.from_env(
        llm=llm,
        prompt_builder=build_comparison_prompt,
        model=config.model,
    )


def generate_comparison_explanation(
    details_a: Mapping[str, Any],
    details_b: Mapping[str, Any],
    *,
    config_diff: str,
    questions: str | None,
    label_a: str,
    label_b: str,
    provider: str | None = None,
    api_key: str | None = None,
    model: str | None = None,
    base_url: str | None = None,
    organization: str | None = None,
) -> ComparisonExplanationResult:
    created_at = datetime.now(timezone.utc).isoformat()
    entries_a = extract_metric_catalog(details_a)
    entries_b = extract_metric_catalog(details_b)
    if not entries_a and not entries_b:
        text = "No metrics were detected in either simulation output."
        return ComparisonExplanationResult(
            text=text,
            trace_url=None,
            claim_issues=[],
            metric_count=0,
            created_at=created_at,
        )

    compact_a = compact_metric_catalog(entries_a, questions=questions, max_entries=400)
    compact_b = compact_metric_catalog(entries_b, questions=questions, max_entries=400)
    prefixed_a = _prefix_metric_entries(compact_a, "A")
    prefixed_b = _prefix_metric_entries(compact_b, "B")
    combined_entries = prefixed_a + prefixed_b

    metrics_a = (
        format_metric_catalog(prefixed_a) if prefixed_a else "No metrics available."
    )
    metrics_b = (
        format_metric_catalog(prefixed_b) if prefixed_b else "No metrics available."
    )

    analysis_output = "\n\n".join(
        [
            _render_analysis_output(details_a, "A"),
            _render_analysis_output(details_b, "B"),
            f"Configuration differences ({label_a} vs {label_b}):\n{config_diff or 'No differences found.'}",
        ]
    )

    chain = _build_comparison_chain(
        provider,
        api_key=api_key,
        model=model,
        base_url=base_url,
        organization=organization,
    )
    response: ResultSummaryResponse = chain.run(
        analysis_output=analysis_output,
        metric_catalog="\n\n".join(
            [
                f"Metrics A ({label_a}):\n{metrics_a}",
                f"Metrics B ({label_b}):\n{metrics_b}",
            ]
        ),
        questions=questions or DEFAULT_COMPARISON_QUESTIONS,
        request_id=uuid4().hex,
        metric_entries=combined_entries,
    )
    text, claim_issues = postprocess_result_text(
        response.text,
        combined_entries,
        include_discrepancy_log=False,
    )
    return ComparisonExplanationResult(
        text=text,
        trace_url=response.trace_url,
        claim_issues=claim_issues,
        metric_count=len(combined_entries),
        created_at=created_at,
    )


def render_comparison_llm(
    *,
    result_a: Any,
    result_b: Any,
    label_a: str,
    label_b: str,
    config_diff: str,
    run_key: str,
) -> None:
    st.subheader("LLM Comparison")
    st.caption(
        "Use an LLM to analyze why the two simulations differ, focusing on parameter changes."
    )

    details_a = getattr(result_a, "details", None)
    details_b = getattr(result_b, "details", None)
    if not isinstance(details_a, Mapping) or not isinstance(details_b, Mapping):
        st.info("LLM comparison is unavailable because detailed results are missing.")
        return

    question_key = f"comparison_llm_questions::{run_key}"
    st.text_area(
        "Questions (optional)",
        value=st.session_state.get(question_key, DEFAULT_COMPARISON_QUESTIONS),
        key=question_key,
        help="Leave blank to use the default comparison prompt.",
    )

    st.markdown("#### LLM Settings")
    with st.expander("Configure provider and API key", expanded=False):
        provider_key = f"comparison_llm_provider::{run_key}"
        api_key_key = f"comparison_llm_api_key::{run_key}"
        model_key = f"comparison_llm_model::{run_key}"
        base_url_key = f"comparison_llm_base_url::{run_key}"
        org_key = f"comparison_llm_org::{run_key}"

        provider_default = (
            st.session_state.get(provider_key)
            or os.environ.get("TREND_LLM_PROVIDER")
            or "openai"
        )
        provider_default = str(provider_default).lower()

        st.selectbox(
            "Provider",
            ["openai", "anthropic", "ollama"],
            index=["openai", "anthropic", "ollama"].index(provider_default),
            key=provider_key,
            help="Defaults to TREND_LLM_PROVIDER if set; otherwise OpenAI.",
        )

        current_api_key = st.session_state.get(api_key_key)
        if not current_api_key or not sanitize_api_key(current_api_key):
            env_key = default_api_key(provider_default)
            if env_key:
                st.session_state[api_key_key] = env_key

        st.text_input(
            "API Key",
            value="",
            key=api_key_key,
            type="password",
            help=(
                "Leave blank to use a stored key, or enter a secret/env var name "
                "(for example OPENAI_API_KEY)."
            ),
        )
        if st.session_state.get(api_key_key):
            st.caption("✓ API key configured")
        else:
            st.caption("⚠️ No API key found in environment or secrets")

        st.text_input(
            "Model (optional)",
            value=st.session_state.get(model_key, ""),
            key=model_key,
        )
        st.text_input(
            "Base URL (optional)",
            value=st.session_state.get(base_url_key, ""),
            key=base_url_key,
        )
        st.text_input(
            "Organization (optional)",
            value=st.session_state.get(org_key, ""),
            key=org_key,
        )

    button_key = hashlib.sha256(run_key.encode("utf-8")).hexdigest()[:12]
    clicked = st.button("Explain comparison", key=f"btn_compare_llm_{button_key}")
    cache = _cache_bucket()
    cached = cache.get(run_key)

    if clicked:
        with st.spinner("Generating comparison analysis..."):
            try:
                raw_key = st.session_state.get(api_key_key)
                resolved_key = resolve_api_key_input(raw_key)
                if not resolved_key:
                    resolved_key = default_api_key(
                        st.session_state.get(provider_key) or "openai"
                    )
                cached = generate_comparison_explanation(
                    details_a,
                    details_b,
                    config_diff=config_diff,
                    questions=st.session_state.get(question_key),
                    label_a=label_a,
                    label_b=label_b,
                    provider=st.session_state.get(provider_key),
                    api_key=resolved_key,
                    model=st.session_state.get(model_key) or None,
                    base_url=st.session_state.get(base_url_key) or None,
                    organization=st.session_state.get(org_key) or None,
                )
                cache[run_key] = cached
            except Exception as exc:
                st.error("We could not generate a comparison explanation.")
                st.caption(str(exc))
                return

    if cached is None:
        st.info("Click Explain comparison to generate analysis.")
        return

    st.markdown(cached.text)
    if cached.trace_url:
        st.caption(f"Trace URL: {cached.trace_url}")

    if cached.claim_issues:
        with st.expander("Discrepancy log", expanded=False):
            for issue in cached.claim_issues:
                st.markdown(f"- {issue.kind}: {issue.message}")
    else:
        st.caption("No discrepancies detected in the comparison output.")

    artifact_payload = {
        "created_at": cached.created_at,
        "text": cached.text,
        "metric_count": cached.metric_count,
        "trace_url": cached.trace_url,
        "questions": st.session_state.get(question_key) or DEFAULT_COMPARISON_QUESTIONS,
        "claim_issues": [serialize_claim_issue(issue) for issue in cached.claim_issues],
        "config_diff": config_diff,
        "labels": {"A": label_a, "B": label_b},
    }

    columns = st.columns(2)
    if len(columns) >= 2:
        with columns[0]:
            st.download_button(
                "Download comparison (TXT)",
                data=cached.text,
                file_name="comparison_explanation.txt",
                mime="text/plain",
            )
        with columns[1]:
            st.download_button(
                "Download comparison (JSON)",
                data=json.dumps(artifact_payload, indent=2, sort_keys=True),
                file_name="comparison_explanation.json",
                mime="application/json",
            )
    else:
        st.download_button(
            "Download comparison (TXT)",
            data=cached.text,
            file_name="comparison_explanation.txt",
            mime="text/plain",
        )
        st.download_button(
            "Download comparison (JSON)",
            data=json.dumps(artifact_payload, indent=2, sort_keys=True),
            file_name="comparison_explanation.json",
            mime="application/json",
        )


__all__ = ["render_comparison_llm", "ComparisonExplanationResult"]
