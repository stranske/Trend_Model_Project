"""Explain Results UI helpers for the Streamlit app."""

from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Mapping
from uuid import uuid4

import pandas as pd
import streamlit as st

from trend_analysis.llm import (
    ResultClaimIssue,
    ResultSummaryChain,
    ResultSummaryResponse,
    build_deterministic_feedback,
    build_result_summary_prompt,
    compact_metric_catalog,
    create_llm,
    ensure_result_disclaimer,
    extract_metric_catalog,
    format_metric_catalog,
    postprocess_result_text,
    serialize_claim_issue,
)
from streamlit_app.components.llm_settings import (
    default_api_key as _default_api_key,
    resolve_api_key_input as _resolve_api_key_input,
    resolve_llm_provider_config as _resolve_llm_provider_config,
    sanitize_api_key as _sanitize_api_key,
)

DEFAULT_QUESTION = """Analyze this manager selection backtest:
1. What drove the selection of specific managers over alternatives?
2. Which managers persisted across periods and why - true quality or lack of underperformance?
3. Are there signs the selection rules favored managers that entered during benign periods?
4. What parameter changes might improve future selection quality?"""
_CACHE_KEY = "explain_results_cache"
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ExplanationResult:
    text: str
    trace_url: str | None
    claim_issues: list[ResultClaimIssue]
    metric_count: int
    created_at: str


def _cache_bucket() -> dict[str, ExplanationResult]:
    cache = st.session_state.get(_CACHE_KEY)
    if not isinstance(cache, dict):
        cache = {}
        st.session_state[_CACHE_KEY] = cache
    return cache


def _resolve_explanation_run_id(details: Mapping[str, Any], run_key: str) -> str:
    candidates = [
        details.get("run_id"),
        (details.get("metadata") or {}).get("run_id"),
        ((details.get("metadata") or {}).get("reporting") or {}).get("run_id"),
        (details.get("reporting") or {}).get("run_id"),
    ]
    for value in candidates:
        if isinstance(value, str) and value.strip():
            return value.strip()
    return hashlib.sha256(run_key.encode("utf-8")).hexdigest()[:12]


def _render_analysis_output(details: Mapping[str, Any]) -> str:
    parts: list[str] = []
    summary = pd.DataFrame()
    try:
        from trend_analysis.export import summary_frame_from_result

        summary = summary_frame_from_result(details)
    except Exception:
        summary = pd.DataFrame()
    if not summary.empty:
        parts.append("Summary table:\n" + summary.to_string(index=False))
    else:
        parts.append("Summary table unavailable.")
    sections = ", ".join(sorted(str(key) for key in details.keys()))
    if sections:
        parts.append(f"Available sections: {sections}")
    return "\n\n".join(parts)


def _format_questions(raw: str | None) -> str:
    if raw is None:
        raw = ""
    lines = [line.strip() for line in raw.splitlines() if line.strip()]
    if not lines:
        lines = [DEFAULT_QUESTION]
    return "\n".join(f"- {line}" for line in lines)


def _build_result_chain(
    provider: str | None = None,
    *,
    api_key: str | None = None,
    model: str | None = None,
    base_url: str | None = None,
    organization: str | None = None,
) -> ResultSummaryChain:
    config = _resolve_llm_provider_config(
        provider,
        api_key=api_key,
        model=model,
        base_url=base_url,
        organization=organization,
    )
    llm = create_llm(config)
    return ResultSummaryChain.from_env(
        llm=llm,
        prompt_builder=build_result_summary_prompt,
        model=config.model,
    )


def generate_result_explanation(
    details: Mapping[str, Any],
    *,
    questions: str | None,
    provider: str | None = None,
    api_key: str | None = None,
    model: str | None = None,
    base_url: str | None = None,
    organization: str | None = None,
) -> ExplanationResult:
    created_at = datetime.now(timezone.utc).isoformat()
    all_entries = extract_metric_catalog(details)
    compacted_entries = compact_metric_catalog(all_entries, questions=questions)
    metric_catalog = format_metric_catalog(compacted_entries)
    if not all_entries:
        text = ensure_result_disclaimer(
            "No metrics were detected in the analysis output."
        )
        return ExplanationResult(
            text=text,
            trace_url=None,
            claim_issues=[],
            metric_count=0,
            created_at=created_at,
        )

    chain = _build_result_chain(
        provider,
        api_key=api_key,
        model=model,
        base_url=base_url,
        organization=organization,
    )
    analysis_output = _render_analysis_output(details)
    diagnostics = build_deterministic_feedback(details, all_entries)
    if diagnostics:
        analysis_output = f"{analysis_output}\n\n{diagnostics}"
    response: ResultSummaryResponse = chain.run(
        analysis_output=analysis_output,
        metric_catalog=metric_catalog,
        questions=_format_questions(questions),
        request_id=uuid4().hex,
        metric_entries=all_entries,
    )
    text, claim_issues = postprocess_result_text(
        response.text,
        all_entries,
        include_discrepancy_log=False,
    )
    return ExplanationResult(
        text=text,
        trace_url=response.trace_url,
        claim_issues=claim_issues,
        metric_count=len(compacted_entries),
        created_at=created_at,
    )


def render_explain_results(
    result: Any,
    *,
    run_key: str,
    provider: str | None = None,
) -> None:
    st.subheader("Explain Results")
    st.caption("Generate a natural-language explanation of the latest analysis run.")

    details = getattr(result, "details", None)
    if not isinstance(details, Mapping):
        st.info("Explanation is unavailable because detailed results are missing.")
        return

    question_key = "explain_results_questions"
    st.text_area(
        "Questions (optional)",
        value=st.session_state.get(question_key, DEFAULT_QUESTION),
        key=question_key,
        help="Leave blank to use the default summary prompt.",
    )
    st.markdown("#### LLM Settings")
    with st.expander("Configure provider and API key", expanded=False):
        provider_key = "explain_results_provider"
        api_key_key = "explain_results_api_key"
        model_key = "explain_results_model"
        base_url_key = "explain_results_base_url"
        org_key = "explain_results_org"

        provider_default = (
            st.session_state.get(provider_key)
            or provider
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
        # Pre-populate API key from environment if empty or not set
        current_api_key = st.session_state.get(api_key_key)
        if not current_api_key or not _sanitize_api_key(current_api_key):
            env_key = _default_api_key(provider_default)
            if env_key:
                st.session_state[api_key_key] = env_key
        st.text_input(
            "API Key",
            value="",  # Ignored when key is set; session_state takes precedence
            key=api_key_key,
            type="password",
            help=(
                "Leave blank to use a stored key, or enter a secret/env var name "
                "(for example OPENAI_API_KEY)."
            ),
        )
        # Show indicator if env key is being used
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
    questions_text = st.session_state.get(question_key)
    if not questions_text:
        questions_text = DEFAULT_QUESTION

    button_key = hashlib.sha256(run_key.encode("utf-8")).hexdigest()[:12]
    clicked = st.button("Explain Results", key=f"btn_explain_results_{button_key}")
    cache = _cache_bucket()
    cached = cache.get(run_key)

    if clicked:
        with st.spinner("Generating explanation..."):
            try:
                raw_key = st.session_state.get("explain_results_api_key")
                resolved_key = _resolve_api_key_input(raw_key)
                # If empty or placeholder, auto-resolve from environment
                if not resolved_key:
                    resolved_key = _default_api_key(
                        st.session_state.get("explain_results_provider") or "openai"
                    )
                cached = generate_result_explanation(
                    details,
                    questions=st.session_state.get(question_key),
                    provider=st.session_state.get("explain_results_provider", provider),
                    api_key=resolved_key,
                    model=st.session_state.get("explain_results_model") or None,
                    base_url=st.session_state.get("explain_results_base_url") or None,
                    organization=st.session_state.get("explain_results_org") or None,
                )
                cache[run_key] = cached
            except Exception as exc:
                st.error("We could not generate an explanation.")
                st.caption(str(exc))
                return

    if cached is None:
        st.info("Click Explain Results to generate a summary.")
        return

    st.markdown(cached.text)
    if cached.trace_url:
        st.caption(f"Trace URL: {cached.trace_url}")

    if cached.claim_issues:
        with st.expander("Discrepancy log", expanded=False):
            for issue in cached.claim_issues:
                st.markdown(f"- {issue.kind}: {issue.message}")
    else:
        st.caption("No discrepancies detected in the explanation.")

    run_id = _resolve_explanation_run_id(details, run_key)
    artifact_payload = {
        "run_id": run_id,
        "created_at": cached.created_at,
        "text": cached.text,
        "metric_count": cached.metric_count,
        "trace_url": cached.trace_url,
        "questions": questions_text,
        "claim_issues": [serialize_claim_issue(issue) for issue in cached.claim_issues],
    }
    columns = st.columns(2)
    if len(columns) >= 2:
        with columns[0]:
            st.download_button(
                "Download explanation (TXT)",
                data=cached.text,
                file_name=f"explanation_{run_id}.txt",
                mime="text/plain",
            )
        with columns[1]:
            st.download_button(
                "Download explanation (JSON)",
                data=json.dumps(artifact_payload, indent=2, sort_keys=True),
                file_name=f"explanation_{run_id}.json",
                mime="application/json",
            )
    else:
        st.download_button(
            "Download explanation (TXT)",
            data=cached.text,
            file_name=f"explanation_{run_id}.txt",
            mime="text/plain",
        )
        st.download_button(
            "Download explanation (JSON)",
            data=json.dumps(artifact_payload, indent=2, sort_keys=True),
            file_name=f"explanation_{run_id}.json",
            mime="application/json",
        )


__all__ = [
    "ExplanationResult",
    "generate_result_explanation",
    "render_explain_results",
]
