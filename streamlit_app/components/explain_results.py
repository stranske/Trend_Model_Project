"""Explain Results UI helpers for the Streamlit app."""

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

from trend_analysis.llm import (
    LLMProviderConfig,
    ResultClaimIssue,
    ResultSummaryChain,
    ResultSummaryResponse,
    build_result_summary_prompt,
    compact_metric_catalog,
    create_llm,
    ensure_result_disclaimer,
    extract_metric_catalog,
    format_metric_catalog,
    postprocess_result_text,
    serialize_claim_issue,
)

DEFAULT_QUESTION = "Summarize key findings and notable risks in the results."
_CACHE_KEY = "explain_results_cache"


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


def _resolve_llm_provider_config(provider: str | None = None) -> LLMProviderConfig:
    provider_name = (provider or os.environ.get("TREND_LLM_PROVIDER") or "openai").lower()
    supported = {"openai", "anthropic", "ollama"}
    if provider_name not in supported:
        raise ValueError(
            f"Unknown LLM provider '{provider_name}'. "
            f"Expected one of: {', '.join(sorted(supported))}."
        )
    api_key = os.environ.get("TREND_LLM_API_KEY")
    if not api_key:
        if provider_name == "openai":
            api_key = os.environ.get("OPENAI_API_KEY")
        elif provider_name == "anthropic":
            api_key = os.environ.get("ANTHROPIC_API_KEY")
    model = os.environ.get("TREND_LLM_MODEL")
    base_url = os.environ.get("TREND_LLM_BASE_URL")
    organization = os.environ.get("TREND_LLM_ORG")
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


def _build_result_chain(provider: str | None = None) -> ResultSummaryChain:
    config = _resolve_llm_provider_config(provider)
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
) -> ExplanationResult:
    created_at = datetime.now(timezone.utc).isoformat()
    all_entries = extract_metric_catalog(details)
    compacted_entries = compact_metric_catalog(all_entries, questions=questions)
    metric_catalog = format_metric_catalog(compacted_entries)
    if not all_entries:
        text = ensure_result_disclaimer("No metrics were detected in the analysis output.")
        return ExplanationResult(
            text=text,
            trace_url=None,
            claim_issues=[],
            metric_count=0,
            created_at=created_at,
        )

    chain = _build_result_chain(provider)
    analysis_output = _render_analysis_output(details)
    response: ResultSummaryResponse = chain.run(
        analysis_output=analysis_output,
        metric_catalog=metric_catalog,
        questions=_format_questions(questions),
        request_id=uuid4().hex,
        metric_entries=all_entries,
    )
    text, claim_issues = postprocess_result_text(response.text, all_entries)
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

    button_key = hashlib.sha256(run_key.encode("utf-8")).hexdigest()[:12]
    clicked = st.button("Explain Results", key=f"btn_explain_results_{button_key}")
    cache = _cache_bucket()
    cached = cache.get(run_key)

    if clicked:
        with st.spinner("Generating explanation..."):
            try:
                cached = generate_result_explanation(
                    details,
                    questions=st.session_state.get(question_key),
                    provider=provider,
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
