"""Result explanation and explanation-artifact implementations."""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping

import pandas as pd

from trend.mc.viz import TrendCLIError
from trend_analysis import export
from trend_analysis.llm import (
    ResultClaimIssue,
    ResultSummaryChain,
    build_result_summary_prompt,
    create_llm,
    serialize_claim_issue,
)
from trend_analysis.llm.result_validation import (
    append_discrepancy_log,
    ensure_result_disclaimer,
)

from trend.commands.llm_support import _resolve_llm_provider_config

logger = logging.getLogger(__name__)


def _resolve_explain_details_path(args: argparse.Namespace) -> Path:
    if args.details:
        return Path(args.details)
    if not args.run_id:
        raise TrendCLIError("The explain command requires --details or --run-id.")
    artifacts_dir = Path(args.artifacts) if args.artifacts else Path("perf")
    return artifacts_dir / f"details_{args.run_id}.json"


def _load_explain_details(path: Path) -> Mapping[str, Any]:
    if not path.exists():
        raise TrendCLIError(f"Details file not found: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise TrendCLIError(f"Details file is not valid JSON: {path}") from exc
    if not isinstance(payload, Mapping):
        raise TrendCLIError("Details file must contain a JSON object at the root.")
    return payload


def _render_analysis_output(details: Mapping[str, Any]) -> str:
    parts: list[str] = []
    summary = pd.DataFrame()
    try:
        summary = export.summary_frame_from_result(details)
    except Exception as exc:
        logger.warning("Failed to build summary frame from explain details: %s", exc)
        summary = pd.DataFrame()
    if not summary.empty:
        parts.append("Summary table:\n" + summary.to_string(index=False))
    else:
        parts.append("Summary table unavailable.")
    sections = ", ".join(sorted(str(k) for k in details.keys()))
    if sections:
        parts.append(f"Available sections: {sections}")
    return "\n\n".join(parts)


def _resolve_explain_questions(args: argparse.Namespace) -> str:
    questions: list[str] = []
    if args.questions:
        questions.extend([q.strip() for q in args.questions if q and q.strip()])
    if args.questions_file:
        if not args.questions_file.exists():
            raise TrendCLIError(f"Questions file not found: {args.questions_file}")
        raw_lines = args.questions_file.read_text(encoding="utf-8").splitlines()
        questions.extend([line.strip() for line in raw_lines if line.strip()])
    if not questions:
        questions = ["Summarize key findings and notable risks in the results."]
    return "\n".join(f"- {question}" for question in questions)


def _infer_explain_run_id(
    details_path: Path,
    run_id: str | None,
    details: Mapping[str, Any] | None = None,
) -> str:
    if run_id:
        return run_id

    def _mapping(value: Any) -> Mapping[str, Any]:
        return value if isinstance(value, Mapping) else {}

    candidates = []
    if isinstance(details, Mapping):
        metadata = _mapping(details.get("metadata"))
        candidates = [
            details.get("run_id"),
            metadata.get("run_id"),
            _mapping(metadata.get("reporting")).get("run_id"),
            _mapping(details.get("reporting")).get("run_id"),
        ]
    for candidate in candidates:
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()
    name = details_path.name
    prefix = "details_"
    suffix = ".json"
    if name.startswith(prefix) and name.endswith(suffix):
        return name[len(prefix) : -len(suffix)]
    return "unknown"


def _resolve_explain_output_paths(output: Path, run_id: str) -> tuple[Path, Path]:
    if output.exists() and output.is_dir():
        prefix = output / f"explanation_{run_id}"
    elif output.exists() and output.is_file():
        prefix = output.with_suffix("")
    elif output.suffix:
        prefix = output.with_suffix("")
    elif output.name.startswith("explanation_"):
        prefix = output
    else:
        prefix = output / f"explanation_{run_id}"
    return prefix.with_suffix(".txt"), prefix.with_suffix(".json")


def _build_explain_artifact_payload(
    *,
    run_id: str,
    created_at: datetime,
    text: str,
    metric_count: int,
    trace_url: str | None,
    claim_issues: Iterable[ResultClaimIssue],
    questions: str | None = None,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "run_id": run_id,
        "created_at": created_at.isoformat(),
        "text": text,
        "metric_count": metric_count,
        "trace_url": trace_url,
        "claim_issues": [serialize_claim_issue(issue) for issue in claim_issues],
    }
    if questions is not None:
        payload["questions"] = questions
    return payload


def _finalize_explanation_text(
    text: str,
    claim_issues: Iterable[ResultClaimIssue],
) -> str:
    output = text
    if claim_issues and "Discrepancy log:" not in output:
        output = append_discrepancy_log(output, claim_issues)
    return ensure_result_disclaimer(output)


def _write_explain_artifacts(
    *,
    output: Path,
    run_id: str,
    text: str,
    payload: Mapping[str, object],
) -> tuple[Path, Path]:
    txt_path, json_path = _resolve_explain_output_paths(output, run_id)
    txt_path.parent.mkdir(parents=True, exist_ok=True)
    txt_path.write_text(text, encoding="utf-8")
    json_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True),
        encoding="utf-8",
    )
    return txt_path, json_path


def _fallback_explanation(metric_catalog: str) -> str:
    if metric_catalog:
        return (
            "Unable to verify the generated explanation against the available metrics. "
            "Here is the metric catalog:\n"
            f"{metric_catalog}"
        )
    return "No metrics were detected in the analysis output."


def _build_result_chain(provider: str | None = None) -> ResultSummaryChain:
    config = _resolve_llm_provider_config(provider)
    try:
        llm = create_llm(config)
    except Exception as exc:
        raise TrendCLIError(str(exc)) from exc
    return ResultSummaryChain.from_env(
        llm=llm,
        prompt_builder=build_result_summary_prompt,
    )
