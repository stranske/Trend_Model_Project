"""Eval harness for ConfigPatchChain."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import yaml

from tools.prompt_evaluator import EvalResult, _load_config, evaluate_prompt
from trend_analysis.llm.chain import ConfigPatchChain
from trend_analysis.llm.prompts import build_config_patch_prompt
from trend_analysis.llm.providers import LLMProviderConfig, create_llm
from trend_analysis.llm.schema import load_compact_schema

BASE_CONFIG: dict[str, Any] = {
    "analysis": {
        "weighting": {"scheme": "equal"},
        "top_n": 8,
        "target_vol": 0.10,
        "frequency": "D",
    },
    "constraints": {"max_weight": 0.2},
}

BASE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "analysis": {
            "type": "object",
            "properties": {
                "weighting": {
                    "type": "object",
                    "properties": {"scheme": {"type": "string"}},
                },
                "top_n": {"type": "integer"},
                "target_vol": {"type": "number"},
                "frequency": {"type": "string"},
            },
        },
        "constraints": {
            "type": "object",
            "properties": {"max_weight": {"type": "number"}},
        },
    },
}

DEFAULT_CASES: list[dict[str, Any]] = [
    {
        "id": "risk_parity_weighting",
        "instruction": "Use risk parity weighting.",
        "current_config": BASE_CONFIG,
        "allowed_schema": BASE_SCHEMA,
        "expected_patch": {
            "operations": [
                {"op": "set", "path": "analysis.weighting.scheme", "value": "risk_parity"}
            ],
            "risk_flags": [],
            "summary": "Set weighting scheme to risk_parity.",
        },
    },
    {
        "id": "select_top_12",
        "instruction": "Select top 12 funds.",
        "current_config": BASE_CONFIG,
        "allowed_schema": BASE_SCHEMA,
        "expected_patch": {
            "operations": [{"op": "set", "path": "analysis.top_n", "value": 12}],
            "risk_flags": [],
            "summary": "Set top_n to 12.",
        },
    },
    {
        "id": "remove_position_limits",
        "instruction": "Remove position limits.",
        "current_config": BASE_CONFIG,
        "allowed_schema": BASE_SCHEMA,
        "expected_patch": {
            "operations": [{"op": "remove", "path": "constraints.max_weight", "value": None}],
            "risk_flags": ["REMOVES_CONSTRAINT"],
            "summary": "Remove max_weight constraint.",
        },
    },
    {
        "id": "target_vol_15",
        "instruction": "Target 15% volatility.",
        "current_config": BASE_CONFIG,
        "allowed_schema": BASE_SCHEMA,
        "expected_patch": {
            "operations": [{"op": "set", "path": "analysis.target_vol", "value": 0.15}],
            "risk_flags": [],
            "summary": "Set target_vol to 0.15.",
        },
    },
    {
        "id": "monthly_and_risk_parity",
        "instruction": "Use monthly frequency and risk parity.",
        "current_config": BASE_CONFIG,
        "allowed_schema": BASE_SCHEMA,
        "expected_patch": {
            "operations": [
                {"op": "set", "path": "analysis.frequency", "value": "M"},
                {"op": "set", "path": "analysis.weighting.scheme", "value": "risk_parity"},
            ],
            "risk_flags": [],
            "summary": "Set frequency to monthly and weighting to risk_parity.",
        },
    },
    {
        "id": "unknown_key_request",
        "instruction": "Enable turbo mode.",
        "case_tags": ["error"],
        "current_config": BASE_CONFIG,
        "allowed_schema": BASE_SCHEMA,
        "expected_patch": {
            "operations": [],
            "risk_flags": [],
            "summary": "Requested unknown key: turbo_mode.",
        },
    },
    {
        "id": "conflicting_top_n",
        "instruction": "Set top_n to 5 and then set top_n to 12.",
        "current_config": BASE_CONFIG,
        "allowed_schema": BASE_SCHEMA,
        "expected_patch": {
            "operations": [{"op": "set", "path": "analysis.top_n", "value": 12}],
            "risk_flags": [],
            "summary": "Resolve conflict by setting top_n to 12.",
        },
    },
    {
        "id": "ambiguous_request",
        "instruction": "Make the portfolio more conservative.",
        "case_tags": ["edge"],
        "current_config": BASE_CONFIG,
        "allowed_schema": BASE_SCHEMA,
        "expected_patch": {
            "operations": [],
            "risk_flags": [],
            "summary": "Need clarification on requested changes.",
        },
    },
    {
        "id": "conflicting_frequency",
        "instruction": "Use monthly frequency and weekly frequency.",
        "current_config": BASE_CONFIG,
        "allowed_schema": BASE_SCHEMA,
        "expected_patch": {
            "operations": [{"op": "set", "path": "analysis.frequency", "value": "M"}],
            "risk_flags": [],
            "summary": "Chose monthly frequency.",
        },
    },
    {
        "id": "typo_key",
        "instruction": "Set analysis.weighting.sheme to risk_parity.",
        "case_tags": ["edge"],
        "current_config": BASE_CONFIG,
        "allowed_schema": BASE_SCHEMA,
        "expected_patch": {
            "operations": [],
            "risk_flags": [],
            "summary": "Unknown key 'analysis.weighting.sheme'; did you mean 'analysis.weighting.scheme'?",
        },
    },
    {
        "id": "json_pointer_path",
        "instruction": "Use cap weighted scheme.",
        "current_config": BASE_CONFIG,
        "allowed_schema": BASE_SCHEMA,
        "expected_patch": {
            "operations": [
                {"op": "set", "path": "/analysis/weighting/scheme", "value": "cap_weighted"}
            ],
            "risk_flags": [],
            "summary": "Set weighting scheme to cap_weighted.",
        },
    },
    {
        "id": "code_fenced_response",
        "instruction": "Set top_n to 10.",
        "case_tags": ["edge"],
        "current_config": BASE_CONFIG,
        "allowed_schema": BASE_SCHEMA,
        "llm_response": '```json\n{"operations":[{"op":"set","path":"analysis.top_n","value":10}],"risk_flags":[],"summary":"Set top_n to 10."}\n```',
        "expected_patch": {
            "operations": [{"op": "set", "path": "analysis.top_n", "value": 10}],
            "risk_flags": [],
            "summary": "Set top_n to 10.",
        },
    },
    {
        "id": "merge_constraints",
        "instruction": "Set max weight to 10% and add min weight 2%.",
        "current_config": BASE_CONFIG,
        "allowed_schema": BASE_SCHEMA,
        "expected_patch": {
            "operations": [
                {
                    "op": "merge",
                    "path": "constraints",
                    "value": {"max_weight": 0.1, "min_weight": 0.02},
                }
            ],
            "risk_flags": ["BROAD_SCOPE"],
            "summary": "Merge constraint updates for max_weight and min_weight.",
        },
    },
    {
        "id": "append_tag",
        "instruction": "Tag this config as momentum.",
        "current_config": BASE_CONFIG,
        "allowed_schema": BASE_SCHEMA,
        "expected_patch": {
            "operations": [{"op": "append", "path": "analysis.tags", "value": "momentum"}],
            "risk_flags": [],
            "summary": "Append tag 'momentum' to analysis.tags.",
        },
    },
    {
        "id": "invalid_value_request",
        "instruction": "Set top_n to -5.",
        "case_tags": ["error"],
        "current_config": BASE_CONFIG,
        "allowed_schema": BASE_SCHEMA,
        "expected_patch": {
            "operations": [],
            "risk_flags": [],
            "summary": "Rejected invalid top_n value; requested value is out of range.",
        },
    },
    {
        "id": "retry_invalid_json_then_success",
        "instruction": "Set top_n to 9.",
        "current_config": BASE_CONFIG,
        "allowed_schema": BASE_SCHEMA,
        "retries": 1,
        "llm_responses": [
            "not-json",
            json.dumps(
                {
                    "operations": [{"op": "set", "path": "analysis.top_n", "value": 9}],
                    "risk_flags": [],
                    "summary": "Set top_n to 9.",
                },
                ensure_ascii=True,
            ),
        ],
        "expected_log_count": 1,
        "expected_log_fragments": ["ConfigPatch parse attempt 1/2 failed"],
        "expected_patch": {
            "operations": [{"op": "set", "path": "analysis.top_n", "value": 9}],
            "risk_flags": [],
            "summary": "Set top_n to 9.",
        },
    },
    {
        "id": "retry_validation_error_then_success",
        "instruction": "Set target_vol to 0.12.",
        "current_config": BASE_CONFIG,
        "allowed_schema": BASE_SCHEMA,
        "retries": 1,
        "llm_responses": [
            json.dumps(
                {
                    "operations": [{"op": "set", "path": "analysis.target_vol", "value": 0.12}],
                    "risk_flags": [],
                },
                ensure_ascii=True,
            ),
            json.dumps(
                {
                    "operations": [{"op": "set", "path": "analysis.target_vol", "value": 0.12}],
                    "risk_flags": [],
                    "summary": "Set target_vol to 0.12.",
                },
                ensure_ascii=True,
            ),
        ],
        "expected_log_count": 1,
        "expected_log_fragments": ["ConfigPatch parse attempt 1/2 failed"],
        "expected_patch": {
            "operations": [{"op": "set", "path": "analysis.target_vol", "value": 0.12}],
            "risk_flags": [],
            "summary": "Set target_vol to 0.12.",
        },
    },
    {
        "id": "retry_invalid_path_then_success",
        "instruction": "Set top_n to 14.",
        "current_config": BASE_CONFIG,
        "allowed_schema": BASE_SCHEMA,
        "retries": 1,
        "llm_responses": [
            json.dumps(
                {
                    "operations": [{"op": "set", "path": "analysis..top_n", "value": 14}],
                    "risk_flags": [],
                    "summary": "Set top_n to 14.",
                },
                ensure_ascii=True,
            ),
            json.dumps(
                {
                    "operations": [{"op": "set", "path": "analysis.top_n", "value": 14}],
                    "risk_flags": [],
                    "summary": "Set top_n to 14.",
                },
                ensure_ascii=True,
            ),
        ],
        "expected_log_count": 1,
        "expected_log_fragments": ["ConfigPatch parse attempt 1/2 failed"],
        "expected_patch": {
            "operations": [{"op": "set", "path": "analysis.top_n", "value": 14}],
            "risk_flags": [],
            "summary": "Set top_n to 14.",
        },
    },
    {
        "id": "retry_exhausted_invalid_json",
        "instruction": "Set frequency to weekly.",
        "current_config": BASE_CONFIG,
        "allowed_schema": BASE_SCHEMA,
        "retries": 1,
        "llm_responses": ["{bad-json", "{still-bad-json"],
        "expected_error_contains": "Failed to parse ConfigPatch after 2 attempts",
        "expected_log_count": 2,
        "expected_log_fragments": [
            "ConfigPatch parse attempt 1/2 failed",
            "ConfigPatch parse attempt 2/2 failed",
        ],
    },
    {
        "id": "retry_exhausted_validation_error",
        "instruction": "Use monthly frequency.",
        "current_config": BASE_CONFIG,
        "allowed_schema": BASE_SCHEMA,
        "retries": 2,
        "llm_responses": [
            json.dumps(
                {
                    "operations": [{"op": "set", "path": "analysis.frequency", "value": "M"}],
                    "risk_flags": [],
                },
                ensure_ascii=True,
            ),
            json.dumps(
                {
                    "operations": [{"op": "set", "path": "analysis.frequency", "value": "M"}],
                    "risk_flags": [],
                },
                ensure_ascii=True,
            ),
            json.dumps(
                {
                    "operations": [{"op": "set", "path": "analysis.frequency", "value": "M"}],
                    "risk_flags": [],
                },
                ensure_ascii=True,
            ),
        ],
        "expected_error_contains": "Failed to parse ConfigPatch after 3 attempts",
        "expected_log_count": 3,
        "expected_log_fragments": [
            "ConfigPatch parse attempt 1/3 failed",
            "ConfigPatch parse attempt 2/3 failed",
            "ConfigPatch parse attempt 3/3 failed",
        ],
    },
    {
        "id": "success_merge_analysis_settings",
        "instruction": "Set frequency to weekly and top_n to 6.",
        "current_config": BASE_CONFIG,
        "allowed_schema": BASE_SCHEMA,
        "expected_patch": {
            "operations": [
                {"op": "merge", "path": "analysis", "value": {"frequency": "W", "top_n": 6}}
            ],
            "risk_flags": ["BROAD_SCOPE"],
            "summary": "Merge frequency and top_n updates into analysis.",
        },
    },
    {
        "id": "success_remove_constraint_json_pointer",
        "instruction": "Remove the max weight constraint.",
        "current_config": BASE_CONFIG,
        "allowed_schema": BASE_SCHEMA,
        "expected_patch": {
            "operations": [{"op": "remove", "path": "/constraints/max_weight", "value": None}],
            "risk_flags": [],
            "summary": "Remove max_weight constraint.",
        },
    },
    {
        "id": "success_set_with_rationale",
        "instruction": "Switch to cap weighted scheme.",
        "current_config": BASE_CONFIG,
        "allowed_schema": BASE_SCHEMA,
        "expected_patch": {
            "operations": [
                {
                    "op": "set",
                    "path": "analysis.weighting.scheme",
                    "value": "cap_weighted",
                }
            ],
            "risk_flags": [],
            "summary": "Set weighting scheme to cap_weighted.",
        },
    },
    {
        "id": "success_append_two_tags",
        "instruction": "Tag with momentum and quality.",
        "current_config": BASE_CONFIG,
        "allowed_schema": BASE_SCHEMA,
        "expected_patch": {
            "operations": [
                {"op": "append", "path": "analysis.tags", "value": "momentum"},
                {"op": "append", "path": "analysis.tags", "value": "quality"},
            ],
            "risk_flags": [],
            "summary": "Append momentum and quality tags to analysis.tags.",
        },
    },
    {
        "id": "success_set_frequency_json_pointer",
        "instruction": "Use monthly frequency.",
        "current_config": BASE_CONFIG,
        "allowed_schema": BASE_SCHEMA,
        "expected_patch": {
            "operations": [{"op": "set", "path": "/analysis/frequency", "value": "M"}],
            "risk_flags": [],
            "summary": "Set frequency to monthly.",
        },
    },
]


def _load_cases(
    path: Path | None,
    *,
    default_cases: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    if path is None:
        if default_cases is None:
            raise ValueError("No test cases provided.")
        return default_cases
    if not path.exists():
        if default_cases is not None:
            return default_cases
        raise FileNotFoundError(f"Test case file not found: {path}")
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
    else:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and "cases" in payload:
        payload = payload["cases"]
    if not isinstance(payload, list):
        raise ValueError("Test cases must be a list or contain a top-level 'cases' list.")
    return [_normalize_case(case, base_dir=path.parent) for case in payload]


def _normalize_case(case: dict[str, Any], *, base_dir: Path) -> dict[str, Any]:
    normalized = dict(case)
    if "current_config" not in normalized and "starting_config" in normalized:
        normalized["current_config"] = _load_config(normalized["starting_config"], base_dir)
    expected_ops = normalized.get("expected_operations")
    if "expected_patch" not in normalized and expected_ops is not None:
        expected_patch: dict[str, Any] = {
            "operations": expected_ops,
            "risk_flags": normalized.get("expected_risk_flags", []),
        }
        if "expected_summary" in normalized:
            expected_patch["summary"] = normalized["expected_summary"]
        if "expected_summary_contains" in normalized:
            expected_patch["summary_contains"] = normalized["expected_summary_contains"]
        normalized["expected_patch"] = expected_patch
    return normalized


def _evaluate_case(case: dict[str, Any]) -> EvalResult:
    return evaluate_prompt(case, chain=None, mode="mock")


def _build_report(results: list[EvalResult]) -> dict[str, Any]:
    total = len(results)
    passed = sum(1 for result in results if result.passed)
    success_rate = (passed / total) if total else 0.0
    return {
        "total": total,
        "passed": passed,
        "failed": total - passed,
        "success_rate": round(success_rate, 4),
        "cases": [
            {
                "id": result.case_id,
                "passed": result.passed,
                "errors": result.errors,
                "patch": result.patch,
                "logs": result.logs,
                "duration": result.duration,
            }
            for result in results
        ],
    }


def _format_summary_table(results: list[EvalResult]) -> str:
    headers = ("Case", "Status", "Time(s)", "Errors", "Warnings", "Summary")
    rows: list[tuple[str, str, str, str, str, str]] = []
    for result in results:
        status = "PASS" if result.passed else "FAIL"
        duration_text = "-" if result.duration is None else f"{result.duration:.2f}"
        error_text = "-"
        if result.errors:
            error_text = f"{len(result.errors)}: " + "; ".join(result.errors)
        log_text = "-"
        if result.logs:
            log_text = f"{len(result.logs)}: " + "; ".join(result.logs)
        summary_text = "-"
        if result.patch and isinstance(result.patch, dict):
            summary_text = str(result.patch.get("summary") or "-")
        rows.append((result.case_id, status, duration_text, error_text, log_text, summary_text))

    case_width = max(len(headers[0]), *(len(row[0]) for row in rows)) if rows else len(headers[0])
    status_width = max(len(headers[1]), *(len(row[1]) for row in rows)) if rows else len(headers[1])
    duration_width = max(len(headers[2]), *(len(row[2]) for row in rows)) if rows else len(headers[2])
    error_width = max(len(headers[3]), *(len(row[3]) for row in rows)) if rows else len(headers[3])
    warn_width = max(len(headers[4]), *(len(row[4]) for row in rows)) if rows else len(headers[4])

    lines = [
        f"{headers[0]:<{case_width}}  {headers[1]:<{status_width}}  "
        f"{headers[2]:<{duration_width}}  {headers[3]:<{error_width}}  "
        f"{headers[4]:<{warn_width}}  {headers[5]}",
        f"{'-' * case_width}  {'-' * status_width}  {'-' * duration_width}  "
        f"{'-' * error_width}  {'-' * warn_width}  {'-' * len(headers[5])}",
    ]
    for case_id, status, duration, errors, warnings, summary in rows:
        lines.append(
            f"{case_id:<{case_width}}  {status:<{status_width}}  "
            f"{duration:<{duration_width}}  {errors:<{error_width}}  "
            f"{warnings:<{warn_width}}  {summary}"
        )
    return "\n".join(lines)


def _resolve_provider_config(
    *,
    provider: str | None,
    model: str | None,
    base_url: str | None,
    organization: str | None,
    max_retries: int | None,
    timeout: float | None,
) -> LLMProviderConfig:
    provider_name = (provider or os.environ.get("TREND_LLM_PROVIDER") or "openai").lower()
    supported = {"openai", "anthropic", "ollama"}
    if provider_name not in supported:
        raise ValueError(
            f"Unknown LLM provider '{provider_name}'. Expected one of: {', '.join(sorted(supported))}."
        )
    api_key = os.environ.get("TREND_LLM_API_KEY")
    if not api_key:
        if provider_name == "openai":
            api_key = os.environ.get("OPENAI_API_KEY")
        elif provider_name == "anthropic":
            api_key = os.environ.get("ANTHROPIC_API_KEY")
    model_name = model or os.environ.get("TREND_LLM_MODEL")
    base_url = base_url or os.environ.get("TREND_LLM_BASE_URL")
    organization = organization or os.environ.get("TREND_LLM_ORG")
    max_retries_value = max_retries
    timeout_value = timeout
    if max_retries_value is None:
        max_retries_env = os.environ.get("TREND_LLM_MAX_RETRIES")
        if max_retries_env:
            max_retries_value = int(max_retries_env)
    if timeout_value is None:
        timeout_env = os.environ.get("TREND_LLM_TIMEOUT")
        if timeout_env:
            timeout_value = float(timeout_env)
    config_kwargs: dict[str, Any] = {"provider": provider_name}
    if model_name:
        config_kwargs["model"] = model_name
    if api_key:
        config_kwargs["api_key"] = api_key
    if base_url:
        config_kwargs["base_url"] = base_url
    if organization:
        config_kwargs["organization"] = organization
    if max_retries_value is not None:
        config_kwargs["max_retries"] = max_retries_value
    if timeout_value is not None:
        config_kwargs["timeout"] = timeout_value
    return LLMProviderConfig(**config_kwargs)


def _build_live_chain(
    *,
    provider: str | None,
    model: str | None,
    temperature: float | None,
    max_tokens: int | None,
    max_retries: int | None,
    timeout: float | None,
    base_url: str | None,
    organization: str | None,
) -> ConfigPatchChain:
    config = _resolve_provider_config(
        provider=provider,
        model=model,
        base_url=base_url,
        organization=organization,
        max_retries=max_retries,
        timeout=timeout,
    )
    llm = create_llm(config)
    schema = load_compact_schema()
    return ConfigPatchChain.from_env(
        llm=llm,
        schema=schema,
        prompt_builder=build_config_patch_prompt,
        temperature=temperature,
        model=config.model,
        max_tokens=max_tokens,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate ConfigPatchChain outputs.")
    parser.add_argument(
        "--cases",
        type=Path,
        default=Path("tools/eval_test_cases.yml"),
        help="Path to JSON/YAML test case file.",
    )
    parser.add_argument(
        "--use-default-cases",
        action="store_true",
        help="Use embedded test cases instead of reading from a file.",
    )
    parser.add_argument(
        "--mode",
        choices=("mock", "live"),
        default="mock",
        help="Evaluation mode: mock uses canned LLM responses; live calls a real provider.",
    )
    parser.add_argument(
        "--live-mode",
        action="store_true",
        help="Run evaluation in live mode (alias for --mode=live).",
    )
    parser.add_argument(
        "--provider",
        help="LLM provider for live mode (defaults to TREND_LLM_PROVIDER or openai).",
    )
    parser.add_argument("--model", help="Override the model for live mode.")
    parser.add_argument("--temperature", type=float, help="Override model temperature.")
    parser.add_argument("--max-tokens", type=int, help="Override max tokens for live mode.")
    parser.add_argument("--max-retries", type=int, help="Override provider max retries.")
    parser.add_argument("--timeout", type=float, help="Override provider timeout in seconds.")
    parser.add_argument("--base-url", help="Override provider base URL.")
    parser.add_argument("--organization", help="Override provider organization.")
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("tools/eval_report.json"),
        help="Path to write evaluation report JSON.",
    )
    args = parser.parse_args(argv)
    if args.live_mode:
        args.mode = "live"

    try:
        case_path = None if args.use_default_cases else args.cases
        cases = _load_cases(case_path, default_cases=DEFAULT_CASES)
    except Exception as exc:
        print(f"Failed to load cases: {exc}", file=sys.stderr)
        return 1

    if args.mode == "live":
        try:
            chain = _build_live_chain(
                provider=args.provider,
                model=args.model,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                max_retries=args.max_retries,
                timeout=args.timeout,
                base_url=args.base_url,
                organization=args.organization,
            )
        except Exception as exc:
            print(f"Failed to initialize live mode: {exc}", file=sys.stderr)
            return 1
        results = [evaluate_prompt(case, chain=chain, mode="live") for case in cases]
    else:
        results = [_evaluate_case(case) for case in cases]
    report = _build_report(results)
    try:
        args.report.write_text(json.dumps(report, indent=2, ensure_ascii=True), encoding="utf-8")
    except Exception as exc:
        print(f"Failed to write report: {exc}", file=sys.stderr)
        return 1

    print(
        f"Evaluated {report['total']} cases. "
        f"Passed {report['passed']} ({report['success_rate'] * 100:.1f}%)."
    )
    print(_format_summary_table(results))
    failed = [result for result in results if not result.passed]
    if failed:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
