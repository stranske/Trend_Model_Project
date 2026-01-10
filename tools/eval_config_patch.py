"""Eval harness for ConfigPatchChain."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from langchain_core.runnables import RunnableLambda

from trend_analysis.config.patch import ConfigPatch
from trend_analysis.llm.chain import ConfigPatchChain
from trend_analysis.llm.prompts import build_config_patch_prompt


@dataclass(frozen=True)
class EvalResult:
    case_id: str
    passed: bool
    errors: list[str]
    patch: dict[str, Any] | None = None
    logs: list[str] | None = None


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
        "expected_log_fragments": ["ConfigPatch parse attempt 2/2 failed"],
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
        "expected_log_fragments": ["ConfigPatch parse attempt 3/3 failed"],
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
    return payload


def _serialize_schema(allowed_schema: Any | None) -> str | None:
    if allowed_schema is None:
        return None
    if isinstance(allowed_schema, str):
        return allowed_schema
    return json.dumps(allowed_schema, indent=2, ensure_ascii=True)


def _build_llm(responses: list[str]) -> RunnableLambda:
    if not responses:
        raise ValueError("LLM responses list cannot be empty.")
    index = {"value": 0}

    def _respond(_prompt_value, **_kwargs) -> str:
        current = responses[min(index["value"], len(responses) - 1)]
        index["value"] += 1
        return current

    return RunnableLambda(_respond)


def _validate_log_expectations(
    messages: list[str],
    expected_fragments: list[str],
    expected_count: int | None,
) -> list[str]:
    errors: list[str] = []
    if expected_count is not None and len(messages) != expected_count:
        errors.append(f"Expected {expected_count} log warnings, got {len(messages)}.")
    for fragment in expected_fragments:
        if not any(fragment in message for message in messages):
            errors.append(f"Missing log message containing: {fragment}")
    return errors


def _evaluate_case(case: dict[str, Any]) -> EvalResult:
    case_id = str(case.get("id") or case.get("name") or "case")
    instruction = case.get("instruction")
    current_config = case.get("current_config", {})
    expected = case.get("expected_patch")
    response_text = case.get("llm_response")
    responses = case.get("llm_responses")
    expected_error_contains = case.get("expected_error_contains")
    expected_log_fragments = case.get("expected_log_fragments", [])
    expected_log_count = case.get("expected_log_count")
    retries = int(case.get("retries", 1))

    errors: list[str] = []
    if not isinstance(instruction, str) or not instruction.strip():
        errors.append("Missing instruction.")
    if responses is not None and not isinstance(responses, list):
        errors.append("llm_responses must be a list.")
    if expected is None and response_text is None and not responses:
        errors.append("Missing expected_patch or llm_response.")
    if errors:
        return EvalResult(case_id=case_id, passed=False, errors=errors)

    if responses is None:
        if response_text is None:
            response_text = json.dumps(expected, ensure_ascii=True)
        responses = [response_text]

    llm = _build_llm(responses)
    chain = ConfigPatchChain(
        llm=llm,
        prompt_builder=build_config_patch_prompt,
        schema=case.get("schema"),
        retries=retries,
    )

    log_messages: list[str] = []

    class _ListHandler(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            log_messages.append(record.getMessage())

    handler = _ListHandler(level=logging.WARNING)
    chain_logger = logging.getLogger("trend_analysis.llm.chain")
    previous_level = chain_logger.level
    chain_logger.addHandler(handler)
    chain_logger.setLevel(logging.WARNING)

    try:
        patch = chain.run(
            current_config=current_config,
            instruction=instruction,
            allowed_schema=_serialize_schema(case.get("allowed_schema")),
            system_prompt=case.get("system_prompt"),
            safety_rules=case.get("safety_rules"),
        )
    except Exception as exc:  # pragma: no cover - surfaced in report
        chain_logger.removeHandler(handler)
        chain_logger.setLevel(previous_level)
        error_text = str(exc)
        if expected_error_contains:
            if expected_error_contains not in error_text:
                errors.append(
                    f"Expected error containing '{expected_error_contains}', got '{error_text}'."
                )
            errors.extend(
                _validate_log_expectations(
                    log_messages, expected_log_fragments, expected_log_count
                )
            )
            return EvalResult(
                case_id=case_id,
                passed=not errors,
                errors=errors,
                patch=None,
                logs=log_messages,
            )
        return EvalResult(case_id=case_id, passed=False, errors=[error_text], logs=log_messages)
    finally:
        if handler in chain_logger.handlers:
            chain_logger.removeHandler(handler)
        chain_logger.setLevel(previous_level)

    errors.extend(
        _validate_log_expectations(log_messages, expected_log_fragments, expected_log_count)
    )
    if expected is not None:
        errors.extend(_compare_patch(expected, patch))
    patch_payload = patch.model_dump(mode="python")
    return EvalResult(
        case_id=case_id,
        passed=not errors,
        errors=errors,
        patch=patch_payload,
        logs=log_messages,
    )


def _compare_patch(expected: dict[str, Any], patch: ConfigPatch) -> list[str]:
    errors: list[str] = []
    expected_ops = expected.get("operations")
    if expected_ops is not None:
        if not isinstance(expected_ops, list):
            errors.append("Expected operations must be a list.")
        else:
            if len(expected_ops) != len(patch.operations):
                errors.append(
                    f"Expected {len(expected_ops)} operations, got {len(patch.operations)}."
                )
            for index, expected_op in enumerate(expected_ops):
                if index >= len(patch.operations):
                    break
                actual = patch.operations[index].model_dump(mode="python")
                for key in ("op", "path", "value"):
                    if key in expected_op and actual.get(key) != expected_op.get(key):
                        errors.append(
                            f"Op {index} {key} mismatch: {expected_op.get(key)} != {actual.get(key)}."
                        )
    expected_flags = expected.get("risk_flags")
    if expected_flags is not None:
        if set(expected_flags) != {flag.value for flag in patch.risk_flags}:
            errors.append(
                f"Risk flags mismatch: {expected_flags} != {[flag.value for flag in patch.risk_flags]}."
            )
    summary_contains = expected.get("summary_contains")
    if summary_contains:
        if summary_contains.lower() not in patch.summary.lower():
            errors.append("Summary missing expected phrase.")
    summary_exact = expected.get("summary")
    if summary_exact and summary_exact != patch.summary:
        errors.append("Summary mismatch.")
    return errors


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
            }
            for result in results
        ],
    }


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
        "--report",
        type=Path,
        default=Path("tools/eval_report.json"),
        help="Path to write evaluation report JSON.",
    )
    args = parser.parse_args(argv)

    try:
        case_path = None if args.use_default_cases else args.cases
        cases = _load_cases(case_path, default_cases=DEFAULT_CASES)
    except Exception as exc:
        print(f"Failed to load cases: {exc}", file=sys.stderr)
        return 1

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
    failed = [result for result in results if not result.passed]
    if failed:
        for result in failed:
            print(f"- {result.case_id}: {', '.join(result.errors)}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
