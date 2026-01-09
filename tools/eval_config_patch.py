"""Eval harness for ConfigPatchChain."""

from __future__ import annotations

import argparse
import json
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


def _load_cases(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
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


def _build_llm(response_text: str) -> RunnableLambda:
    def _respond(_prompt_value, **_kwargs) -> str:
        return response_text

    return RunnableLambda(_respond)


def _evaluate_case(case: dict[str, Any]) -> EvalResult:
    case_id = str(case.get("id") or case.get("name") or "case")
    instruction = case.get("instruction")
    current_config = case.get("current_config", {})
    expected = case.get("expected_patch")
    response_text = case.get("llm_response")

    errors: list[str] = []
    if not isinstance(instruction, str) or not instruction.strip():
        errors.append("Missing instruction.")
    if expected is None and response_text is None:
        errors.append("Missing expected_patch or llm_response.")
    if errors:
        return EvalResult(case_id=case_id, passed=False, errors=errors)

    if response_text is None:
        response_text = json.dumps(expected, ensure_ascii=True)

    llm = _build_llm(response_text)
    chain = ConfigPatchChain(
        llm=llm,
        prompt_builder=build_config_patch_prompt,
        schema=case.get("schema"),
    )

    try:
        patch = chain.run(
            current_config=current_config,
            instruction=instruction,
            allowed_schema=_serialize_schema(case.get("allowed_schema")),
            system_prompt=case.get("system_prompt"),
            safety_rules=case.get("safety_rules"),
        )
    except Exception as exc:  # pragma: no cover - surfaced in report
        return EvalResult(case_id=case_id, passed=False, errors=[str(exc)])

    errors = _compare_patch(expected, patch) if expected is not None else []
    patch_payload = patch.model_dump(mode="python")
    return EvalResult(case_id=case_id, passed=not errors, errors=errors, patch=patch_payload)


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
        "--report",
        type=Path,
        default=Path("tools/eval_report.json"),
        help="Path to write evaluation report JSON.",
    )
    args = parser.parse_args(argv)

    try:
        cases = _load_cases(args.cases)
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
