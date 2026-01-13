"""Prompt evaluation helpers for ConfigPatchChain."""

from __future__ import annotations

import json
import logging
import re
import time
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
    duration: float | None = None


def _load_config(config_path: str | Path, base_dir: Path) -> dict[str, Any]:
    path = Path(config_path)
    if not path.is_absolute():
        path = base_dir / path
        if not path.exists():
            path = Path.cwd() / config_path
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected mapping config in {path}")
    return payload


def _serialize_schema(allowed_schema: Any | None) -> str | None:
    if allowed_schema is None:
        return None
    if isinstance(allowed_schema, str):
        schema_path = Path(allowed_schema)
        if schema_path.exists():
            if schema_path.suffix.lower() == ".json":
                return json.dumps(
                    json.loads(schema_path.read_text(encoding="utf-8")),
                    indent=2,
                    ensure_ascii=True,
                )
            if schema_path.suffix.lower() in {".yml", ".yaml"}:
                return json.dumps(
                    yaml.safe_load(schema_path.read_text(encoding="utf-8")),
                    indent=2,
                    ensure_ascii=True,
                )
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


def _validate_constraints(constraints: list[str], patch: ConfigPatch) -> list[str]:
    errors: list[str] = []
    for constraint in constraints:
        normalized = constraint.strip()
        if not normalized:
            continue
        if normalized == "not patch.risk_flags":
            if patch.risk_flags:
                errors.append(f"Constraint failed: {constraint}")
            continue
        if normalized.startswith("patch.operations"):
            match = re.match(r"patch\.operations\s*\|\s*length\s*==\s*(\d+)$", normalized)
            if match is None:
                errors.append(f"Unsupported constraint: {constraint}")
                continue
            expected_length = int(match.group(1))
            if len(patch.operations) != expected_length:
                errors.append(f"Constraint failed: {constraint}")
            continue
        errors.append(f"Unsupported constraint: {constraint}")
    return errors


def _default_summary(operations: Any) -> str:
    count = len(operations) if isinstance(operations, list) else 0
    return f"Apply {count} operation(s)."


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


def evaluate_prompt(
    case: dict[str, Any],
    chain: ConfigPatchChain | None,
    mode: str,
) -> EvalResult:
    case_id = str(case.get("id") or case.get("name") or "case")
    mode_value = (mode or "").strip().lower()
    if mode_value not in {"mock", "live"}:
        return EvalResult(
            case_id=case_id,
            passed=False,
            errors=[f"Unsupported evaluation mode: {mode!r}."],
        )

    instruction = case.get("instruction")
    if "current_config" in case:
        current_config = case.get("current_config", {})
    elif "starting_config" in case:
        current_config = _load_config(case["starting_config"], Path.cwd())
    else:
        current_config = {}
    expected = case.get("expected_patch")
    expected_ops = case.get("expected_operations")
    response_text = case.get("llm_response")
    responses = case.get("llm_responses")
    expected_error_contains = case.get("expected_error_contains")
    expected_log_fragments = case.get("expected_log_fragments", [])
    expected_log_count = case.get("expected_log_count")
    retries = int(case.get("retries", 1))
    constraints = case.get("constraints")

    errors: list[str] = []
    if not isinstance(instruction, str) or not instruction.strip():
        errors.append("Missing instruction.")
    if responses is not None:
        if not isinstance(responses, list):
            errors.append("llm_responses must be a list.")
        elif not responses:
            errors.append("llm_responses must be a non-empty list.")
        elif not all(isinstance(item, str) for item in responses):
            errors.append("llm_responses must contain only strings.")
    if response_text is not None and not isinstance(response_text, str):
        errors.append("llm_response must be a string.")
    if responses is not None and response_text is not None:
        errors.append("Provide only one of llm_response or llm_responses.")
    if expected is None and expected_ops is not None:
        expected = {
            "operations": expected_ops,
            "risk_flags": case.get("expected_risk_flags", []),
        }
        if "expected_summary" in case:
            expected["summary"] = case["expected_summary"]
        if "expected_summary_contains" in case:
            expected["summary_contains"] = case["expected_summary_contains"]
    if expected is None and response_text is None and not responses:
        errors.append("Missing expected_patch or llm_response.")
    if mode_value == "live" and chain is None:
        errors.append("Live mode requires a ConfigPatchChain instance.")
    if constraints is not None and not isinstance(constraints, list):
        errors.append("constraints must be a list.")
    if errors:
        return EvalResult(case_id=case_id, passed=False, errors=errors)

    if mode_value == "mock":
        if responses is None:
            if response_text is None:
                response_patch = case.get("response_patch") or expected or {}
                if isinstance(response_patch, dict):
                    response_patch = dict(response_patch)
                    if "risk_flags" not in response_patch:
                        response_patch["risk_flags"] = []
                    if "summary" not in response_patch:
                        response_patch["summary"] = case.get(
                            "response_summary"
                        ) or _default_summary(response_patch.get("operations"))
                response_text = json.dumps(response_patch, ensure_ascii=True)
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

    start_time = time.perf_counter()
    patch_payload: dict[str, Any] | None = None
    elapsed: float | None = None

    try:
        patch = chain.run(
            current_config=current_config,
            instruction=instruction,
            allowed_schema=_serialize_schema(case.get("allowed_schema")),
            system_prompt=case.get("system_prompt"),
            safety_rules=case.get("safety_rules"),
        )
    except Exception as exc:  # pragma: no cover - surfaced in report
        error_text = str(exc)
        if expected_error_contains:
            if expected_error_contains not in error_text:
                errors.append(
                    f"Expected error containing '{expected_error_contains}', got '{error_text}'."
                )
            errors.extend(
                _validate_log_expectations(log_messages, expected_log_fragments, expected_log_count)
            )
        else:
            errors.append(error_text)
    else:
        errors.extend(
            _validate_log_expectations(log_messages, expected_log_fragments, expected_log_count)
        )
        if expected is not None:
            errors.extend(_compare_patch(expected, patch))
        if expected_error_contains:
            errors.append(
                f"Expected error containing '{expected_error_contains}', but evaluation succeeded."
            )
        if constraints:
            errors.extend(_validate_constraints(constraints, patch))
        patch_payload = patch.model_dump(mode="python")
    finally:
        if handler in chain_logger.handlers:
            chain_logger.removeHandler(handler)
        chain_logger.setLevel(previous_level)

    if start_time is not None:
        elapsed = time.perf_counter() - start_time
        if mode_value == "mock" and elapsed > 10.0:
            errors.append(f"Mock mode execution exceeded 10 seconds ({elapsed:.3f}s).")

    return EvalResult(
        case_id=case_id,
        passed=not errors,
        errors=errors,
        patch=patch_payload,
        logs=log_messages,
        duration=elapsed,
    )
