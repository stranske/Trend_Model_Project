"""Natural-language config, replay, and patch implementations."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

from trend.commands.llm_support import _resolve_llm_provider_config
from trend.mc.viz import TrendCLIError
from trend_analysis.config import (
    ConfigPatch,
    diff_configs,
    format_validation_messages,
    validate_config,
)
from trend_analysis.config import (
    apply_patch as apply_config_patch,
)
from trend_analysis.config.schema_validation import load_config as load_config_yaml
from trend_analysis.config.validation import ValidationResult
from trend_analysis.llm import (
    ConfigPatchChain,
    build_config_patch_prompt,
    create_llm,
)
from trend_analysis.llm.nl_logging import NLOperationLog, write_nl_log
from trend_analysis.llm.replay import ReplayResult
from trend_analysis.llm.schema import load_compact_schema

logger = logging.getLogger(__name__)


def _build_nl_chain(
    provider: str | None = None,
    *,
    model: str | None = None,
    temperature: float | None = None,
) -> ConfigPatchChain:
    config = _resolve_llm_provider_config(provider, model=model)
    try:
        llm = create_llm(config)
        schema = load_compact_schema()
    except Exception as exc:
        raise TrendCLIError(str(exc)) from exc
    return ConfigPatchChain.from_env(
        llm=llm,
        schema=schema,
        prompt_builder=build_config_patch_prompt,
        model=config.model,
        temperature=temperature,
    )


def _load_nl_log_entry(path: Path, entry: int) -> NLOperationLog:
    from trend_analysis.llm.replay import load_nl_log_entry

    return load_nl_log_entry(path, entry)


def _replay_nl_entry(
    entry: NLOperationLog,
    *,
    provider: str | None = None,
    model: str | None = None,
    temperature: float | None = None,
) -> ReplayResult:
    from trend_analysis.llm.replay import replay_nl_entry

    return replay_nl_entry(entry, provider=provider, model=model, temperature=temperature)


def _build_nl_replay_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="trend nl replay",
        description="Replay a logged NL operation entry.",
    )
    parser.add_argument("log_file", type=Path, help="Path to nl_ops_<date>.jsonl log file")
    parser.add_argument("--entry", type=int, required=True, help="1-based entry index")
    parser.add_argument("--provider", help="Override the logged LLM provider")
    parser.add_argument("--model", help="Override the logged LLM model")
    parser.add_argument("--temperature", type=float, help="Override the logged temperature")
    parser.add_argument("--show-prompt", action="store_true", help="Print the prompt text")
    return parser


def _run_nl_replay(argv: list[str]) -> int:
    parser = _build_nl_replay_parser()
    args = parser.parse_args(argv)
    log_path = Path(args.log_file)
    if not log_path.exists():
        raise TrendCLIError(f"Log file not found: {log_path}")
    try:
        entry = _load_nl_log_entry(log_path, args.entry)
    except (ValueError, IndexError) as exc:
        raise TrendCLIError(str(exc)) from exc
    try:
        result = _replay_nl_entry(
            entry,
            provider=args.provider,
            model=args.model,
            temperature=args.temperature,
        )
    except ValueError as exc:
        raise TrendCLIError(str(exc)) from exc
    if args.show_prompt:
        print("Prompt:")
        print(result.prompt)
    print(f"Prompt hash: {result.prompt_hash}")
    print(f"Output hash: {result.output_hash}")
    if result.trace_url:
        print(f"Trace URL: {result.trace_url}")
    if result.recorded_hash is None:
        print("Recorded hash: <none>")
    else:
        print(f"Recorded hash: {result.recorded_hash}")
    print(f"Matches: {result.matches}")
    if result.recorded_output is None:
        print("Recorded output: <none>")
    else:
        print("Recorded output:")
        print(result.recorded_output)
    if result.recorded_output is None:
        print("Comparison: skipped (no recorded output)")
        exit_code = 0
    elif result.matches:
        print("Comparison: match")
        exit_code = 0
    else:
        print("Comparison: mismatch")
        exit_code = 1
    if result.diff:
        print("Diff:")
        print(result.diff)
    print("Replay output:")
    print(result.output)
    return exit_code


def _maybe_handle_nl_replay(argv: list[str]) -> int | None:
    if len(argv) >= 2 and argv[0] == "nl" and argv[1] == "replay":
        return _run_nl_replay(argv[2:])
    return None


def _load_nl_config(path: Path) -> dict[str, Any]:
    try:
        payload = load_config_yaml(path)
    except Exception as exc:
        raise TrendCLIError(str(exc)) from exc
    if not isinstance(payload, dict):
        raise TrendCLIError("Config file must contain a mapping at the root.")
    return payload


def _hash_nl_payload(payload: dict[str, Any]) -> str:
    text = json.dumps(
        payload, sort_keys=True, ensure_ascii=True, separators=(",", ":"), default=str
    )
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _log_nl_operation(
    *,
    request_id: str,
    operation: str,
    input_payload: dict[str, Any],
    model_name: str,
    temperature: float,
    parsed_patch: ConfigPatch | None = None,
    validation_result: ValidationResult | None = None,
    error: str | None = None,
    started_at: float,
    timestamp: datetime,
) -> None:
    entry = NLOperationLog(
        request_id=request_id,
        timestamp=timestamp,
        operation=cast(Any, operation),
        input_hash=_hash_nl_payload(input_payload),
        prompt_template="",
        prompt_variables={},
        model_output=None,
        parsed_patch=parsed_patch,
        validation_result=validation_result,
        error=error,
        duration_ms=(time.perf_counter() - started_at) * 1000,
        model_name=model_name,
        temperature=temperature,
        token_usage=None,
    )
    write_nl_log(entry)


def _apply_nl_instruction(
    config: dict[str, Any],
    instruction: str,
    *,
    provider: str | None = None,
    model: str | None = None,
    temperature: float | None = None,
    request_id: str,
) -> tuple[ConfigPatch, dict[str, Any], str, str, float]:
    chain = _build_nl_chain(provider, model=model, temperature=temperature)
    try:
        patch = chain.run(current_config=config, instruction=instruction, request_id=request_id)
    except Exception as exc:
        raise TrendCLIError(str(exc)) from exc
    apply_started = time.perf_counter()
    apply_timestamp = datetime.now(timezone.utc)
    apply_error: str | None = None
    try:
        updated = apply_config_patch(config, patch)
    except Exception as exc:
        apply_error = str(exc) or type(exc).__name__
        raise TrendCLIError(str(exc)) from exc
    finally:
        try:
            _log_nl_operation(
                request_id=request_id,
                operation="apply_patch",
                input_payload={
                    "config": config,
                    "patch": patch.model_dump(mode="json"),
                },
                model_name=chain.model or "unknown",
                temperature=chain.temperature,
                parsed_patch=patch,
                error=apply_error,
                started_at=apply_started,
                timestamp=apply_timestamp,
            )
        except Exception as log_exc:  # noqa: BLE001 - logging must not mask the patch error
            logger.warning("Failed to write NL operation log: %s", log_exc)
    diff = diff_configs(config, updated)
    return patch, updated, diff, chain.model or "unknown", chain.temperature


def _format_nl_explanation(patch: ConfigPatch) -> str:
    lines = [f"Summary: {patch.summary}"]
    if patch.risk_flags:
        flags = ", ".join(flag.value for flag in patch.risk_flags)
        lines.append(f"Risk flags: {flags}")
    if patch.needs_review:
        lines.append("Needs review: unknown config keys detected.")
    rationales = [
        (operation.path, operation.rationale)
        for operation in patch.operations
        if operation.rationale
    ]
    if rationales:
        lines.append("Rationales:")
        lines.extend(f"- {path}: {rationale}" for path, rationale in rationales)
    return "\n".join(lines).strip() + "\n"


def _validate_nl_run_config(updated: dict[str, Any], *, base_path: Path) -> None:
    validation = validate_config(
        updated,
        base_path=base_path,
        include_model_validation=True,
    )
    if not validation.valid:
        details = "\n".join(format_validation_messages(validation))
        raise TrendCLIError(f"Config validation failed:\n{details}")


def _confirm_risky_patch(patch: ConfigPatch, *, no_confirm: bool) -> None:
    flags = [flag.value for flag in patch.risk_flags]
    if patch.needs_review:
        flags.append("UNKNOWN_KEYS")
    if not flags or no_confirm:
        return
    flags_text = ", ".join(flags)
    if not sys.stdin.isatty():
        raise TrendCLIError(
            f"Risky changes detected ({flags_text}). Re-run with --no-confirm to apply without prompting."
        )
    response = input(f"Risky changes detected ({flags_text}). Continue? [y/N]: ")
    if response.strip().lower() not in {"y", "yes"}:
        raise TrendCLIError("Update cancelled by user.")
