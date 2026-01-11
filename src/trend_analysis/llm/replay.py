"""Replay helpers for NL operation logs."""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from trend_analysis.llm.nl_logging import NLOperationLog
from trend_analysis.llm.providers import LLMProviderConfig, create_llm
from trend_analysis.logging import iter_jsonl


@dataclass(frozen=True)
class ReplayResult:
    prompt: str
    prompt_hash: str
    output: str
    output_hash: str
    recorded_output: str | None
    recorded_hash: str | None
    matches: bool


def load_nl_log_entry(path: Path, entry: int) -> NLOperationLog:
    if entry < 1:
        raise ValueError("entry index must be >= 1")
    for index, payload in enumerate(iter_jsonl(path), start=1):
        if index == entry:
            return NLOperationLog.model_validate(payload)
    raise IndexError(f"Entry {entry} not found in {path}")


def render_prompt(entry: NLOperationLog) -> str:
    template = entry.prompt_template or ""
    variables = entry.prompt_variables or {}
    if not variables:
        return template
    try:
        return template.format(**variables)
    except Exception:
        return template


def replay_nl_entry(
    entry: NLOperationLog,
    *,
    llm: Any | None = None,
    provider: str | None = None,
    model: str | None = None,
    temperature: float | None = None,
) -> ReplayResult:
    prompt_text = render_prompt(entry)
    active_llm = llm or _create_llm_from_env(entry, provider=provider, model=model)
    active_model = model or entry.model_name
    active_temperature = entry.temperature if temperature is None else float(temperature)
    output_text = _invoke_llm(
        prompt_text,
        active_llm,
        temperature=active_temperature,
        model=active_model,
    )
    recorded = entry.model_output
    output_hash = _hash_text(output_text)
    recorded_hash = _hash_text(recorded) if recorded is not None else None
    matches = recorded == output_text if recorded is not None else False
    return ReplayResult(
        prompt=prompt_text,
        prompt_hash=_hash_text(prompt_text),
        output=output_text,
        output_hash=output_hash,
        recorded_output=recorded,
        recorded_hash=recorded_hash,
        matches=matches,
    )


def _create_llm_from_env(
    entry: NLOperationLog,
    *,
    provider: str | None = None,
    model: str | None = None,
) -> Any:
    provider_name = provider or os.environ.get("TREND_LLM_PROVIDER", "openai")
    model_name = model or entry.model_name or os.environ.get("TREND_LLM_MODEL", "gpt-4o-mini")
    config = LLMProviderConfig(provider=provider_name, model=model_name)
    return create_llm(config)


def _invoke_llm(prompt_text: str, llm: Any, *, temperature: float, model: str | None) -> str:
    from langchain_core.prompts import ChatPromptTemplate

    if hasattr(llm, "bind"):
        params: dict[str, Any] = {"temperature": temperature}
        if model is not None:
            params["model"] = model
        try:
            llm = llm.bind(**params)
        except TypeError:
            pass
    template = ChatPromptTemplate.from_messages([("system", "{prompt}")])
    try:
        response = (template | llm).invoke({"prompt": prompt_text})
    except TypeError:
        if hasattr(llm, "invoke"):
            response = llm.invoke(prompt_text)
        else:
            response = llm(prompt_text)
    return getattr(response, "content", None) or str(response)


def _hash_text(text: str | None) -> str:
    payload = text or ""
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


__all__ = ["ReplayResult", "load_nl_log_entry", "render_prompt", "replay_nl_entry"]
