"""LangChain wrapper for ConfigPatch generation."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Iterable, Iterator
from uuid import uuid4

from trend_analysis.config.patch import (
    ConfigPatch,
    format_retry_error,
    parse_config_patch_with_retries,
)
from trend_analysis.llm.injection import DEFAULT_BLOCK_SUMMARY, detect_prompt_injection
from trend_analysis.llm.nl_logging import NLOperationLog, write_nl_log
from trend_analysis.llm.prompts import build_retry_prompt, format_config_for_prompt
from trend_analysis.llm.schema import load_compact_schema, select_schema_sections
from trend_analysis.llm.validation import flag_unknown_keys

PromptBuilder = Callable[..., str]

logger = logging.getLogger(__name__)


class _LLMResponse(str):
    trace_url: str | None

    def __new__(cls, text: str, trace_url: str | None) -> "_LLMResponse":
        obj = super().__new__(cls, text)
        obj.trace_url = trace_url
        return obj

    def __iter__(self) -> Iterator[str]:
        yield str(self)
        yield self.trace_url or ""


@dataclass(slots=True)
class ConfigPatchChain:
    """Container for the ConfigPatch LangChain pipeline."""

    llm: Any
    prompt_builder: PromptBuilder
    schema: dict[str, Any] | None
    temperature: float = 0.0
    model: str | None = None
    max_tokens: int | None = None
    retries: int = 1

    @classmethod
    def from_defaults(
        cls,
        *,
        llm: Any,
        schema: dict[str, Any] | None = None,
        prompt_builder: PromptBuilder,
        temperature: float = 0.0,
        model: str | None = None,
        max_tokens: int | None = None,
        retries: int = 1,
    ) -> "ConfigPatchChain":
        """Build a chain with standard prompt builder + schema."""

        return cls(
            llm=llm,
            prompt_builder=prompt_builder,
            schema=schema,
            temperature=temperature,
            model=model,
            max_tokens=max_tokens,
            retries=retries,
        )

    @classmethod
    def from_env(
        cls,
        *,
        llm: Any,
        schema: dict[str, Any] | None = None,
        prompt_builder: PromptBuilder,
        temperature: float | None = None,
        model: str | None = None,
        max_tokens: int | None = None,
        retries: int = 1,
    ) -> "ConfigPatchChain":
        """Build a chain using environment overrides for model/temperature."""

        env_temperature = (
            temperature
            if temperature is not None
            else _read_env_float("TREND_LLM_TEMPERATURE", default=0.0)
        )
        env_model = model if model is not None else os.environ.get("TREND_LLM_MODEL")
        return cls(
            llm=llm,
            prompt_builder=prompt_builder,
            schema=schema,
            temperature=env_temperature,
            model=env_model,
            max_tokens=max_tokens,
            retries=retries,
        )

    def build_prompt(
        self,
        *,
        current_config: str | dict[str, Any],
        instruction: str,
        allowed_schema: str | None = None,
        system_prompt: str | None = None,
        safety_rules: Iterable[str] | None = None,
    ) -> str:
        """Render the ConfigPatch prompt text."""

        config_text = (
            current_config
            if isinstance(current_config, str)
            else format_config_for_prompt(current_config)
        )
        schema_text = allowed_schema or self._serialize_schema(
            self._select_schema(instruction=instruction)
        )
        return self.prompt_builder(
            current_config=config_text,
            allowed_schema=schema_text,
            instruction=instruction,
            system_prompt=system_prompt,
            safety_rules=safety_rules,
        )

    def run(
        self,
        *,
        current_config: str | dict[str, Any],
        instruction: str,
        allowed_schema: str | None = None,
        system_prompt: str | None = None,
        safety_rules: Iterable[str] | None = None,
        request_id: str | None = None,
        log_operation: bool = True,
    ) -> ConfigPatch:
        """Invoke the LLM and parse the ConfigPatch response."""

        started_at = time.perf_counter()
        timestamp = datetime.now(timezone.utc)
        request_id = request_id or uuid4().hex
        prompt_text = ""
        input_hash = ""
        response_text: str | None = None
        trace_url: str | None = None
        patch: ConfigPatch | None = None
        error: str | None = None

        config_text = (
            current_config
            if isinstance(current_config, str)
            else format_config_for_prompt(current_config)
        )
        schema_text = allowed_schema or self._serialize_schema(
            self._select_schema(instruction=instruction)
        )
        prompt_text = self.prompt_builder(
            current_config=config_text,
            allowed_schema=schema_text,
            instruction=instruction,
            system_prompt=system_prompt,
            safety_rules=safety_rules,
        )
        input_hash = _hash_payload(
            {
                "prompt": prompt_text,
                "model": self.model,
                "temperature": self.temperature,
            }
        )
        injection_hits = detect_prompt_injection(instruction)
        try:
            if injection_hits:
                logger.warning(
                    "Prompt injection detected (%s); skipping LLM call.",
                    ", ".join(sorted(set(injection_hits))),
                )
                patch = ConfigPatch(operations=[], summary=DEFAULT_BLOCK_SUMMARY, risk_flags=[])
                return patch

            def _response_provider(attempt: int, last_error: Exception | None) -> str:
                nonlocal response_text, trace_url
                prompt = (
                    prompt_text
                    if attempt == 0
                    else build_retry_prompt(
                        current_config=config_text,
                        allowed_schema=schema_text,
                        instruction=instruction,
                        error_message=format_retry_error(last_error),
                        system_prompt=system_prompt,
                        safety_rules=safety_rules,
                    )
                )
                response = self._invoke_llm(
                    prompt,
                    request_id=request_id,
                    operation="nl_to_patch",
                )
                response_text = str(response)
                trace_url = response.trace_url
                return response_text

            patch = parse_config_patch_with_retries(
                _response_provider,
                retries=max(1, self.retries + 1),
                logger=logger,
            )
            schema = self._schema_for_validation(allowed_schema, instruction)
            flag_unknown_keys(patch, schema, logger=logger)
            return patch
        except Exception as exc:
            error = str(exc) or type(exc).__name__
            raise
        finally:
            if log_operation:
                elapsed_ms = (time.perf_counter() - started_at) * 1000
                entry = NLOperationLog(
                    request_id=request_id,
                    timestamp=timestamp,
                    operation="nl_to_patch",
                    input_hash=input_hash,
                    prompt_template=prompt_text,
                    prompt_variables={},
                    model_output=response_text,
                    parsed_patch=patch,
                    validation_result=None,
                    error=error,
                    duration_ms=elapsed_ms,
                    model_name=self.model or "unknown",
                    temperature=self.temperature,
                    token_usage=None,
                    trace_url=trace_url,
                )
                write_nl_log(entry)

    def _invoke_llm(
        self,
        prompt_text: str,
        *,
        request_id: str | None = None,
        operation: str | None = None,
    ) -> _LLMResponse:
        from langchain_core.prompts import ChatPromptTemplate

        from trend_analysis.llm.tracing import langsmith_tracing_context

        template = ChatPromptTemplate.from_messages([("system", "{prompt}")])
        chain = template | self._bind_llm()
        metadata = {
            "request_id": request_id,
            "operation": operation or "nl_operation",
            "model": self.model,
            "temperature": self.temperature,
        }
        trace_url: str | None = None
        with langsmith_tracing_context(
            name=operation or "nl_operation",
            run_type="chain",
            inputs={"prompt": prompt_text},
            metadata=metadata,
        ) as run:
            response = chain.invoke({"prompt": prompt_text})
            response_text = getattr(response, "content", None) or str(response)
            if run is not None:
                run.end(outputs={"output": response_text})
                trace_url = getattr(run, "url", None)
                if trace_url:
                    logger.info("LangSmith trace: %s", trace_url)
        return _LLMResponse(response_text, trace_url)

    def _bind_llm(self) -> Any:
        if not hasattr(self.llm, "bind"):
            return self.llm
        params: dict[str, Any] = {"temperature": self.temperature}
        if self.model is not None:
            params["model"] = self.model
        if self.max_tokens is not None:
            params["max_tokens"] = self.max_tokens
        try:
            return self.llm.bind(**params)
        except TypeError:
            return self.llm

    def _serialize_schema(self, schema: dict[str, Any]) -> str:
        return json.dumps(schema, indent=2, ensure_ascii=True)

    def _schema_for_validation(
        self,
        allowed_schema: str | None,
        instruction: str,
    ) -> dict[str, Any] | None:
        if allowed_schema:
            try:
                return json.loads(allowed_schema)  # type: ignore[no-any-return]
            except json.JSONDecodeError:
                return None
        return self._select_schema(instruction=instruction)

    def _select_schema(self, *, instruction: str) -> dict[str, Any]:
        schema = self.schema or load_compact_schema()
        return select_schema_sections(schema, instruction)


def _read_env_float(name: str, *, default: float) -> float:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a float, got {value!r}.") from exc


def _hash_payload(payload: dict[str, Any]) -> str:
    text = json.dumps(payload, sort_keys=True, ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
