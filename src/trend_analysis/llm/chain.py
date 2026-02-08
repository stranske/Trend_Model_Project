"""LangChain wrapper for ConfigPatch generation."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Callable, Iterable, Iterator, Self
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from trend_analysis.config.patch import (
    ConfigPatch,
    format_retry_error,
    parse_config_patch,
    parse_config_patch_with_retries,
)
from trend_analysis.llm.injection import (
    DEFAULT_BLOCK_SUMMARY,
    detect_prompt_injection_payload,
)
from trend_analysis.llm.nl_logging import NLOperationLog, write_nl_log
from trend_analysis.llm.prompts import (
    build_retry_prompt,
    build_variant_retry_prompt,
    format_config_for_prompt,
)
from trend_analysis.llm.result_validation import (
    detect_unavailable_metric_requests,
    ensure_result_disclaimer,
)
from trend_analysis.llm.schema import load_compact_schema, select_schema_sections
from trend_analysis.llm.validation import (
    flag_unknown_keys,
    normalize_patch_path,
)

if TYPE_CHECKING:
    from trend_analysis.llm.result_metrics import MetricEntry

PromptBuilder = Callable[..., str]

logger = logging.getLogger(__name__)

VARIANT_LABELS = ("conservative", "baseline", "aggressive")


class ConfigPatchVariant(BaseModel):
    label: str = Field(description="Variant label.")
    patch: ConfigPatch = Field(description="ConfigPatch payload for the variant.")

    model_config = ConfigDict(extra="forbid")

    @field_validator("label")
    @classmethod
    def _label_non_empty(cls, value: str) -> str:
        label = value.strip()
        if not label:
            raise ValueError("label must be a non-empty string")
        normalized = label.casefold()
        canonical_map = {variant.casefold(): variant for variant in VARIANT_LABELS}
        if normalized not in canonical_map:
            raise ValueError(
                "label must be one of: " + ", ".join(sorted(canonical_map.values()))
            )
        return canonical_map[normalized]


class ConfigPatchVariants(BaseModel):
    variants: list[ConfigPatchVariant] = Field(description="Variant patch entries.")

    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="after")
    def _validate_variants(self) -> "ConfigPatchVariants":
        if len(self.variants) != 3:
            raise ValueError("variants must contain exactly three entries")
        labels = [variant.label for variant in self.variants]
        normalized = [label.casefold() for label in labels]
        if len(set(normalized)) != len(labels):
            raise ValueError("variants must have unique labels")
        return self


class _LLMResponse(str):
    trace_url: str | None

    def __new__(cls, text: str, trace_url: str | None) -> "_LLMResponse":
        obj = super().__new__(cls, text)
        obj.trace_url = trace_url
        return obj

    def __iter__(self) -> Iterator[str]:
        yield str(self)
        yield self.trace_url or ""


def _default_variant_patches() -> ConfigPatchVariants:
    return ConfigPatchVariants(
        variants=[
            ConfigPatchVariant(
                label=label,
                patch=ConfigPatch(operations=[], summary=DEFAULT_BLOCK_SUMMARY, risk_flags=[]),
            )
            for label in VARIANT_LABELS
        ]
    )


def _strip_code_fence(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```") and stripped.endswith("```"):
        lines = stripped.splitlines()
        if len(lines) >= 3:
            return "\n".join(lines[1:-1]).strip()
    return stripped


def parse_config_patch_variants(response_text: str) -> ConfigPatchVariants:
    return ConfigPatchVariants.model_validate_json(_strip_code_fence(response_text))


def parse_config_patch_variants_with_retries(
    response_provider: Callable[[int, Exception | None], str],
    *,
    retries: int,
    logger: logging.Logger | None = None,
) -> ConfigPatchVariants:
    last_error: Exception | None = None
    total_attempts = max(1, retries)
    active_logger = logger or logging.getLogger(__name__)
    retry_id = f"configpatch-variants-{id(response_provider):x}"
    for attempt in range(total_attempts):
        response_text = response_provider(attempt, last_error)
        try:
            return parse_config_patch_variants(response_text)
        except Exception as exc:
            last_error = exc
            active_logger.error(
                "ConfigPatchVariants parse attempt %s/%s failed (retry_id=%s): %s",
                attempt + 1,
                total_attempts,
                retry_id,
                format_retry_error(exc),
            )
    raise ValueError(
        "Failed to parse ConfigPatchVariants after "
        f"{total_attempts} attempts: {format_retry_error(last_error)}"
    )


@dataclass(slots=True)
class _BaseConfigPatchChain:
    """Container for shared ConfigPatch LangChain pipeline behavior."""

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
    ) -> Self:
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
    ) -> Self:
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

    def _invoke_llm(
        self,
        prompt_text: str,
        *,
        request_id: str | None = None,
        operation: str | None = None,
        llm_override: Any | None = None,
        structured_output: bool = False,
    ) -> _LLMResponse:
        from langchain_core.prompts import ChatPromptTemplate

        from trend_analysis.llm.tracing import (
            langsmith_tracing_context,
            resolve_trace_url,
        )

        template = ChatPromptTemplate.from_messages([("system", "{prompt}")])
        llm = llm_override or self._bind_llm()
        chain = template | llm
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
            if structured_output:
                response_text = self._serialize_structured_response(response)
            else:
                response_text = getattr(response, "content", None) or str(response)
            if run is not None:
                run.end(outputs={"output": response_text})
                trace_url = resolve_trace_url(run)
                if trace_url:
                    logger.info("LangSmith trace: %s", trace_url)
        return _LLMResponse(response_text, trace_url)

    def _bind_llm(self) -> Any:
        return self._bind_llm_with(self.llm)

    def _structured_output_llm(self) -> Any | None:
        return self._structured_output_llm_for(ConfigPatch)

    def _structured_output_llm_for(self, schema: type[BaseModel]) -> Any | None:
        base_llm = self.llm
        supports_attr = getattr(base_llm, "supports_structured_output", None)
        if supports_attr is not None:
            try:
                supports = supports_attr() if callable(supports_attr) else bool(supports_attr)
            except Exception as exc:
                logger.info(
                    "Structured output availability check failed; falling back to text output: %s",
                    exc,
                )
                return None
            if not supports:
                return None
        if not hasattr(base_llm, "with_structured_output"):
            return None
        try:
            structured_llm = base_llm.with_structured_output(schema)
        except Exception as exc:
            logger.info("Structured output unavailable; falling back to text output: %s", exc)
            return None
        if structured_llm is None:
            return None
        return self._bind_llm_with(structured_llm)

    def _bind_llm_with(self, llm: Any) -> Any:
        if not hasattr(llm, "bind"):
            return llm
        params: dict[str, Any] = {"temperature": self.temperature}
        if self.model is not None:
            params["model"] = self.model
        if self.max_tokens is not None:
            params["max_tokens"] = self.max_tokens
        try:
            return llm.bind(**params)
        except TypeError:
            return llm

    def _serialize_structured_response(self, response: Any) -> str:
        if isinstance(response, str):
            return response
        if isinstance(response, ConfigPatch):
            payload = response.model_dump(mode="json")
            return json.dumps(payload, ensure_ascii=True)
        if isinstance(response, dict):
            return json.dumps(response, ensure_ascii=True)
        if hasattr(response, "model_dump"):
            try:
                payload = response.model_dump(mode="json")
            except TypeError:
                payload = response.model_dump()
            return json.dumps(payload, ensure_ascii=True, default=str)
        if hasattr(response, "dict"):
            payload = response.dict()
            return json.dumps(payload, ensure_ascii=True, default=str)
        return str(response)

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

    def _filter_unknown_keys(
        self,
        patch: ConfigPatch,
        unknown_keys: list[Any],
    ) -> None:
        if not unknown_keys:
            return
        unknown_paths = {normalize_patch_path(entry.path) for entry in unknown_keys}

        def _filter_merge_value(
            value: Any,
            base_path: str,
        ) -> Any:
            if not isinstance(value, dict):
                return value
            filtered: dict[str, Any] = {}
            for key, child in value.items():
                if not isinstance(key, str):
                    filtered[key] = child
                    continue
                child_path = f"{base_path}.{key}" if base_path else key
                if child_path in unknown_paths:
                    continue
                filtered_child = _filter_merge_value(child, child_path)
                if isinstance(filtered_child, dict) and not filtered_child:
                    continue
                filtered[key] = filtered_child
            return filtered

        filtered_ops = []
        for operation in patch.operations:
            op_path = normalize_patch_path(operation.path)
            if op_path in unknown_paths:
                continue
            if operation.op == "merge":
                filtered_value = _filter_merge_value(operation.value, op_path)
                if isinstance(filtered_value, dict) and not filtered_value:
                    continue
                operation.value = filtered_value
            filtered_ops.append(operation)
        if len(filtered_ops) != len(patch.operations):
            patch.operations = filtered_ops


@dataclass(slots=True)
class ConfigPatchChain(_BaseConfigPatchChain):
    """Container for the ConfigPatch LangChain pipeline."""

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
        injection_hits = detect_prompt_injection_payload(
            instruction=instruction,
            current_config=current_config,
        )
        try:
            if injection_hits:
                logger.warning(
                    "Prompt injection detected (%s); skipping LLM call.",
                    ", ".join(sorted(set(injection_hits))),
                )
                patch = ConfigPatch(operations=[], summary=DEFAULT_BLOCK_SUMMARY, risk_flags=[])
                return patch
            structured_llm = self._structured_output_llm()
            if structured_llm is not None:
                last_error: Exception | None = None
                total_attempts = 2
                retry_id = f"configpatch-structured-{id(structured_llm):x}"
                for attempt in range(total_attempts):
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
                        llm_override=structured_llm,
                        structured_output=True,
                    )
                    response_text = str(response)
                    trace_url = response.trace_url
                    try:
                        patch = parse_config_patch(response_text)
                        break
                    except Exception as exc:
                        last_error = exc
                        logger.error(
                            "Structured ConfigPatch parse attempt %s/%s failed (retry_id=%s): %s",
                            attempt + 1,
                            total_attempts,
                            retry_id,
                            format_retry_error(exc),
                        )
                        if attempt + 1 >= total_attempts:
                            raise ValueError(
                                "Failed to parse ConfigPatch after "
                                f"{total_attempts} attempts: {format_retry_error(exc)}"
                            ) from exc
            else:

                def _response_provider(attempt: int, last_error: Exception | None) -> str:
                    nonlocal response_text, trace_url
                    prompt = (
                        prompt_text
                        if attempt == 0
                        else build_variant_retry_prompt(
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
            assert patch is not None  # appease mypy; patch is set unless an exception is raised
            schema = self._schema_for_validation(allowed_schema, instruction)
            unknown_keys = flag_unknown_keys(patch, schema, logger=logger)
            self._filter_unknown_keys(patch, unknown_keys)

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


@dataclass(slots=True)
class ConfigPatchVariantsChain(_BaseConfigPatchChain):
    """Container for the variant ConfigPatch LangChain pipeline."""

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
    ) -> ConfigPatchVariants:
        """Invoke the LLM and parse the variant ConfigPatch response."""

        started_at = time.perf_counter()
        timestamp = datetime.now(timezone.utc)
        request_id = request_id or uuid4().hex
        prompt_text = ""
        input_hash = ""
        response_text: str | None = None
        trace_url: str | None = None
        variants: ConfigPatchVariants | None = None
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
        injection_hits = detect_prompt_injection_payload(
            instruction=instruction,
            current_config=current_config,
        )
        try:
            if injection_hits:
                logger.warning(
                    "Prompt injection detected (%s); skipping LLM call.",
                    ", ".join(sorted(set(injection_hits))),
                )
                variants = _default_variant_patches()
                return variants
            structured_llm = self._structured_output_llm_for(ConfigPatchVariants)
            if structured_llm is not None:
                last_error: Exception | None = None
                total_attempts = 2
                retry_id = f"configpatch-variants-structured-{id(structured_llm):x}"
                for attempt in range(total_attempts):
                    prompt = (
                        prompt_text
                        if attempt == 0
                        else build_variant_retry_prompt(
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
                        llm_override=structured_llm,
                        structured_output=True,
                    )
                    response_text = str(response)
                    trace_url = response.trace_url
                    try:
                        variants = parse_config_patch_variants(response_text)
                        break
                    except Exception as exc:
                        last_error = exc
                        logger.error(
                            "Structured ConfigPatchVariants parse attempt %s/%s failed "
                            "(retry_id=%s): %s",
                            attempt + 1,
                            total_attempts,
                            retry_id,
                            format_retry_error(exc),
                        )
                        if attempt + 1 >= total_attempts:
                            raise ValueError(
                                "Failed to parse ConfigPatchVariants after "
                                f"{total_attempts} attempts: {format_retry_error(exc)}"
                            ) from exc
            else:

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

                variants = parse_config_patch_variants_with_retries(
                    _response_provider,
                    retries=max(1, self.retries + 1),
                    logger=logger,
                )
            assert variants is not None
            schema = self._schema_for_validation(allowed_schema, instruction)
            for variant in variants.variants:
                unknown_keys = flag_unknown_keys(variant.patch, schema, logger=logger)
                self._filter_unknown_keys(variant.patch, unknown_keys)
            return variants
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
                    parsed_patch=None,
                    validation_result=None,
                    error=error,
                    duration_ms=elapsed_ms,
                    model_name=self.model or "unknown",
                    temperature=self.temperature,
                    token_usage=None,
                    trace_url=trace_url,
                )
                write_nl_log(entry)


@dataclass(slots=True)
class ResultSummaryResponse:
    text: str
    trace_url: str | None = None


@dataclass(slots=True)
class ResultSummaryChain:
    """Container for the result summary explanation chain."""

    llm: Any
    prompt_builder: PromptBuilder
    temperature: float = 0.0
    model: str | None = None
    max_tokens: int | None = None

    @classmethod
    def from_env(
        cls,
        *,
        llm: Any,
        prompt_builder: PromptBuilder,
        temperature: float | None = None,
        model: str | None = None,
        max_tokens: int | None = None,
    ) -> "ResultSummaryChain":
        env_temperature = (
            temperature
            if temperature is not None
            else _read_env_float("TREND_LLM_TEMPERATURE", default=0.0)
        )
        env_model = model if model is not None else os.environ.get("TREND_LLM_MODEL")
        return cls(
            llm=llm,
            prompt_builder=prompt_builder,
            temperature=env_temperature,
            model=env_model,
            max_tokens=max_tokens,
        )

    def build_prompt(
        self,
        *,
        analysis_output: str,
        metric_catalog: str,
        questions: str,
        system_prompt: str | None = None,
        safety_rules: Iterable[str] | None = None,
    ) -> str:
        return self.prompt_builder(
            analysis_output=analysis_output,
            metric_catalog=metric_catalog,
            questions=questions,
            system_prompt=system_prompt,
            safety_rules=safety_rules,
        )

    def run(
        self,
        *,
        analysis_output: str,
        metric_catalog: str,
        questions: str,
        system_prompt: str | None = None,
        safety_rules: Iterable[str] | None = None,
        request_id: str | None = None,
        metric_entries: Iterable["MetricEntry"] | None = None,
    ) -> ResultSummaryResponse:
        questions_text = questions
        if metric_entries is not None:
            missing_metrics = detect_unavailable_metric_requests(questions, metric_entries)
            if missing_metrics:
                missing_text = ", ".join(missing_metrics)
                response_text = (
                    "Requested data is unavailable in the analysis output for: " f"{missing_text}."
                )
                return ResultSummaryResponse(
                    text=ensure_result_disclaimer(response_text),
                    trace_url=None,
                )
        prompt_text = self.build_prompt(
            analysis_output=analysis_output,
            metric_catalog=metric_catalog,
            questions=questions_text,
            system_prompt=system_prompt,
            safety_rules=safety_rules,
        )
        response = self._invoke_llm(
            prompt_text,
            request_id=request_id,
            operation="result_explain",
        )
        return ResultSummaryResponse(
            text=ensure_result_disclaimer(str(response)),
            trace_url=response.trace_url,
        )

    def _invoke_llm(
        self,
        prompt_text: str,
        *,
        request_id: str | None = None,
        operation: str | None = None,
    ) -> _LLMResponse:
        from langchain_core.prompts import ChatPromptTemplate

        from trend_analysis.llm.tracing import (
            langsmith_tracing_context,
            resolve_trace_url,
        )

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
                trace_url = resolve_trace_url(run)
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
