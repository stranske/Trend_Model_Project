"""LangChain wrapper for ConfigPatch generation."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Callable, Iterable

from trend_analysis.config.patch import ConfigPatch
from trend_analysis.llm.prompts import format_config_for_prompt
from trend_analysis.llm.schema import load_compact_schema, select_schema_sections

PromptBuilder = Callable[..., str]


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

        env_temperature = temperature if temperature is not None else _read_env_float(
            "TREND_LLM_TEMPERATURE", default=0.0
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
    ) -> ConfigPatch:
        """Invoke the LLM and parse the ConfigPatch response."""

        prompt_text = self.build_prompt(
            current_config=current_config,
            instruction=instruction,
            allowed_schema=allowed_schema,
            system_prompt=system_prompt,
            safety_rules=safety_rules,
        )
        response_text = self._invoke_llm(prompt_text)
        return self._parse_patch(response_text)

    def _invoke_llm(self, prompt_text: str) -> str:
        from langchain_core.prompts import ChatPromptTemplate

        template = ChatPromptTemplate.from_messages([("system", "{prompt}")])
        chain = template | self._bind_llm()
        response = chain.invoke({"prompt": prompt_text})
        return getattr(response, "content", None) or str(response)

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

    def _parse_patch(self, response_text: str) -> ConfigPatch:
        return ConfigPatch.model_validate_json(_strip_code_fence(response_text))

    def _select_schema(self, *, instruction: str) -> dict[str, Any]:
        schema = self.schema or load_compact_schema()
        return select_schema_sections(schema, instruction)


def _strip_code_fence(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```") and stripped.endswith("```"):
        lines = stripped.splitlines()
        if len(lines) >= 3:
            return "\n".join(lines[1:-1]).strip()
    return stripped


def _read_env_float(name: str, *, default: float) -> float:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a float, got {value!r}.") from exc
