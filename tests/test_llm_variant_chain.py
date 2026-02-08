"""Tests for ConfigPatchVariantsChain structured-output behavior."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

pytest.importorskip("langchain_core")

from langchain_core.runnables import RunnableLambda  # noqa: E402

from trend_analysis.llm.chain import (  # noqa: E402
    ConfigPatchVariants,
    ConfigPatchVariantsChain,
)
from trend_analysis.llm.prompts import build_variant_patch_prompt  # noqa: E402


class _NoStructuredOutputLLM(RunnableLambda):
    def __init__(self, *, responses: list[str]) -> None:
        self.unstructured_calls = 0
        self.structured_calls = 0
        self._responses = iter(responses)
        super().__init__(self._respond)

    def _respond(self, _prompt_value, **_kwargs) -> str:
        self.unstructured_calls += 1
        return next(self._responses)

    def supports_structured_output(self) -> bool:
        return False

    def with_structured_output(self, _schema) -> RunnableLambda:
        self.structured_calls += 1
        return RunnableLambda(self._respond)


class _StructuredOutputLLM:
    def __init__(self, *, responses: list[object]) -> None:
        self.structured_requests = 0
        self.structured_invocations = 0
        self._responses = iter(responses)

    def supports_structured_output(self) -> bool:
        return True

    def with_structured_output(self, _schema) -> RunnableLambda:
        self.structured_requests += 1
        return RunnableLambda(self._respond)

    def _respond(self, _prompt_value, **_kwargs) -> object:
        self.structured_invocations += 1
        return next(self._responses)


def _make_patch(summary: str, value: float) -> dict[str, object]:
    return {
        "operations": [
            {
                "op": "set",
                "path": "portfolio.max_weight",
                "value": value,
                "rationale": "Align with instruction",
            }
        ],
        "risk_flags": [],
        "summary": summary,
    }


def _make_payload() -> dict[str, object]:
    return {
        "variants": [
            {"label": "conservative", "patch": _make_patch("Conservative tweak", 0.3)},
            {"label": "baseline", "patch": _make_patch("Baseline tweak", 0.4)},
            {"label": "aggressive", "patch": _make_patch("Aggressive tweak", 0.5)},
        ]
    }


def test_variant_structured_output_returns_expected_payload() -> None:
    payload = _make_payload()
    llm = _StructuredOutputLLM(responses=[payload])
    chain = ConfigPatchVariantsChain(
        llm=llm,
        prompt_builder=build_variant_patch_prompt,
        schema={"type": "object"},
    )

    variants = chain.run(
        current_config={"portfolio": {"max_weight": 0.2}},
        instruction="Generate three variants with different max_weight.",
    )

    assert [variant.label for variant in variants.variants] == [
        "conservative",
        "baseline",
        "aggressive",
    ]
    assert llm.structured_requests == 1
    assert llm.structured_invocations == 1


def test_variant_fallback_when_structured_unsupported() -> None:
    payload = _make_payload()
    llm = _NoStructuredOutputLLM(responses=[json.dumps(payload)])
    chain = ConfigPatchVariantsChain(
        llm=llm,
        prompt_builder=build_variant_patch_prompt,
        schema={"type": "object"},
    )

    variants = chain.run(
        current_config={"portfolio": {"max_weight": 0.2}},
        instruction="Generate three variants with different max_weight.",
    )

    assert len(variants.variants) == 3
    assert variants.variants[0].patch.summary == "Conservative tweak"
    assert llm.unstructured_calls == 1
    assert llm.structured_calls == 0


def test_variant_labels_are_validated() -> None:
    payload = _make_payload()
    payload["variants"][0]["label"] = "steady"

    with pytest.raises(ValidationError):
        ConfigPatchVariants.model_validate(payload)
