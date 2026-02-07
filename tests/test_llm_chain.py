"""Tests for ConfigPatchChain structured-output fallback behavior."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("langchain_core")

from langchain_core.runnables import RunnableLambda  # noqa: E402

from trend_analysis.config.patch import ConfigPatch  # noqa: E402
from trend_analysis.llm.chain import ConfigPatchChain  # noqa: E402
from trend_analysis.llm.prompts import build_config_patch_prompt  # noqa: E402


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


class _NoStructuredOutputLLMRaises(RunnableLambda):
    def __init__(self, *, responses: list[str]) -> None:
        self.unstructured_calls = 0
        self._responses = iter(responses)
        super().__init__(self._respond)

    def _respond(self, _prompt_value, **_kwargs) -> str:
        self.unstructured_calls += 1
        return next(self._responses)

    def supports_structured_output(self) -> bool:
        return False

    def with_structured_output(self, _schema) -> RunnableLambda:
        raise AssertionError("Structured output should not be requested.")


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


def _make_payload() -> dict[str, object]:
    return {
        "operations": [
            {
                "op": "set",
                "path": "portfolio.max_weight",
                "value": 0.4,
                "rationale": "Align with instruction",
            }
        ],
        "risk_flags": [],
        "summary": "Update max_weight",
    }


def _load_structured_fixture() -> dict[str, object]:
    fixture_path = Path(__file__).resolve().parent / "fixtures" / "config_patch_structured.json"
    return json.loads(fixture_path.read_text(encoding="utf-8"))


def test_structured_output_returns_expected_configpatch() -> None:
    payload = _load_structured_fixture()
    llm = _StructuredOutputLLM(responses=[payload])
    chain = ConfigPatchChain(
        llm=llm,
        prompt_builder=build_config_patch_prompt,
        schema={"type": "object"},
    )

    patch = chain.run(
        current_config={"portfolio": {"max_weight": 0.2}},
        instruction="Set max_weight to 0.4.",
    )

    assert isinstance(patch, ConfigPatch)
    assert set(patch.model_dump(mode="json").keys()) == set(payload.keys())
    validated = ConfigPatch.model_validate(patch.model_dump(mode="json"))
    assert validated.model_dump(mode="json") == payload
    assert patch.model_dump(mode="json") == payload
    assert llm.structured_requests == 1
    assert llm.structured_invocations == 1


def test_structured_output_retries_once_on_malformed_response() -> None:
    payload = _load_structured_fixture()
    malformed = {"operations": []}
    llm = _StructuredOutputLLM(responses=[malformed, payload])
    chain = ConfigPatchChain(
        llm=llm,
        prompt_builder=build_config_patch_prompt,
        schema={"type": "object"},
    )

    patch = chain.run(
        current_config={"portfolio": {"max_weight": 0.2}},
        instruction="Set max_weight to 0.4.",
    )

    assert patch.model_dump(mode="json") == payload
    retry_count = llm.structured_invocations - 1
    assert retry_count == 1
    assert llm.structured_invocations == 2


def test_fallback_when_structured_unsupported() -> None:
    payload = _make_payload()
    llm = _NoStructuredOutputLLM(responses=[json.dumps(payload)])
    chain = ConfigPatchChain(
        llm=llm,
        prompt_builder=build_config_patch_prompt,
        schema={"type": "object"},
    )

    patch = chain.run(
        current_config={"portfolio": {"max_weight": 0.2}},
        instruction="Set max_weight to 0.4.",
    )

    assert isinstance(patch, ConfigPatch)
    assert patch.summary == "Update max_weight"
    assert patch.operations[0].path == "portfolio.max_weight"
    assert llm.unstructured_calls == 1
    assert llm.structured_calls == 0


def test_fallback_produces_valid_configpatch() -> None:
    payload = _make_payload()
    llm = _NoStructuredOutputLLM(responses=[json.dumps(payload)])
    chain = ConfigPatchChain(
        llm=llm,
        prompt_builder=build_config_patch_prompt,
        schema={"type": "object"},
    )

    patch = chain.run(
        current_config={"portfolio": {"max_weight": 0.2}},
        instruction="Set max_weight to 0.4.",
    )

    validated = ConfigPatch.model_validate(patch.model_dump(mode="json"))
    assert validated.summary == "Update max_weight"
    assert patch.model_dump(mode="json") == {
        "operations": [
            {
                "op": "set",
                "path": "portfolio.max_weight",
                "value": 0.4,
                "rationale": "Align with instruction",
            }
        ],
        "needs_review": False,
        "risk_flags": [],
        "summary": "Update max_weight",
    }


def test_no_structured_request_when_unsupported() -> None:
    payload = _make_payload()
    llm = _NoStructuredOutputLLMRaises(responses=[json.dumps(payload)])
    chain = ConfigPatchChain(
        llm=llm,
        prompt_builder=build_config_patch_prompt,
        schema={"type": "object"},
    )

    patch = chain.run(
        current_config={"portfolio": {"max_weight": 0.2}},
        instruction="Set max_weight to 0.4.",
    )

    assert patch.summary == "Update max_weight"
    assert llm.unstructured_calls == 1
