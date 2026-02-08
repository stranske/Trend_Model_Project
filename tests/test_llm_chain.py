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


class _StructuredOutputLLM(RunnableLambda):
    def __init__(
        self,
        *,
        responses: list[object],
        supports_structured_output: bool = True,
    ) -> None:
        self.invocation_count = 0
        self.structured_requests = 0
        self.structured_invocations = 0
        self._supports_structured_output = supports_structured_output
        self._responses = iter(responses)
        super().__init__(self._respond)

    def supports_structured_output(self) -> bool:
        return self._supports_structured_output

    def with_structured_output(self, _schema) -> RunnableLambda:
        self.structured_requests += 1
        return RunnableLambda(self._respond)

    def _respond(self, _prompt_value, **_kwargs) -> object:
        self.invocation_count += 1
        if self._supports_structured_output:
            self.structured_invocations += 1
        response = next(self._responses)
        if isinstance(response, BaseException):
            raise response
        return response


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
    assert chain.structured_repair_retry_count == 0
    assert llm.structured_requests == 1
    assert llm.invocation_count == 1
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
    assert chain.structured_repair_retry_count == 1
    assert llm.invocation_count > 1
    assert llm.invocation_count <= 2
    assert llm.invocation_count == 2
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


def test_structured_output_invocation_count_zero_when_prompt_builder_fails() -> None:
    def _boom_prompt_builder(**_kwargs) -> str:
        raise ValueError("prompt builder failure")

    llm = _StructuredOutputLLM(responses=[])
    chain = ConfigPatchChain(
        llm=llm,
        prompt_builder=_boom_prompt_builder,
        schema={"type": "object"},
    )

    with pytest.raises(ValueError, match="prompt builder failure"):
        chain.run(
            current_config={"portfolio": {"max_weight": 0.2}},
            instruction="Set max_weight to 0.4.",
        )

    assert llm.invocation_count == 0


def test_structured_output_invocation_count_one_when_llm_raises() -> None:
    llm = _StructuredOutputLLM(responses=[ValueError("structured boom")])
    chain = ConfigPatchChain(
        llm=llm,
        prompt_builder=build_config_patch_prompt,
        schema={"type": "object"},
    )

    with pytest.raises(ValueError, match="structured boom"):
        chain.run(
            current_config={"portfolio": {"max_weight": 0.2}},
            instruction="Set max_weight to 0.4.",
        )

    assert llm.invocation_count == 1


def test_structured_output_invocation_count_three_when_fallback_retries() -> None:
    llm = _StructuredOutputLLM(
        responses=["not json", "still not json", "nope"],
        supports_structured_output=False,
    )
    chain = ConfigPatchChain(
        llm=llm,
        prompt_builder=build_config_patch_prompt,
        schema={"type": "object"},
        retries=2,
    )

    with pytest.raises(ValueError, match="Failed to parse ConfigPatch after 3 attempts"):
        chain.run(
            current_config={"portfolio": {"max_weight": 0.2}},
            instruction="Set max_weight to 0.4.",
        )

    assert llm.invocation_count == 3
