"""Tests for ConfigPatchChain."""

from __future__ import annotations

import json
import logging
from contextlib import contextmanager

import pytest

pytest.importorskip("langchain_core")

import jsonschema
from langchain_core.runnables import RunnableLambda

from trend_analysis.config.patch import ConfigPatch
from trend_analysis.llm.chain import ConfigPatchChain
from trend_analysis.llm.prompts import build_config_patch_prompt


def test_config_patch_chain_run_parses_patch() -> None:
    captured: dict[str, str] = {}

    def _respond(prompt_value, **_kwargs) -> str:
        messages = prompt_value.to_messages()
        captured["prompt"] = messages[0].content
        payload = {
            "operations": [
                {
                    "op": "set",
                    "path": "portfolio.max_weight",
                    "value": 0.3,
                    "rationale": "Align with instruction",
                }
            ],
            "risk_flags": [],
            "summary": "Update max_weight",
        }
        return json.dumps(payload)

    llm = RunnableLambda(_respond)
    chain = ConfigPatchChain(
        llm=llm,
        prompt_builder=build_config_patch_prompt,
        schema={"type": "object"},
    )

    patch = chain.run(
        current_config={"portfolio": {"max_weight": 0.2}},
        instruction="Set max_weight to 0.3.",
    )

    assert patch.operations[0].path == "portfolio.max_weight"
    assert patch.summary == "Update max_weight"
    assert "USER INSTRUCTION" in captured["prompt"]


def test_config_patch_chain_batch_schema_conformance() -> None:
    total_cases = 100
    valid_cases = 96
    responses: list[str] = []
    for idx in range(total_cases):
        if idx < valid_cases:
            payload = {
                "operations": [
                    {
                        "op": "set",
                        "path": "portfolio.max_weight",
                        "value": round(0.2 + idx * 0.001, 3),
                        "rationale": "Align with instruction",
                    }
                ],
                "risk_flags": [],
                "summary": f"Update max_weight {idx}",
            }
            responses.append(json.dumps(payload))
        else:
            responses.append(json.dumps({"operations": [], "risk_flags": []}))

    response_iter = iter(responses)

    def _respond(prompt_value, **_kwargs) -> str:
        _ = prompt_value.to_messages()
        return next(response_iter)

    llm = RunnableLambda(_respond)
    chain = ConfigPatchChain(
        llm=llm,
        prompt_builder=build_config_patch_prompt,
        schema={"type": "object"},
    )

    schema = ConfigPatch.json_schema()
    validator = jsonschema.Draft202012Validator(schema)
    valid_count = 0
    for idx in range(total_cases):
        prompt = chain.build_prompt(
            current_config={"portfolio": {"max_weight": 0.2}},
            instruction=f"Set max_weight to {0.2 + idx * 0.001:.3f}.",
        )
        response_text, _trace_url = chain._invoke_llm(prompt)
        try:
            payload = json.loads(response_text)
        except json.JSONDecodeError:
            continue
        if validator.is_valid(payload):
            valid_count += 1

    assert valid_count / total_cases >= 0.95


def test_config_patch_chain_flags_unknown_keys() -> None:
    def _respond(_prompt_value, **_kwargs) -> str:
        payload = {
            "operations": [
                {
                    "op": "set",
                    "path": "analysis.turbo_mode",
                    "value": True,
                }
            ],
            "risk_flags": [],
            "summary": "Enable turbo mode.",
        }
        return json.dumps(payload)

    llm = RunnableLambda(_respond)
    chain = ConfigPatchChain(
        llm=llm,
        prompt_builder=build_config_patch_prompt,
        schema={
            "type": "object",
            "properties": {
                "analysis": {"type": "object", "properties": {"top_n": {"type": "integer"}}}
            },
        },
    )

    patch = chain.run(
        current_config={"analysis": {"top_n": 8}},
        instruction="Enable turbo mode.",
    )

    assert patch.needs_review is True


def test_config_patch_chain_unknown_keys_raise() -> None:
    def _respond(prompt_value, **_kwargs) -> str:
        _ = prompt_value.to_messages()
        payload = {
            "operations": [],
            "risk_flags": [],
            "summary": "No changes",
            "extra_field": "unexpected",
        }
        return json.dumps(payload)

    llm = RunnableLambda(_respond)
    chain = ConfigPatchChain(
        llm=llm,
        prompt_builder=build_config_patch_prompt,
        schema={"type": "object"},
    )

    with pytest.raises(ValueError) as excinfo:
        chain.run(
            current_config={"portfolio": {"max_weight": 0.2}},
            instruction="Do nothing.",
        )
    message = str(excinfo.value)
    assert "Failed to parse ConfigPatch after 2 attempts" in message
    assert "ValidationError" in message


def test_config_patch_chain_omits_unknown_key_instruction() -> None:
    def _respond(prompt_value, **_kwargs) -> str:
        _ = prompt_value.to_messages()
        payload = {
            "operations": [],
            "risk_flags": [],
            "summary": "Unknown key portfolio.unknown_setting; no changes applied.",
        }
        return json.dumps(payload)

    llm = RunnableLambda(_respond)
    chain = ConfigPatchChain(
        llm=llm,
        prompt_builder=build_config_patch_prompt,
        schema={"type": "object"},
    )

    patch = chain.run(
        current_config={"portfolio": {"max_weight": 0.2}},
        instruction="Set portfolio.unknown_setting to 0.5.",
    )

    assert patch.operations == []
    assert "unknown key" in patch.summary.lower()


def test_config_patch_chain_retries_on_parse_error() -> None:
    prompts: list[str] = []
    responses = iter(
        [
            "not-json",
            json.dumps(
                {
                    "operations": [
                        {
                            "op": "set",
                            "path": "portfolio.max_weight",
                            "value": 0.25,
                        }
                    ],
                    "risk_flags": [],
                    "summary": "Update max_weight",
                }
            ),
        ]
    )

    def _respond(prompt_value, **_kwargs) -> str:
        messages = prompt_value.to_messages()
        prompts.append(messages[0].content)
        return next(responses)

    llm = RunnableLambda(_respond)
    chain = ConfigPatchChain(
        llm=llm,
        prompt_builder=build_config_patch_prompt,
        schema={"type": "object"},
        retries=1,
    )

    patch = chain.run(
        current_config={"portfolio": {"max_weight": 0.2}},
        instruction="Set max_weight to 0.25.",
    )

    assert patch.summary == "Update max_weight"
    assert len(prompts) == 2
    assert "PREVIOUS ERROR" in prompts[1]
    assert "ValidationError" in prompts[1]


def test_config_patch_chain_retries_on_validation_error() -> None:
    prompts: list[str] = []
    responses = iter(
        [
            json.dumps({"operations": [], "risk_flags": []}),
            json.dumps(
                {
                    "operations": [
                        {
                            "op": "set",
                            "path": "portfolio.max_weight",
                            "value": 0.25,
                        }
                    ],
                    "risk_flags": [],
                    "summary": "Update max_weight",
                }
            ),
        ]
    )

    def _respond(prompt_value, **_kwargs) -> str:
        messages = prompt_value.to_messages()
        prompts.append(messages[0].content)
        return next(responses)

    llm = RunnableLambda(_respond)
    chain = ConfigPatchChain(
        llm=llm,
        prompt_builder=build_config_patch_prompt,
        schema={"type": "object"},
        retries=1,
    )

    patch = chain.run(
        current_config={"portfolio": {"max_weight": 0.2}},
        instruction="Set max_weight to 0.25.",
    )

    assert patch.summary == "Update max_weight"
    assert len(prompts) == 2
    assert "PREVIOUS ERROR" in prompts[1]
    assert "ValidationError" in prompts[1]


def test_config_patch_chain_retry_exhausted_raises_error() -> None:
    def _respond(_prompt_value, **_kwargs) -> str:
        return "not-json"

    llm = RunnableLambda(_respond)
    chain = ConfigPatchChain(
        llm=llm,
        prompt_builder=build_config_patch_prompt,
        schema={"type": "object"},
        retries=1,
    )

    with pytest.raises(ValueError) as excinfo:
        chain.run(
            current_config={"portfolio": {"max_weight": 0.2}},
            instruction="Set max_weight to 0.25.",
        )

    message = str(excinfo.value)
    assert "Failed to parse ConfigPatch after 2 attempts" in message
    assert "ValidationError" in message


def test_config_patch_chain_logs_parse_errors(caplog: pytest.LogCaptureFixture) -> None:
    responses = iter(
        [
            "not-json",
            "still-not-json",
            json.dumps(
                {
                    "operations": [
                        {
                            "op": "set",
                            "path": "portfolio.max_weight",
                            "value": 0.25,
                        }
                    ],
                    "risk_flags": [],
                    "summary": "Update max_weight",
                }
            ),
        ]
    )

    def _respond(_prompt_value, **_kwargs) -> str:
        return next(responses)

    llm = RunnableLambda(_respond)
    chain = ConfigPatchChain(
        llm=llm,
        prompt_builder=build_config_patch_prompt,
        schema={"type": "object"},
        retries=2,
    )

    with caplog.at_level(logging.WARNING, logger="trend_analysis.llm.chain"):
        patch = chain.run(
            current_config={"portfolio": {"max_weight": 0.2}},
            instruction="Set max_weight to 0.25.",
        )

    assert patch.summary == "Update max_weight"
    parse_logs = [
        record for record in caplog.records if "ConfigPatch parse attempt" in record.message
    ]
    assert len(parse_logs) == 2
    assert "attempt 1/3" in parse_logs[0].message
    assert "ValidationError" in parse_logs[0].message
    assert "attempt 2/3" in parse_logs[1].message
    assert "ValidationError" in parse_logs[1].message


def test_config_patch_chain_logs_validation_errors(
    caplog: pytest.LogCaptureFixture,
) -> None:
    responses = iter(
        [
            json.dumps({"operations": [], "risk_flags": []}),
            json.dumps({"operations": [], "risk_flags": []}),
            json.dumps(
                {
                    "operations": [
                        {
                            "op": "set",
                            "path": "portfolio.max_weight",
                            "value": 0.25,
                        }
                    ],
                    "risk_flags": [],
                    "summary": "Update max_weight",
                }
            ),
        ]
    )

    def _respond(_prompt_value, **_kwargs) -> str:
        return next(responses)

    llm = RunnableLambda(_respond)
    chain = ConfigPatchChain(
        llm=llm,
        prompt_builder=build_config_patch_prompt,
        schema={"type": "object"},
        retries=2,
    )

    with caplog.at_level(logging.WARNING, logger="trend_analysis.llm.chain"):
        patch = chain.run(
            current_config={"portfolio": {"max_weight": 0.2}},
            instruction="Set max_weight to 0.25.",
        )

    assert patch.summary == "Update max_weight"
    parse_logs = [
        record for record in caplog.records if "ConfigPatch parse attempt" in record.message
    ]
    assert len(parse_logs) == 2
    assert "attempt 2/3" in parse_logs[-1].message
    assert "ValidationError" in parse_logs[-1].message


def test_config_patch_chain_logs_parse_errors_on_exhaustion(
    caplog: pytest.LogCaptureFixture,
) -> None:
    responses = iter(["not-json", "still-not-json", "bad-json"])

    def _respond(_prompt_value, **_kwargs) -> str:
        return next(responses)

    llm = RunnableLambda(_respond)
    chain = ConfigPatchChain(
        llm=llm,
        prompt_builder=build_config_patch_prompt,
        schema={"type": "object"},
        retries=2,
    )

    with caplog.at_level(logging.WARNING, logger="trend_analysis.llm.chain"):
        with pytest.raises(ValueError) as excinfo:
            chain.run(
                current_config={"portfolio": {"max_weight": 0.2}},
                instruction="Set max_weight to 0.25.",
            )

    parse_logs = [
        record for record in caplog.records if "ConfigPatch parse attempt" in record.message
    ]
    assert len(parse_logs) == 3
    assert "attempt 3/3" in parse_logs[-1].message
    assert "Failed to parse ConfigPatch after 3 attempts" in str(excinfo.value)
    assert "ValidationError" in str(excinfo.value)


def test_config_patch_chain_no_retry_when_retries_zero(
    caplog: pytest.LogCaptureFixture,
) -> None:
    responses = iter(["not-json"])

    def _respond(_prompt_value, **_kwargs) -> str:
        return next(responses)

    llm = RunnableLambda(_respond)
    chain = ConfigPatchChain(
        llm=llm,
        prompt_builder=build_config_patch_prompt,
        schema={"type": "object"},
        retries=0,
    )

    with caplog.at_level(logging.WARNING, logger="trend_analysis.llm.chain"):
        with pytest.raises(ValueError) as excinfo:
            chain.run(
                current_config={"portfolio": {"max_weight": 0.2}},
                instruction="Set max_weight to 0.25.",
            )

    parse_logs = [
        record for record in caplog.records if "ConfigPatch parse attempt" in record.message
    ]
    assert len(parse_logs) == 1
    assert "attempt 1/1" in parse_logs[0].message
    assert "Failed to parse ConfigPatch after 1 attempts" in str(excinfo.value)


def test_config_patch_chain_logs_langsmith_trace_url(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    def _respond(_prompt_value, **_kwargs) -> str:
        payload = {
            "operations": [
                {
                    "op": "set",
                    "path": "portfolio.max_weight",
                    "value": 0.3,
                    "rationale": "Align with instruction",
                }
            ],
            "risk_flags": [],
            "summary": "Update max_weight",
        }
        return json.dumps(payload)

    class FakeRun:
        url = "https://example.test/trace/123"

        def end(self, outputs: dict[str, str]) -> None:
            self.outputs = outputs

    @contextmanager
    def _fake_context(**_kwargs):
        yield FakeRun()

    monkeypatch.setattr(
        "trend_analysis.llm.tracing.langsmith_tracing_context",
        _fake_context,
    )
    llm = RunnableLambda(_respond)
    chain = ConfigPatchChain(
        llm=llm,
        prompt_builder=build_config_patch_prompt,
        schema={"type": "object"},
    )

    with caplog.at_level(logging.INFO, logger="trend_analysis.llm.chain"):
        _, trace_url = chain._invoke_llm("Prompt")

    assert "LangSmith trace: https://example.test/trace/123" in caplog.text
    assert trace_url == "https://example.test/trace/123"
