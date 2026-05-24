"""Tests for ConfigPatchChain environment settings."""

from __future__ import annotations

import json

from trend_analysis.llm.chain import ConfigPatchChain
from trend_analysis.llm.prompts import build_config_patch_prompt


class DummyLLM:
    def __init__(self) -> None:
        self.bound: dict[str, object] | None = None

    def bind(self, **kwargs):
        self.bound = kwargs
        return self


class InvokingLLM(DummyLLM):
    def __call__(self, payload):
        return self.invoke(payload)

    def invoke(self, _payload):
        return json.dumps(
            {
                "operations": [
                    {"op": "set", "path": "analysis.top_n", "value": 12},
                ],
                "summary": "Update top_n.",
                "risk_flags": [],
            }
        )


def test_chain_from_env_uses_temperature_and_model(monkeypatch) -> None:
    monkeypatch.setenv("TREND_LLM_TEMPERATURE", "0.42")
    monkeypatch.setenv("TREND_LLM_MODEL", "unit-test-model")

    llm = DummyLLM()
    chain = ConfigPatchChain.from_env(
        llm=llm,
        prompt_builder=build_config_patch_prompt,
        schema={"type": "object"},
    )

    assert chain.temperature == 0.42
    assert chain.model == "unit-test-model"

    chain._bind_llm()
    assert llm.bound == {"temperature": 0.42, "model": "unit-test-model"}


def test_config_patch_chain_emits_fleet_record(monkeypatch, tmp_path) -> None:
    fleet_path = tmp_path / "fleet.ndjson"
    monkeypatch.setenv("TREND_LANGSMITH_FLEET_PATH", str(fleet_path))
    monkeypatch.setenv("TREND_LLM_PROVIDER", "openai")

    chain = ConfigPatchChain.from_env(
        llm=InvokingLLM(),
        prompt_builder=build_config_patch_prompt,
        schema={"type": "object"},
        model="gpt-test",
    )

    chain.run(
        current_config={"analysis": {"top_n": 10}, "scenario": "base"},
        instruction="Set top_n to 12.",
        request_id="req-chain",
        log_operation=True,
    )

    record = json.loads(fleet_path.read_text(encoding="utf-8").splitlines()[0])
    assert record["schema_version"] == "langsmith-fleet/v1"
    assert record["operation"] == "nl_to_patch"
    assert record["status"] == "success"
    assert record["provider"] == "openai"
    assert record["model"] == "gpt-test"
    assert record["domain"]["request_id"] == "req-chain"
    assert record["domain"]["scenario_id"] == "base"
    assert record["domain"]["config_fingerprint"].startswith("sha256:")
    assert "Set top_n" not in json.dumps(record)
