from __future__ import annotations

from pathlib import Path

import trend.cli as cli
from trend_analysis.config import ConfigPatch


def test_apply_nl_instruction_logs_apply_patch(monkeypatch) -> None:
    captured: dict[str, object] = {}
    entries: list[cli.NLOperationLog] = []

    class DummyChain:
        model = "test-model"
        temperature = 0.2

        def run(self, *, request_id: str, **_: object) -> ConfigPatch:
            captured["request_id"] = request_id
            return ConfigPatch(operations=[], summary="noop")

    def fake_build_chain(provider: str | None = None) -> DummyChain:
        return DummyChain()

    def fake_write(entry: cli.NLOperationLog, **_: object) -> Path:
        entries.append(entry)
        return Path("nl_ops_2099-01-01.jsonl")

    monkeypatch.setattr(cli, "_build_nl_chain", fake_build_chain)
    monkeypatch.setattr(cli, "write_nl_log", fake_write)

    patch, updated, diff, model_name, temperature = cli._apply_nl_instruction(
        {"portfolio": {"foo": "bar"}},
        "update foo",
        request_id="req-123",
    )

    assert captured["request_id"] == "req-123"
    assert entries
    entry = entries[0]
    assert entry.operation == "apply_patch"
    assert entry.request_id == "req-123"
    assert entry.parsed_patch == patch
    assert model_name == "test-model"
    assert temperature == 0.2
    assert updated["portfolio"]["foo"] == "bar"
    assert diff == ""
