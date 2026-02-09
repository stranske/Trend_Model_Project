"""Tests for ConfigPatchVariantsChain retry counters."""

from __future__ import annotations

import pytest

pytest.importorskip("langchain_core")

from tests.test_doubles import _StructuredOutputLLM  # noqa: E402
from trend_analysis.llm.chain import ConfigPatchVariantsChain  # noqa: E402
from trend_analysis.llm.prompts import build_variant_patch_prompt  # noqa: E402


def _valid_patch(summary: str, value: float) -> dict[str, object]:
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


def _valid_payload() -> dict[str, object]:
    return {
        "variants": [
            {"label": "conservative", "patch": _valid_patch("Conservative tweak", 0.3)},
            {"label": "baseline", "patch": _valid_patch("Baseline tweak", 0.4)},
            {"label": "aggressive", "patch": _valid_patch("Aggressive tweak", 0.5)},
        ]
    }


def _invalid_payload() -> dict[str, object]:
    return {"variants": []}


def _make_chain(responses: list[object], *, retries: int = 1) -> ConfigPatchVariantsChain:
    llm = _StructuredOutputLLM(responses=responses)
    return ConfigPatchVariantsChain(
        llm=llm,
        prompt_builder=build_variant_patch_prompt,
        schema={"type": "object"},
        retries=retries,
    )


def test_structured_retry_count_initializes_to_zero() -> None:
    chain = _make_chain([_valid_payload()])
    assert chain.structured_repair_retry_count == 0


def test_structured_retry_count_zero_on_first_attempt() -> None:
    chain = _make_chain([_valid_payload()])

    chain.run(
        current_config={"portfolio": {"max_weight": 0.2}},
        instruction="Generate variants.",
    )

    assert chain.structured_repair_retry_count == 0


def test_structured_retry_count_one_retry() -> None:
    chain = _make_chain([_invalid_payload(), _valid_payload()], retries=1)

    chain.run(
        current_config={"portfolio": {"max_weight": 0.2}},
        instruction="Generate variants.",
    )

    assert chain.structured_repair_retry_count == 1


def test_structured_retry_count_multiple_retries() -> None:
    chain = _make_chain(
        [_invalid_payload(), _invalid_payload(), _valid_payload()],
        retries=2,
    )

    chain.run(
        current_config={"portfolio": {"max_weight": 0.2}},
        instruction="Generate variants.",
    )

    assert chain.structured_repair_retry_count == 2


def test_structured_retry_count_cumulative_across_runs() -> None:
    chain = _make_chain(
        [
            _invalid_payload(),
            _valid_payload(),
            _invalid_payload(),
            _invalid_payload(),
            _valid_payload(),
        ],
        retries=2,
    )

    chain.run(
        current_config={"portfolio": {"max_weight": 0.2}},
        instruction="Generate variants.",
    )
    chain.run(
        current_config={"portfolio": {"max_weight": 0.2}},
        instruction="Generate variants again.",
    )

    assert chain.structured_repair_retry_count == 3


def test_structured_retry_count_is_isolated_per_instance() -> None:
    chain_with_retry = _make_chain([_invalid_payload(), _valid_payload()])
    chain_without_retry = _make_chain([_valid_payload()])

    chain_with_retry.run(
        current_config={"portfolio": {"max_weight": 0.2}},
        instruction="Generate variants.",
    )

    assert chain_with_retry.structured_repair_retry_count == 1
    assert chain_without_retry.structured_repair_retry_count == 0
