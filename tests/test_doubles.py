"""Tests for shared test doubles."""

from __future__ import annotations

import pytest

pytest.importorskip("langchain_core")

from langchain_core.runnables import RunnableLambda  # noqa: E402


class _StructuredOutputLLM(RunnableLambda):
    def __init__(self, *, responses: list[object]) -> None:
        self._responses = iter(responses)
        # RunnableLambda expects a callable that accepts the dict input shape used by
        # chain.invoke; this coupling is intentional and this test should fail if
        # RunnableLambda.__init__ or invocation semantics change.
        super().__init__(self._respond)

    def supports_structured_output(self) -> bool:
        return True

    def with_structured_output(self, _schema) -> RunnableLambda:
        # If RunnableLambda changes its callable expectations, this should break fast.
        return RunnableLambda(self._respond)

    def _respond(self, _prompt_value, **_kwargs) -> object:
        response = next(self._responses)
        if isinstance(response, BaseException):
            raise response
        return response


def test_structured_output_llm_runnable_lambda_accepts_prompt_dict() -> None:
    llm = _StructuredOutputLLM(responses=[{"ok": True}])
    structured_llm = llm.with_structured_output(dict)

    result = structured_llm.invoke({"prompt": "hello"})

    assert result == {"ok": True}
