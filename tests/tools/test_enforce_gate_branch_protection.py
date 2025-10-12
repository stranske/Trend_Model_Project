import pytest

from tools.enforce_gate_branch_protection import (
    BranchProtectionError,
    StatusCheckState,
    diff_contexts,
    format_contexts,
    normalise_contexts,
    parse_contexts,
    fetch_status_checks,
    update_status_checks,
)


class DummyResponse:
    def __init__(self, status_code: int, payload: dict | None = None, text: str = "") -> None:
        self.status_code = status_code
        self._payload = payload or {}
        self.text = text

    def json(self) -> dict:
        return self._payload


class DummySession:
    def __init__(self, response: DummyResponse) -> None:
        self._response = response
        self.last_payload: dict | None = None

    def get(self, *_args, **_kwargs) -> DummyResponse:
        return self._response

    def patch(self, *_args, json: dict, **_kwargs) -> DummyResponse:
        self.last_payload = json
        return self._response


def test_parse_contexts_defaults_to_gate_when_missing() -> None:
    assert parse_contexts(None) == ["Gate / gate"]
    assert parse_contexts([""]) == ["Gate / gate"]


def test_parse_contexts_preserves_non_empty_values() -> None:
    assert parse_contexts([" Gate / gate ", "Extra"]) == ["Gate / gate", "Extra"]


def test_normalise_contexts_deduplicates_and_sorts() -> None:
    assert normalise_contexts(["Extra", "Gate / gate", "Gate / gate"]) == ["Extra", "Gate / gate"]


def test_diff_contexts_returns_expected_differences() -> None:
    to_add, to_remove = diff_contexts(["Gate / gate"], ["Gate / gate", "Extra"])
    assert to_add == ["Extra"]
    assert to_remove == []

    to_add, to_remove = diff_contexts(["Legacy"], ["Gate / gate"])
    assert to_add == ["Gate / gate"]
    assert to_remove == ["Legacy"]


def test_format_contexts_handles_empty_and_multiple_values() -> None:
    assert format_contexts([]) == "(none)"
    assert format_contexts(["Gate / gate", "Extra"]) == "Gate / gate, Extra"


def test_fetch_status_checks_raises_for_missing_rule() -> None:
    session = DummySession(DummyResponse(404))
    with pytest.raises(BranchProtectionError):
        fetch_status_checks(session, "owner/repo", "main")


def test_fetch_status_checks_returns_state() -> None:
    response = DummyResponse(200, {"strict": True, "contexts": ["Gate / gate"]})
    session = DummySession(response)
    state = fetch_status_checks(session, "owner/repo", "main")
    assert state == StatusCheckState(strict=True, contexts=["Gate / gate"])


def test_update_status_checks_submits_payload_and_returns_state() -> None:
    response = DummyResponse(200, {"strict": True, "contexts": ["Gate / gate"]})
    session = DummySession(response)

    state = update_status_checks(
        session,
        "owner/repo",
        "main",
        contexts=["Gate / gate"],
        strict=True,
    )

    assert session.last_payload == {"contexts": ["Gate / gate"], "strict": True}
    assert state == StatusCheckState(strict=True, contexts=["Gate / gate"])


def test_update_status_checks_raises_on_failure() -> None:
    session = DummySession(DummyResponse(500, text="error"))
    with pytest.raises(BranchProtectionError):
        update_status_checks(session, "owner/repo", "main", contexts=["Gate / gate"], strict=True)
