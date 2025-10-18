import json
from collections.abc import Sequence
from pathlib import Path

import pytest
import requests

from tools.enforce_gate_branch_protection import (
    BranchProtectionError,
    BranchProtectionMissingError,
    DEFAULT_CONTEXTS,
    StatusCheckState,
    diff_contexts,
    fetch_status_checks,
    format_contexts,
    main,
    normalise_contexts,
    parse_contexts,
    require_token,
    resolve_api_root,
    update_status_checks,
)


REQUIRED_CONTEXTS = list(DEFAULT_CONTEXTS)


class DummyResponse(requests.Response):
    def __init__(
        self, status_code: int, payload: dict | None = None, text: str = ""
    ) -> None:
        super().__init__()
        self.status_code = status_code
        content = json.dumps(payload or {}) if payload is not None else text
        self._content = content.encode("utf-8")
        self.encoding = "utf-8"


class DummySession(requests.Session):
    def __init__(self, response: DummyResponse) -> None:
        super().__init__()
        self._response = response
        self.last_payload: dict | None = None

    def get(self, *_args: object, **_kwargs: object) -> requests.Response:
        return self._response

    def patch(self, *_args: object, **_kwargs: object) -> requests.Response:
        json_payload = _kwargs.get("json")
        self.last_payload = json_payload if isinstance(json_payload, dict) else None
        return self._response

    def put(self, *_args: object, **_kwargs: object) -> requests.Response:
        json_payload = _kwargs.get("json")
        self.last_payload = json_payload if isinstance(json_payload, dict) else None
        return self._response


def test_parse_contexts_defaults_to_required_contexts() -> None:
    assert parse_contexts(None) == REQUIRED_CONTEXTS
    assert parse_contexts([""]) == REQUIRED_CONTEXTS


def test_parse_contexts_preserves_non_empty_values() -> None:
    assert parse_contexts([" Gate / gate ", "Extra"]) == ["Gate / gate", "Extra"]


def test_normalise_contexts_deduplicates_and_sorts() -> None:
    assert normalise_contexts(["Extra", "Gate / gate", "Gate / gate"]) == [
        "Extra",
        "Gate / gate",
    ]


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
        update_status_checks(
            session, "owner/repo", "main", contexts=["Gate / gate"], strict=True
        )


def test_main_reports_no_changes_in_dry_run(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("GITHUB_TOKEN", "token")
    monkeypatch.setattr(
        "tools.enforce_gate_branch_protection._build_session",
        lambda _token: object(),
    )
    monkeypatch.setattr(
        "tools.enforce_gate_branch_protection.fetch_status_checks",
        lambda *_args, **_kwargs: StatusCheckState(
            strict=True,
            contexts=list(REQUIRED_CONTEXTS),
        ),
    )

    exit_code = main(["--repo", "owner/repo", "--branch", "main"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "No changes required." in captured.out


def test_main_applies_changes_when_requested(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("GITHUB_TOKEN", "token")
    monkeypatch.setattr(
        "tools.enforce_gate_branch_protection._build_session",
        lambda _token: object(),
    )
    monkeypatch.setattr(
        "tools.enforce_gate_branch_protection.fetch_status_checks",
        lambda *_args, **_kwargs: StatusCheckState(
            strict=False,
            contexts=["Legacy"],
        ),
    )

    captured_payload: dict[str, object] = {}

    def fake_update(
        _session: object,
        repo: str,
        branch: str,
        *,
        contexts: list[str],
        strict: bool,
        api_root: str = "",
    ) -> StatusCheckState:
        captured_payload.update(
            {
                "repo": repo,
                "branch": branch,
                "contexts": contexts,
                "strict": strict,
                "api_root": api_root,
            }
        )
        return StatusCheckState(strict=True, contexts=contexts)

    monkeypatch.setattr(
        "tools.enforce_gate_branch_protection.update_status_checks",
        fake_update,
    )

    exit_code = main(["--repo", "owner/repo", "--branch", "main", "--apply"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert captured_payload == {
        "repo": "owner/repo",
        "branch": "main",
        "contexts": REQUIRED_CONTEXTS,
        "strict": True,
        "api_root": "https://api.github.com",
    }
    assert "Update successful." in captured.out


def test_main_apply_with_no_clean_keeps_existing_contexts(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("GITHUB_TOKEN", "token")
    monkeypatch.setattr(
        "tools.enforce_gate_branch_protection._build_session",
        lambda _token: object(),
    )
    monkeypatch.setattr(
        "tools.enforce_gate_branch_protection.fetch_status_checks",
        lambda *_args, **_kwargs: StatusCheckState(
            strict=False,
            contexts=["Legacy"],
        ),
    )

    captured_payload: dict[str, object] = {}

    def fake_update(
        _session: object,
        repo: str,
        branch: str,
        *,
        contexts: list[str],
        strict: bool,
        api_root: str = "",
    ) -> StatusCheckState:
        captured_payload.update(
            {
                "repo": repo,
                "branch": branch,
                "contexts": contexts,
                "strict": strict,
                "api_root": api_root,
            }
        )
        return StatusCheckState(strict=True, contexts=contexts)

    monkeypatch.setattr(
        "tools.enforce_gate_branch_protection.update_status_checks",
        fake_update,
    )

    exit_code = main(
        [
            "--repo",
            "owner/repo",
            "--branch",
            "main",
            "--apply",
            "--no-clean",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert captured_payload == {
        "repo": "owner/repo",
        "branch": "main",
        "contexts": [
            "Gate / gate",
            "Enforce agents workflow protections",
            "Health 45 Agents Guard / Enforce agents workflow protections",
            "Legacy",
        ],
        "strict": True,
        "api_root": "https://api.github.com",
    }
    assert "Update successful." in captured.out


def test_main_reports_missing_rule_in_dry_run(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("GITHUB_TOKEN", "token")
    monkeypatch.setattr(
        "tools.enforce_gate_branch_protection._build_session",
        lambda _token: object(),
    )

    def _raise_missing(*_args: object, **_kwargs: object) -> None:
        raise BranchProtectionMissingError("missing")

    monkeypatch.setattr(
        "tools.enforce_gate_branch_protection.fetch_status_checks",
        _raise_missing,
    )

    exit_code = main(["--repo", "owner/repo", "--branch", "main"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "Would create branch protection" in captured.out


def test_main_check_mode_detects_drift(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("GITHUB_TOKEN", "token")
    monkeypatch.setattr(
        "tools.enforce_gate_branch_protection._build_session",
        lambda _token: object(),
    )
    monkeypatch.setattr(
        "tools.enforce_gate_branch_protection.fetch_status_checks",
        lambda *_args, **_kwargs: StatusCheckState(strict=False, contexts=["Legacy"]),
    )

    exit_code = main(
        [
            "--repo",
            "owner/repo",
            "--branch",
            "main",
            "--check",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "Would add contexts" in captured.out


def test_main_check_mode_succeeds_when_clean(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("GITHUB_TOKEN", "token")
    monkeypatch.setattr(
        "tools.enforce_gate_branch_protection._build_session",
        lambda _token: object(),
    )
    monkeypatch.setattr(
        "tools.enforce_gate_branch_protection.fetch_status_checks",
        lambda *_args, **_kwargs: StatusCheckState(
            strict=True,
            contexts=list(REQUIRED_CONTEXTS),
        ),
    )

    exit_code = main(
        [
            "--repo",
            "owner/repo",
            "--branch",
            "main",
            "--check",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "No changes required." in captured.out


def test_main_bootstraps_when_apply(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("GITHUB_TOKEN", "token")
    monkeypatch.setattr(
        "tools.enforce_gate_branch_protection._build_session",
        lambda _token: object(),
    )

    def _raise_missing(*_args: object, **_kwargs: object) -> None:
        raise BranchProtectionMissingError("missing")

    monkeypatch.setattr(
        "tools.enforce_gate_branch_protection.fetch_status_checks",
        _raise_missing,
    )

    captured_payload: dict[str, object] = {}

    def fake_bootstrap(
        _session: object,
        repo: str,
        branch: str,
        *,
        contexts: Sequence[str],
        strict: bool,
        api_root: str = "",
    ) -> StatusCheckState:
        captured_payload.update(
            {
                "repo": repo,
                "branch": branch,
                "contexts": list(contexts),
                "strict": strict,
                "api_root": api_root,
            }
        )
        return StatusCheckState(strict=True, contexts=list(contexts))

    monkeypatch.setattr(
        "tools.enforce_gate_branch_protection.bootstrap_branch_protection",
        fake_bootstrap,
    )

    exit_code = main(["--repo", "owner/repo", "--branch", "main", "--apply"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert captured_payload == {
        "repo": "owner/repo",
        "branch": "main",
        "contexts": REQUIRED_CONTEXTS,
        "strict": True,
        "api_root": "https://api.github.com",
    }
    assert "Created branch protection rule." in captured.out


def test_main_surfaces_branch_protection_errors(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("GITHUB_TOKEN", "token")
    monkeypatch.setattr(
        "tools.enforce_gate_branch_protection._build_session",
        lambda _token: object(),
    )

    def raise_error(*_args: object, **_kwargs: object) -> StatusCheckState:
        raise BranchProtectionError("boom")

    monkeypatch.setattr(
        "tools.enforce_gate_branch_protection.fetch_status_checks",
        raise_error,
    )

    exit_code = main(["--repo", "owner/repo", "--branch", "main"])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "error: boom" in captured.err


def test_main_flags_missing_require_up_to_date(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("GITHUB_TOKEN", "token")
    monkeypatch.setattr(
        "tools.enforce_gate_branch_protection._build_session",
        lambda _token: object(),
    )
    monkeypatch.setattr(
        "tools.enforce_gate_branch_protection.fetch_status_checks",
        lambda *_args, **_kwargs: StatusCheckState(
            strict=False,
            contexts=list(REQUIRED_CONTEXTS),
        ),
    )

    exit_code = main(["--repo", "owner/repo", "--branch", "main", "--check"])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "Would enable 'require branches to be up to date'." in captured.out


def test_main_writes_snapshot_when_drift_detected(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("GITHUB_TOKEN", "token")
    monkeypatch.setattr(
        "tools.enforce_gate_branch_protection._build_session",
        lambda _token: object(),
    )
    monkeypatch.setattr(
        "tools.enforce_gate_branch_protection.fetch_status_checks",
        lambda *_args, **_kwargs: StatusCheckState(strict=False, contexts=["Legacy"]),
    )

    snapshot_file = tmp_path / "snapshots" / "state.json"
    exit_code = main(
        [
            "--repo",
            "owner/repo",
            "--branch",
            "main",
            "--snapshot",
            str(snapshot_file),
        ]
    )

    assert exit_code == 0
    data = json.loads(snapshot_file.read_text())
    assert data["repository"] == "owner/repo"
    assert data["branch"] == "main"
    assert data["mode"] == "inspect"
    assert data["current"] == {"strict": False, "contexts": ["Legacy"]}
    assert data["desired"] == {"strict": True, "contexts": REQUIRED_CONTEXTS}
    assert data["changes_required"] is True
    assert data["changes_applied"] is False
    assert data["strict_unknown"] is False
    assert data["require_strict"] is False
    assert "after" not in data


def test_main_snapshot_updates_after_apply(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("GITHUB_TOKEN", "token")
    monkeypatch.setattr(
        "tools.enforce_gate_branch_protection._build_session",
        lambda _token: object(),
    )
    monkeypatch.setattr(
        "tools.enforce_gate_branch_protection.fetch_status_checks",
        lambda *_args, **_kwargs: StatusCheckState(strict=False, contexts=["Legacy"]),
    )

    def fake_update(
        _session: object,
        repo: str,
        branch: str,
        *,
        contexts: list[str],
        strict: bool,
        api_root: str = "",
    ) -> StatusCheckState:
        assert repo == "owner/repo"
        assert branch == "main"
        assert contexts == REQUIRED_CONTEXTS
        assert strict is True
        assert api_root == "https://api.github.com"
        return StatusCheckState(strict=True, contexts=contexts)

    monkeypatch.setattr(
        "tools.enforce_gate_branch_protection.update_status_checks",
        fake_update,
    )

    snapshot_file = tmp_path / "state.json"
    exit_code = main(
        [
            "--repo",
            "owner/repo",
            "--branch",
            "main",
            "--apply",
            "--snapshot",
            str(snapshot_file),
        ]
    )

    assert exit_code == 0
    data = json.loads(snapshot_file.read_text())
    assert data["mode"] == "apply"
    assert data["changes_required"] is True
    assert data["changes_applied"] is True
    assert data["strict_unknown"] is False
    assert data["require_strict"] is False
    assert data["after"] == {"strict": True, "contexts": REQUIRED_CONTEXTS}


def test_main_check_mode_allows_unknown_strict(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("GITHUB_TOKEN", "token")
    monkeypatch.setattr(
        "tools.enforce_gate_branch_protection._build_session",
        lambda _token: object(),
    )
    monkeypatch.setattr(
        "tools.enforce_gate_branch_protection.fetch_status_checks",
        lambda *_args, **_kwargs: StatusCheckState(
            strict=None,
            contexts=list(REQUIRED_CONTEXTS),
        ),
    )

    exit_code = main(
        [
            "--repo",
            "owner/repo",
            "--branch",
            "main",
            "--check",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "unknown - supply BRANCH_PROTECTION_TOKEN" in captured.out
    assert "Strict enforcement could not be confirmed" in captured.out


def test_main_check_mode_requires_unknown_strict_when_requested(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("GITHUB_TOKEN", "token")
    monkeypatch.setattr(
        "tools.enforce_gate_branch_protection._build_session",
        lambda _token: object(),
    )
    monkeypatch.setattr(
        "tools.enforce_gate_branch_protection.fetch_status_checks",
        lambda *_args, **_kwargs: StatusCheckState(
            strict=None,
            contexts=list(REQUIRED_CONTEXTS),
        ),
    )

    exit_code = main(
        [
            "--repo",
            "owner/repo",
            "--branch",
            "main",
            "--check",
            "--require-strict",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "requested; treating this as drift" in captured.out


def test_require_token_prefers_explicit_value(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    monkeypatch.delenv("GH_TOKEN", raising=False)
    assert require_token(" explicit ") == "explicit"


def test_require_token_env_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    monkeypatch.setenv("GH_TOKEN", "env-token")
    assert require_token() == "env-token"


def test_require_token_raises_when_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    monkeypatch.delenv("GH_TOKEN", raising=False)
    with pytest.raises(BranchProtectionError):
        require_token()


def test_resolve_api_root_prefers_explicit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GITHUB_API_URL", "https://enterprise/api/v3")
    assert (
        resolve_api_root("https://custom.example/api/v3/")
        == "https://custom.example/api/v3"
    )


def test_resolve_api_root_env_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GITHUB_API_URL", "https://enterprise/api/v3/")
    assert resolve_api_root() == "https://enterprise/api/v3"
