from pathlib import Path


def test_issue_ledger_files_are_absent() -> None:
    ledger_files = list(Path(".").rglob("issue-*-ledger.yml"))
    allowed_roots = {
        Path("archives/agents/ledgers").resolve(),
        Path(".workflows-lib/.agents").resolve(),
    }
    unexpected = [
        path
        for path in ledger_files
        if not any(path.resolve().is_relative_to(root) for root in allowed_roots)
    ]
    assert not unexpected
