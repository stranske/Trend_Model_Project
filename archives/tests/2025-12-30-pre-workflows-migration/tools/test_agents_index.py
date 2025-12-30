from pathlib import Path

from tools.agents_index import AgentBootstrap, list_agent_bootstraps


def test_list_agent_bootstraps_returns_sorted_issue_numbers():
    agents_dir = Path(__file__).resolve().parents[2] / "agents"

    bootstraps = list_agent_bootstraps(agents_dir)

    assert all(isinstance(entry, AgentBootstrap) for entry in bootstraps)
    issues = [entry.issue for entry in bootstraps]
    assert issues == sorted(issues)


def test_list_agent_bootstraps_includes_current_issue_file():
    agents_dir = Path(__file__).resolve().parents[2] / "agents"

    bootstraps = list_agent_bootstraps(agents_dir)

    # Check for any active codex file (3878 was archived, 3572 is current)
    assert any(entry.issue == 3572 for entry in bootstraps)
    assert any(entry.path.name == "codex-3572.md" for entry in bootstraps)
