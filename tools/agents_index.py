from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

CODEx_PATTERN = re.compile(r"^codex-(\d+)\.md$")


@dataclass(frozen=True)
class AgentBootstrap:
    issue: int
    path: Path


def _iter_agent_files(agent_dir: Path) -> Iterable[Path]:
    yield from agent_dir.glob("codex-*.md")


def list_agent_bootstraps(
    agent_dir: Path | str = Path(__file__).resolve().parents[1] / "agents",
) -> list[AgentBootstrap]:
    """Return sorted bootstrap metadata for Codex issue trackers.

    Args:
        agent_dir: Directory containing codex issue markdown files.

    Returns:
        A list of ``AgentBootstrap`` entries sorted by issue number.
    """

    base = Path(agent_dir)
    bootstraps: list[AgentBootstrap] = []

    for path in _iter_agent_files(base):
        match = CODEx_PATTERN.match(path.name)
        if match:
            bootstraps.append(AgentBootstrap(int(match.group(1)), path))

    return sorted(bootstraps, key=lambda entry: entry.issue)
