from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class DiagnosticEntry:
    tool: str
    scenario: str
    outcome: str
    changed: bool
    notes: Optional[str] = None


@dataclass
class DiagnosticsRecorder:
    _entries: List[DiagnosticEntry] = field(default_factory=list)

    def reset(self) -> None:
        self._entries.clear()

    def has_entries(self) -> bool:
        return bool(self._entries)

    def record(
        self,
        *,
        tool: str,
        scenario: str,
        outcome: str,
        changed: bool,
        notes: Optional[str] = None,
    ) -> None:
        self._entries.append(
            DiagnosticEntry(
                tool=tool,
                scenario=scenario,
                outcome=outcome,
                changed=changed,
                notes=notes,
            )
        )

    def _tool_summary(self) -> Dict[str, Any]:
        summary: Dict[str, Any] = {}
        for entry in self._entries:
            tool_bucket = summary.setdefault(
                entry.tool,
                {
                    "total": 0,
                    "changed": 0,
                    "unchanged": 0,
                    "outcomes": {},
                    "scenarios": [],
                },
            )
            tool_bucket["total"] += 1
            if entry.changed:
                tool_bucket["changed"] += 1
            else:
                tool_bucket["unchanged"] += 1
            tool_bucket["outcomes"][entry.outcome] = (
                tool_bucket["outcomes"].get(entry.outcome, 0) + 1
            )
            tool_bucket["scenarios"].append(
                {
                    "scenario": entry.scenario,
                    "outcome": entry.outcome,
                    "changed": entry.changed,
                    **({"notes": entry.notes} if entry.notes else {}),
                }
            )
        return summary

    def flush(self, target: Path) -> Path:
        target.parent.mkdir(parents=True, exist_ok=True)
        payload: Dict[str, Any] = {
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "entry_count": len(self._entries),
            "tools": self._tool_summary(),
        }
        target.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        return target


_GLOBAL_RECORDER = DiagnosticsRecorder()


def get_recorder() -> DiagnosticsRecorder:
    """Return the global diagnostics recorder used by pytest fixtures."""
    return _GLOBAL_RECORDER
