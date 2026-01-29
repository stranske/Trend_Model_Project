from __future__ import annotations

from collections import Counter

import yaml

from utils.paths import proj_path


def test_registry_scenario_names_unique() -> None:
    registry_path = proj_path("config", "scenarios", "monte_carlo", "index.yml")
    payload = yaml.safe_load(registry_path.read_text(encoding="utf-8"))
    scenarios = payload.get("scenarios", [])
    names = [str(entry.get("name") or "").strip() for entry in scenarios]

    assert all(names), "Scenario registry contains empty names"

    counts = Counter(names)
    duplicates = sorted(name for name, count in counts.items() if count > 1)
    assert not duplicates, f"Scenario registry has duplicate names: {duplicates}"
