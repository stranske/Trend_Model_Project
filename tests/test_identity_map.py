from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from trend_analysis.identity import IdentityMap
from trend_analysis.config.schema_validation import load_schema, validate_config_data
from trend_analysis.reporting.run_artifacts import write_run_artifacts


def _identity_config() -> dict[str, object]:
    return {
        "identity": {
            "entities": [
                {
                    "canonical_id": "fund:aqr-managed-futures",
                    "display_name": "AQR Managed Futures",
                    "aliases": ["AQR MF", "AQR Managed Futures"],
                }
            ]
        }
    }


def _write_manifest(
    tmp_path: Path,
    *,
    run_id: str = "run123",
    selected_funds: list[str],
    config: dict[str, object] | None = None,
) -> dict[str, object]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    input_path = tmp_path / "returns.csv"
    pd.DataFrame({"AQR MF": [0.01], "AQR Managed Futures": [0.02]}).to_csv(
        input_path, index=False
    )
    run_dir = write_run_artifacts(
        output_dir=tmp_path / "out",
        run_id=run_id,
        config=config or _identity_config(),
        config_path="config/demo.yml",
        input_path=input_path,
        data_frame=pd.DataFrame({"date": ["2026-01-31"], "AQR MF": [0.01]}),
        metrics_frame=pd.DataFrame({"cagr": [0.1]}),
        run_details={"selected_funds": selected_funds},
        exported_files=[],
        summary_text="summary",
        identity_map=IdentityMap.from_config(config or _identity_config()),
    )
    return json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))


def test_aliases_collapse_to_one_canonical_id(tmp_path: Path) -> None:
    manifest = _write_manifest(
        tmp_path,
        selected_funds=["AQR MF", "AQR Managed Futures"],
    )

    entities = manifest["selected_entities"]
    canonical_ids = [entity["canonical_id"] for entity in entities]

    assert manifest["selected_funds"] == ["AQR MF", "AQR Managed Futures"]
    assert canonical_ids.count("fund:aqr-managed-futures") == 1
    assert entities[0]["labels"] == ["AQR MF", "AQR Managed Futures"]
    assert entities[0]["resolved"] is True


def test_unmatched_label_marked_unknown(tmp_path: Path) -> None:
    manifest = _write_manifest(tmp_path, selected_funds=["Unmapped Fund"])

    entity = manifest["selected_entities"][0]

    assert entity["canonical_id"] == "unknown:Unmapped Fund"
    assert entity["display_name"] == "Unmapped Fund"
    assert entity["resolved"] is False


def test_identity_resolution_is_deterministic(tmp_path: Path) -> None:
    first = _write_manifest(
        tmp_path / "first",
        run_id="run-a",
        selected_funds=["AQR MF", "AQR Managed Futures"],
    )
    second = _write_manifest(
        tmp_path / "second",
        run_id="run-b",
        selected_funds=["AQR MF", "AQR Managed Futures"],
    )

    assert first["selected_entities"] == second["selected_entities"]


def test_identity_resolution_prefers_exact_alias_before_normalized_collision() -> None:
    resolver = IdentityMap.from_config(
        {
            "identity": {
                "entities": [
                    {"canonical_id": "fund:upper", "display_name": "ABC Fund"},
                    {"canonical_id": "fund:lower", "display_name": "abc fund"},
                ]
            }
        }
    )

    assert resolver.resolve("ABC Fund").canonical_id == "fund:upper"
    assert resolver.resolve("abc fund").canonical_id == "fund:lower"


def test_identity_config_validates_against_schema() -> None:
    schema = load_schema()
    payload = {
        "version": "1",
        "identity": {
            "entities": [
                {
                    "canonical_id": "fund:aqr-managed-futures",
                    "display_name": "AQR Managed Futures",
                    "aliases": ["AQR MF"],
                }
            ],
            "universes": ["universe/core.yml"],
        },
    }

    assert validate_config_data(payload, schema) == []


def test_identity_universe_docs_path_resolves_from_config_dir() -> None:
    resolver = IdentityMap.from_config(
        {"identity": {"universes": ["universe/core.yml"]}},
        base_path=Path("config"),
    )

    assert resolver.resolve("AHL Dimension").canonical_id == "fund:ahl-dimension"
