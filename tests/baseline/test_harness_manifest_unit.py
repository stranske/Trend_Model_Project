from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from . import manifest
from .harness import _resolve_csv_path, apply_patch


def test_apply_patch_materializes_sections_and_nested_keys() -> None:
    cfg = SimpleNamespace(existing={"k": 1}, seed=42)

    apply_patch(
        cfg,
        {
            "seed": 99,
            "portfolio.constraints.max_weight": 0.15,
            "existing.inner.value": "x",
        },
    )

    assert cfg.seed == 99
    assert cfg.portfolio["constraints"]["max_weight"] == 0.15
    assert cfg.existing["inner"]["value"] == "x"


def test_apply_patch_none_is_noop() -> None:
    cfg = SimpleNamespace(seed=123)
    out = apply_patch(cfg, None)
    assert out is cfg
    assert cfg.seed == 123


def test_resolve_csv_path_prefers_config_dir_then_repo_root(tmp_path: Path, monkeypatch) -> None:
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    cfg_path = config_dir / "demo.yml"
    cfg_path.write_text("demo: true\n", encoding="utf-8")

    nested_data = config_dir / "data"
    nested_data.mkdir()
    local_csv = nested_data / "returns.csv"
    local_csv.write_text("a,b\n1,2\n", encoding="utf-8")

    cfg = SimpleNamespace(data={"csv_path": "data/returns.csv"})
    assert _resolve_csv_path(cfg, cfg_path) == local_csv.resolve()

    local_csv.unlink()
    repo_data = tmp_path / "repo-data"
    repo_data.mkdir()
    repo_csv = repo_data / "returns.csv"
    repo_csv.write_text("a,b\n1,2\n", encoding="utf-8")

    monkeypatch.setattr("tests.baseline.harness.REPO_ROOT", tmp_path)
    cfg = SimpleNamespace(data={"csv_path": "repo-data/returns.csv"})
    assert _resolve_csv_path(cfg, cfg_path) == repo_csv.resolve()


def test_schema_leaf_keys_handles_wrappers(tmp_path: Path) -> None:
    schema_path = tmp_path / "schema.json"
    schema_path.write_text(
        """
{
  "allOf": [
    {
      "properties": {
        "portfolio": {
          "properties": {
            "constraints": {
              "properties": {
                "max_weight": {"type": "number"}
              }
            }
          }
        }
      }
    }
  ],
  "anyOf": [
    {
      "properties": {
        "vol_adjust": {
          "properties": {
            "target_vol": {"type": "number"}
          }
        }
      }
    }
  ],
  "oneOf": [
    {
      "properties": {
        "selection": {
          "properties": {
            "selection_count": {"type": "integer"}
          }
        }
      }
    }
  ]
}
""".strip(),
        encoding="utf-8",
    )

    leaves = manifest.schema_leaf_keys(schema_path)
    assert "portfolio.constraints.max_weight" in leaves
    assert "vol_adjust.target_vol" in leaves
    assert "selection.selection_count" in leaves


def test_catalog_touched_keys_collects_scenarios_and_toggles() -> None:
    catalog = {
        "scenarios": [
            {
                "base": {"portfolio.constraints.max_weight": 0.2},
                "control": {"selection.selection_count": 5},
                "vary": {"selection.selection_count": 10},
                "param": "selection.selection_count",
            }
        ],
        "toggles": [
            {
                "base": {"portfolio.weighting.name": "robust_mv"},
                "flag": "vol_adjust.enabled",
            }
        ],
    }

    touched = manifest.catalog_touched_keys(catalog)
    assert "portfolio.constraints.max_weight" in touched
    assert "selection.selection_count" in touched
    assert "portfolio.weighting.name" in touched
    assert "vol_adjust.enabled" in touched


def test_catalog_touched_keys_expands_object_parent_to_leaves() -> None:
    # A scenario that clears an object-valued config key (e.g. null-ing
    # portfolio.custom_weights so weighting.name can take effect; #5537)
    # references only the parent path, but the schema enumerates that object
    # at its leaves. With schema_leaves provided, the parent expands to its
    # leaf children so the schema typo-guard does not flag it as unknown.
    catalog = {
        "scenarios": [
            {
                "base": {"portfolio.custom_weights": None},
                "control": {"portfolio.weighting.name": "equal"},
                "vary": {"portfolio.weighting.name": "risk_parity"},
                "param": "portfolio.weighting.name",
            }
        ]
    }
    leaves = {
        "portfolio.custom_weights.Mgr_01",
        "portfolio.custom_weights.Mgr_02",
        "portfolio.weighting.name",
    }

    touched = manifest.catalog_touched_keys(catalog, leaves)

    assert "portfolio.custom_weights" not in touched
    assert {"portfolio.custom_weights.Mgr_01", "portfolio.custom_weights.Mgr_02"} <= touched
    assert "portfolio.weighting.name" in touched
    # touched keys are all real schema leaves -> no false "unknown" flag.
    assert touched <= leaves


def test_catalog_touched_keys_keeps_unknown_keys_for_typo_guard() -> None:
    # A key that is neither a leaf nor the parent of one is left as-is so the
    # schema guard still surfaces genuine typos.
    catalog = {"scenarios": [{"base": {"portfolio.custom_wieghts": None}}]}
    leaves = {"portfolio.custom_weights.Mgr_01"}

    touched = manifest.catalog_touched_keys(catalog, leaves)

    assert "portfolio.custom_wieghts" in touched
