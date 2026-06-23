from __future__ import annotations

import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DENSE_ENTRYPOINTS = (
    "streamlit_app/pages/1_Data.py",
    "streamlit_app/pages/2_Model.py",
    "streamlit_app/pages/3_Results.py",
    "streamlit_app/pages/5_Monte_Carlo.py",
    "streamlit_app/pages/8_Validation.py",
)


def _streamlit_entrypoints() -> tuple[Path, ...]:
    pages = sorted((REPO_ROOT / "streamlit_app" / "pages").glob("*.py"))
    return (REPO_ROOT / "streamlit_app" / "app.py", *pages)


def test_theme_adopted() -> None:
    config = tomllib.loads((REPO_ROOT / ".streamlit" / "config.toml").read_text())
    assert config["theme"]["base"] == "light"

    adapter = (REPO_ROOT / "streamlit_app" / "theme.py").read_text()
    assert "inject_theme" in adapter

    missing = [
        str(path.relative_to(REPO_ROOT))
        for path in _streamlit_entrypoints()
        if "apply_ds_theme()" not in path.read_text()
    ]
    assert not missing, f"Streamlit entrypoints missing apply_ds_theme(): {missing}"


def test_streamlit_entrypoint_discovery_covers_every_page() -> None:
    discovered = {str(path.relative_to(REPO_ROOT)) for path in _streamlit_entrypoints()}
    assert "streamlit_app/app.py" in discovered
    assert "streamlit_app/pages/8_Validation.py" in discovered
    assert not any("pages/6_" in path or "pages/7_" in path for path in discovered)


def test_data_dense_entrypoints_apply_compact_density() -> None:
    adapter = (REPO_ROOT / "streamlit_app" / "theme.py").read_text()
    assert "density-compact" in adapter

    missing = [
        path
        for path in DATA_DENSE_ENTRYPOINTS
        if "apply_density_compact()" not in (REPO_ROOT / path).read_text()
    ]
    assert not missing, f"Data-dense entrypoints missing compact density: {missing}"
