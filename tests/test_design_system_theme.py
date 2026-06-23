from __future__ import annotations

import tomllib
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
STREAMLIT_ENTRYPOINTS = (
    REPO_ROOT / "streamlit_app" / "app.py",
    REPO_ROOT / "streamlit_app" / "pages" / "1_Data.py",
    REPO_ROOT / "streamlit_app" / "pages" / "2_Model.py",
    REPO_ROOT / "streamlit_app" / "pages" / "3_Results.py",
    REPO_ROOT / "streamlit_app" / "pages" / "4_Help.py",
    REPO_ROOT / "streamlit_app" / "pages" / "5_Monte_Carlo.py",
    REPO_ROOT / "streamlit_app" / "pages" / "8_Validation.py",
)


def test_theme_adopted() -> None:
    config = tomllib.loads((REPO_ROOT / ".streamlit" / "config.toml").read_text())
    assert config["theme"]["base"] == "light"

    adapter = (REPO_ROOT / "streamlit_app" / "theme.py").read_text()
    assert "inject_theme" in adapter

    missing = [
        str(path.relative_to(REPO_ROOT))
        for path in STREAMLIT_ENTRYPOINTS
        if "apply_ds_theme()" not in path.read_text()
    ]
    assert not missing, f"Streamlit entrypoints missing apply_ds_theme(): {missing}"
