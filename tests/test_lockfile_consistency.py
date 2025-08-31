from pathlib import Path
import subprocess


def test_lockfile_up_to_date():
    result = subprocess.run(
        ["uv", "pip", "compile", "pyproject.toml"],
        capture_output=True,
        text=True,
        check=True,
    )
    lines = result.stdout.splitlines()
    if len(lines) >= 2:
        lines[1] = "#    uv pip compile pyproject.toml -o requirements.lock"
    compiled = "\n".join(lines) + "\n"
    assert compiled == Path("requirements.lock").read_text()
