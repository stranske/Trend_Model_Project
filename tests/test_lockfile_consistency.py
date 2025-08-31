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
    # Normalize the command comment line to ensure consistency, regardless of its position.
    for i, line in enumerate(lines):
        if line.strip().startswith("#") and "uv pip compile pyproject.toml" in line:
            lines[i] = "#    uv pip compile pyproject.toml -o requirements.lock"
            break
    compiled = "\n".join(lines) + "\n"
    assert compiled == Path("requirements.lock").read_text()
