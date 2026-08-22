import json
import os
import subprocess
import zipfile
from pathlib import Path

import yaml

DEMO_CONFIG = Path("config/demo.yml")
DEMO_RETURNS = Path("demo/demo_returns.csv")


def _isolated_demo_config(tmp_path: Path) -> Path:
    payload = yaml.safe_load(DEMO_CONFIG.read_text(encoding="utf-8"))
    payload["data"]["csv_path"] = str(DEMO_RETURNS.resolve())
    payload["export"]["directory"] = str(tmp_path / "exports")
    payload["run"]["checkpoint_dir"] = str(tmp_path / "checkpoints")
    config_path = tmp_path / "demo.yml"
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return config_path


def _run_cli(command: list[str], env: dict[str, str]) -> subprocess.CompletedProcess[bytes]:
    result = subprocess.run(
        command,
        check=False,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert result.returncode == 0, result.stderr.decode("utf-8", errors="replace")
    return result


def test_cli_reproducible_same_seed(tmp_path: Path) -> None:
    # Ensure demo data exists (if not, skip fast)
    if not DEMO_RETURNS.exists():  # pragma: no cover - safety
        import pytest

        pytest.skip("Demo returns file missing")
    config = _isolated_demo_config(tmp_path)
    returns = DEMO_RETURNS.resolve()
    bundle1 = tmp_path / "b1.zip"
    bundle2 = tmp_path / "b2.zip"
    cmd_base = [
        "bash",
        "scripts/trend",
        "run",
        "-c",
        str(config),
        "-i",
        str(returns),
        "--log-file",
        str(tmp_path / "run.jsonl"),
    ]

    env = os.environ.copy()
    env.pop("PYTHONHASHSEED", None)  # ensure script sets it deterministically

    r1 = _run_cli(
        cmd_base + ["--seed", "777", "--bundle", str(bundle1)],
        env,
    )
    r2 = _run_cli(
        cmd_base + ["--seed", "777", "--bundle", str(bundle2)],
        env,
    )
    assert r1.returncode == 0 and r2.returncode == 0

    # Hash run_meta.json inside each bundle
    import zipfile

    def _read_meta(b: Path) -> dict:
        with zipfile.ZipFile(b) as z:
            with z.open("run_meta.json") as f:
                return json.loads(f.read().decode("utf-8"))

    meta1 = _read_meta(bundle1)
    meta2 = _read_meta(bundle2)
    assert meta1["seed"] == 777 == meta2["seed"]
    # Same config + seed => same run_id
    assert meta1["run_id"] == meta2["run_id"]


def test_cli_seed_precedence_env_vs_flag(tmp_path: Path) -> None:
    if not DEMO_RETURNS.exists():  # pragma: no cover - safety
        import pytest

        pytest.skip("Demo returns file missing")
    config = _isolated_demo_config(tmp_path)
    returns = DEMO_RETURNS.resolve()
    bundle = tmp_path / "b.zip"
    cmd_base = [
        "bash",
        "scripts/trend",
        "run",
        "-c",
        str(config),
        "-i",
        str(returns),
        "--log-file",
        str(tmp_path / "run.jsonl"),
    ]
    env = os.environ.copy()
    env["TREND_SEED"] = "123"
    env.pop("PYTHONHASHSEED", None)
    # CLI flag should override TREND_SEED
    _run_cli(
        cmd_base + ["--seed", "999", "--bundle", str(bundle)],
        env,
    )

    with zipfile.ZipFile(bundle) as z:
        with z.open("run_meta.json") as f:
            meta = json.loads(f.read().decode("utf-8"))
    assert meta["seed"] == 999
