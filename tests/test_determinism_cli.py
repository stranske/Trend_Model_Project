import json
import os
import subprocess
import zipfile
from pathlib import Path

DEMO_CONFIG = Path("config/demo.yml")
DEMO_RETURNS = Path("demo/demo_returns.csv")


def test_cli_reproducible_same_seed(tmp_path: Path) -> None:
    # Ensure demo data exists (if not, skip fast)
    if not DEMO_RETURNS.exists():  # pragma: no cover - safety
        import pytest

        pytest.skip("Demo returns file missing")
    config = DEMO_CONFIG
    returns = DEMO_RETURNS
    bundle1 = tmp_path / "b1.zip"
    bundle2 = tmp_path / "b2.zip"
    cmd_base = [
        "bash",
        "scripts/trend-model",
        "run",
        "-c",
        str(config),
        "-i",
        str(returns),
    ]

    env = os.environ.copy()
    env.pop("PYTHONHASHSEED", None)  # ensure script sets it deterministically

    r1 = subprocess.run(
        cmd_base + ["--seed", "777", "--bundle", str(bundle1)],
        check=True,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    r2 = subprocess.run(
        cmd_base + ["--seed", "777", "--bundle", str(bundle2)],
        check=True,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
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
    config = DEMO_CONFIG
    returns = DEMO_RETURNS
    bundle = tmp_path / "b.zip"
    cmd_base = [
        "bash",
        "scripts/trend-model",
        "run",
        "-c",
        str(config),
        "-i",
        str(returns),
    ]
    env = os.environ.copy()
    env["TREND_SEED"] = "123"
    env.pop("PYTHONHASHSEED", None)
    # CLI flag should override TREND_SEED
    subprocess.run(
        cmd_base + ["--seed", "999", "--bundle", str(bundle)],
        check=True,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    with zipfile.ZipFile(bundle) as z:
        with z.open("run_meta.json") as f:
            meta = json.loads(f.read().decode("utf-8"))
    assert meta["seed"] == 999
