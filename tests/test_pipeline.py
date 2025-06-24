import pathlib
import pytest
import sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from trend_analysis import config, pipeline


def build_cfg(path: str) -> config.Config:
    return config.Config(
        version="test",
        data={"returns_path": path},
        preprocessing={},
        vol_adjust={"target_vol": 0.1},
        sample_split={
            "in_start": "2020-01",
            "in_end": "2020-12",
            "out_start": "2021-01",
            "out_end": "2021-12",
        },
        portfolio={},
        metrics={},
        export={},
        run={},
    )


def test_pipeline_run(tmp_path):
    cfg = build_cfg("tests/data/sample.csv")
    df = pipeline.run(cfg)
    assert not df.empty


def test_pipeline_missing_file(tmp_path):
    cfg = build_cfg(str(tmp_path / "missing.csv"))
    try:
        pipeline.run(cfg)
    except FileNotFoundError:
        assert True
    else:
        assert False


def test_pipeline_no_path():
    cfg = build_cfg("tests/data/sample.csv")
    cfg.data.pop("returns_path")
    with pytest.raises(ValueError):
        pipeline.run(cfg)

def test_parquet_fixture(parquet_data):
    assert parquet_data.shape[0] == 24
    assert parquet_data.shape[1] == 7
