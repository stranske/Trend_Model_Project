"""Tests covering price frame handling in the multi-period engine."""

from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

from trend_analysis.multi_period import engine as mp_engine


class DummyCfg:
    """Minimal configuration object for exercising ``engine.run``."""

    def __init__(self) -> None:
        self.data: dict[str, object] = {"csv_path": "unused.csv"}
        self.portfolio: dict[str, object] = {
            "policy": "random",
            "selection_mode": "all",
            "random_n": 1,
            "rank": {},
            "manual_list": None,
            "indices_list": [],
        }
        self.vol_adjust: dict[str, float] = {"target_vol": 1.0}
        self.run: dict[str, float] = {"monthly_cost": 0.0}
        self.benchmarks: dict[str, str] = {}
        self.seed: int = 0
        self.performance: dict[str, object] = {"enable_cache": False}

    def model_dump(self) -> dict[str, object]:  # pragma: no cover - trivial
        return {"multi_period": {}}


def test_run_price_frames_requires_mapping() -> None:
    cfg = DummyCfg()
    df = pd.DataFrame({"Date": pd.to_datetime(["2020-01-31"])})

    with pytest.raises(TypeError):
        mp_engine.run(cfg, df=df, price_frames="not-a-mapping")


def test_run_price_frames_enforces_dataframe_members() -> None:
    cfg = DummyCfg()
    df = pd.DataFrame({"Date": pd.to_datetime(["2020-01-31"])})
    bad_frames = {"2020-01": [1, 2, 3]}

    with pytest.raises(TypeError, match="must be a pandas DataFrame"):
        mp_engine.run(cfg, df=df, price_frames=bad_frames)


def test_run_price_frames_validate_columns() -> None:
    cfg = DummyCfg()
    df = pd.DataFrame({"Date": pd.to_datetime(["2020-01-31"])})
    bad_frames = {"2020-01": pd.DataFrame({"Foo": [1.0]})}

    with pytest.raises(ValueError, match="missing required columns"):
        mp_engine.run(cfg, df=df, price_frames=bad_frames)


def test_run_price_frames_error_lists_required_and_available_columns() -> None:
    cfg = DummyCfg()
    df = pd.DataFrame({"Date": pd.to_datetime(["2020-01-31"])})
    bad_frame = pd.DataFrame({"Foo": [1.0], "Bar": [2.0]})
    price_frames = {"2020-01": bad_frame}

    with pytest.raises(ValueError) as exc:
        mp_engine.run(cfg, df=df, price_frames=price_frames)

    message = str(exc.value)
    assert "Required columns are: ['Date']" in message
    assert "Available columns are: ['Foo', 'Bar']" in message


def test_run_price_frames_error_mentions_available_columns() -> None:
    cfg = DummyCfg()
    df = pd.DataFrame({"Date": pd.to_datetime(["2020-01-31"])})
    bad_frames = {
        "2020-01": pd.DataFrame({"Foo": [1.0], "Bar": [2.0]})
    }

    with pytest.raises(ValueError) as exc:
        mp_engine.run(cfg, df=df, price_frames=bad_frames)

    message = str(exc.value)
    assert "Available columns" in message
    assert "['Foo', 'Bar']" in message


def test_run_price_frames_rejects_empty_mapping() -> None:
    cfg = DummyCfg()

    with pytest.raises(ValueError, match="price_frames is empty"):
        mp_engine.run(cfg, df=None, price_frames={})


def test_run_price_frames_combines_and_sorts(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = DummyCfg()

    frame_a = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2020-01-31", "2020-02-29"]),
            "FundA": [0.01, 0.02],
        }
    )
    frame_b = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2020-02-29", "2020-03-31"]),
            "FundB": [0.03, 0.04],
        }
    )
    price_frames = {"a": frame_a, "b": frame_b}

    captured: dict[str, pd.DataFrame] = {}

    def fake_generate_periods(_cfg_dict: dict[str, object]) -> list[SimpleNamespace]:
        return [
            SimpleNamespace(
                in_start="2020-01-01",
                in_end="2020-02-01",
                out_start="2020-03-01",
                out_end="2020-04-01",
            )
        ]

    def fake_run_analysis(*args, **kwargs):
        captured["df"] = args[0]
        return {"summary": "ok"}

    monkeypatch.setattr(mp_engine, "generate_periods", fake_generate_periods)
    monkeypatch.setattr(mp_engine, "_run_analysis", fake_run_analysis)

    results = mp_engine.run(cfg, df=None, price_frames=price_frames)

    assert results and results[0]["period"] == (
        "2020-01-01",
        "2020-02-01",
        "2020-03-01",
        "2020-04-01",
    )

    combined = captured["df"]
    assert list(combined.columns) == ["Date", "FundA", "FundB"]
    assert combined["Date"].is_monotonic_increasing

    dup_row = combined.loc[combined["Date"] == pd.Timestamp("2020-02-29")]
    assert dup_row.shape[0] == 1
    # Latest observation should win when de-duplicating on Date.
    assert pd.isna(dup_row["FundA"].iloc[0])
    assert dup_row["FundB"].iloc[0] == pytest.approx(0.03)


def test_run_price_frames_overrides_dataframe(monkeypatch: pytest.MonkeyPatch) -> None:
    """Provided price frames should take precedence over ``df`` inputs."""

    cfg = DummyCfg()

    existing_df = pd.DataFrame(
        {"Date": pd.to_datetime(["2020-01-31"]), "FundA": [0.99], "FundB": [0.88]}
    )
    frame_custom = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2020-01-31", "2020-02-29"]),
            "FundA": [0.01, 0.02],
            "FundB": [0.03, 0.04],
        }
    )
    price_frames = {"primary": frame_custom}

    captured: dict[str, pd.DataFrame] = {}

    def fake_generate_periods(_cfg: dict[str, object]) -> list[SimpleNamespace]:
        return [
            SimpleNamespace(
                in_start="2020-01-01",
                in_end="2020-02-01",
                out_start="2020-03-01",
                out_end="2020-04-01",
            )
        ]

    def fake_run_analysis(*args, **kwargs):
        captured["df"] = args[0]
        return {"summary": "ok"}

    monkeypatch.setattr(mp_engine, "generate_periods", fake_generate_periods)
    monkeypatch.setattr(mp_engine, "_run_analysis", fake_run_analysis)

    mp_engine.run(cfg, df=existing_df, price_frames=price_frames)

    assert "df" in captured
    used_df = captured["df"].reset_index(drop=True)
    expected = frame_custom.sort_values("Date").reset_index(drop=True)
    pd.testing.assert_frame_equal(used_df, expected)
