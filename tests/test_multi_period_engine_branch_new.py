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
        self.performance: dict[str, object] = {"enable_cache": False}
        self.seed: int = 0

    def model_dump(self) -> dict[str, object]:
        return {
            "multi_period": {
                "frequency": "M",
                "in_sample_len": 1,
                "out_sample_len": 1,
                "start": "2020-01",
                "end": "2020-02",
            },
            "portfolio": dict(self.portfolio),
            "vol_adjust": dict(self.vol_adjust),
        }


def test_run_uses_nan_policy_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = DummyCfg()
    cfg.data = {
        "csv_path": "dummy.csv",
        "nan_policy": "bfill",
        "nan_limit": 3,
    }

    captured: dict[str, object] = {}

    def fake_load_csv(
        csv_path: str,
        *,
        errors: str,
        missing_policy: str | None,
        missing_limit: int | None,
    ):
        captured["missing_policy"] = missing_policy
        captured["missing_limit"] = missing_limit
        return pd.DataFrame(
            {"Date": pd.to_datetime(["2020-01-31", "2020-02-29"]), "Alpha": [0.1, 0.2]}
        )

    def fake_apply_missing_policy(
        frame: pd.DataFrame, *, policy: str | None, limit: int | None
    ):
        captured["applied_policy"] = policy
        captured["applied_limit"] = limit
        return frame, {}

    monkeypatch.setattr(mp_engine, "load_csv", fake_load_csv)
    monkeypatch.setattr(mp_engine, "apply_missing_policy", fake_apply_missing_policy)
    monkeypatch.setattr(mp_engine, "generate_periods", lambda _cfg: [])

    results = mp_engine.run(cfg, df=None, price_frames=None)

    assert results == []
    assert captured["missing_policy"] == "bfill"
    assert captured["missing_limit"] == 3
    assert captured["applied_policy"] == "bfill"
    assert captured["applied_limit"] == 3


def test_run_skips_missing_policy_when_price_frames(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = DummyCfg()
    frame = pd.DataFrame(
        {"Date": pd.to_datetime(["2020-01-31", "2020-02-29"]), "Alpha": [0.1, 0.2]}
    )
    price_frames = {"2020-01": frame}

    called = False

    def fake_apply_missing_policy(*args, **kwargs):  # pragma: no cover - defensive
        nonlocal called
        called = True
        return args[0], {}

    monkeypatch.setattr(mp_engine, "apply_missing_policy", fake_apply_missing_policy)
    monkeypatch.setattr(mp_engine, "generate_periods", lambda _cfg: [])

    results = mp_engine.run(cfg, df=None, price_frames=price_frames)

    assert results == []
    assert called is False


def test_run_raises_when_policy_drops_all(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = DummyCfg()

    def fake_load_csv(*_args, **_kwargs):
        return pd.DataFrame({"Date": pd.to_datetime(["2020-01-31"]), "Alpha": [pd.NA]})

    def fake_apply_missing_policy(
        frame: pd.DataFrame, *, policy: str | None, limit: int | None
    ):
        empty = frame.copy()
        empty.iloc[:, :] = pd.NA
        return empty, {"status": "dropped"}

    monkeypatch.setattr(mp_engine, "load_csv", fake_load_csv)
    monkeypatch.setattr(mp_engine, "apply_missing_policy", fake_apply_missing_policy)
    monkeypatch.setattr(mp_engine, "generate_periods", lambda _cfg: [])

    with pytest.raises(ValueError, match="Missing-data policy removed all assets"):
        mp_engine.run(cfg, df=None, price_frames=None)


def test_apply_weight_bounds_reduces_excess_weight() -> None:
    weights = pd.Series({"A": 0.9, "B": 0.4, "C": 0.1})
    result = mp_engine._apply_weight_bounds(weights, 0.2, 0.6)

    assert pytest.approx(result.sum(), rel=0, abs=1e-12) == 1.0
    assert result.between(0.2 - 1e-12, 0.6 + 1e-12).all()
    assert set(result.index) == {"A", "B", "C"}


def test_apply_weight_bounds_fills_deficit() -> None:
    weights = pd.Series({"A": 0.05, "B": 0.05, "C": 0.05})
    result = mp_engine._apply_weight_bounds(weights, 0.1, 0.6)

    assert pytest.approx(result.sum(), rel=0, abs=1e-12) == 1.0
    assert result.between(0.1 - 1e-12, 0.6 + 1e-12).all()


def test_apply_weight_bounds_final_normalise_handles_excess(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(mp_engine, "NUMERICAL_TOLERANCE_HIGH", 0.2)
    weights = pd.Series({"A": 0.7, "B": 0.45})
    result = mp_engine._apply_weight_bounds(weights, 0.2, 0.7)

    assert pytest.approx(result.sum(), rel=0, abs=1e-12) == 1.0
    assert result.between(0.2 - 1e-12, 0.7 + 1e-12).all()


def test_apply_weight_bounds_final_normalise_handles_deficit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(mp_engine, "NUMERICAL_TOLERANCE_HIGH", 0.2)
    weights = pd.Series({"A": 0.2, "B": 0.3, "C": 0.3})
    result = mp_engine._apply_weight_bounds(weights, 0.2, 0.5)

    assert pytest.approx(result.sum(), rel=0, abs=1e-12) == 1.0
    assert result.between(0.2 - 1e-12, 0.5 + 1e-12).all()


def test_apply_weight_bounds_returns_empty_series() -> None:
    empty = pd.Series(dtype=float)
    result = mp_engine._apply_weight_bounds(empty, 0.2, 0.6)

    assert result is empty


def test_apply_weight_bounds_handles_zero_available_share(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(mp_engine, "NUMERICAL_TOLERANCE_HIGH", -1e-9)
    weights = pd.Series({"A": 0.3, "B": 0.3, "C": 0.3, "D": 0.3})
    result = mp_engine._apply_weight_bounds(weights, 0.3, 0.9)

    assert pytest.approx(result.sum(), rel=0, abs=1e-12) == 1.2
    assert result.between(0.3 - 1e-12, 0.9 + 1e-12).all()


def test_apply_weight_bounds_handles_zero_room_share(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(mp_engine, "NUMERICAL_TOLERANCE_HIGH", -1e-9)
    weights = pd.Series({"A": 0.4, "B": 0.4})
    result = mp_engine._apply_weight_bounds(weights, 0.1, 0.4)

    assert pytest.approx(result.sum(), rel=0, abs=1e-12) == 0.8
    assert result.between(0.1 - 1e-12, 0.4 + 1e-12).all()


def test_run_schedule_applies_rebalancer(monkeypatch: pytest.MonkeyPatch) -> None:
    score_frames = {
        "2020-01": pd.DataFrame(
            {"Sharpe": [1.0, 0.5]}, index=["Alpha One", "Alpha Two"]
        )
    }

    class StubSelector:
        def select(
            self, score_frame: pd.DataFrame
        ) -> tuple[pd.DataFrame, pd.DataFrame]:
            return score_frame, score_frame

    class StubWeighting:
        def weight(self, selected: pd.DataFrame, date: pd.Timestamp) -> pd.DataFrame:
            return selected.assign(weight=[0.6, 0.4])

    calls: list[pd.Series] = []

    class StubRebalancer:
        def __init__(self) -> None:
            pass

        def apply_triggers(
            self, prev_weights: pd.Series, sf: pd.DataFrame
        ) -> pd.Series:
            calls.append(prev_weights)
            return prev_weights

    portfolio = mp_engine.run_schedule(
        score_frames,
        StubSelector(),
        StubWeighting(),
        rebalancer=StubRebalancer(),
    )

    assert calls and isinstance(portfolio, mp_engine.Portfolio)


def test_run_schedule_applies_rebalance_strategies(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    score_frames = {
        "2020-01": pd.DataFrame({"Sharpe": [1.2, 0.8]}, index=["Alpha", "Beta"])
    }

    class StubSelector:
        def select(
            self, score_frame: pd.DataFrame
        ) -> tuple[pd.DataFrame, pd.DataFrame]:
            return score_frame, score_frame

    class StubWeighting:
        def weight(self, selected: pd.DataFrame, date: pd.Timestamp) -> pd.DataFrame:
            return selected.assign(weight=[0.7, 0.3])

    captured: dict[str, object] = {}

    def fake_apply_strategies(names, params, current, target, *, scores=None):
        captured["names"] = names
        captured["scores"] = scores
        captured["current"] = current
        return target * 0.5, 1.23

    monkeypatch.setattr(
        mp_engine, "apply_rebalancing_strategies", fake_apply_strategies
    )

    portfolio = mp_engine.run_schedule(
        score_frames,
        StubSelector(),
        StubWeighting(),
        rebalance_strategies=["turnover_cap"],
        rebalance_params={"turnover_cap": {"limit": 0.1}},
    )

    assert captured["names"] == ["turnover_cap"]
    assert isinstance(portfolio, mp_engine.Portfolio)
