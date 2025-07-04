from __future__ import annotations

import pandas as pd
import numpy as np


class BaseWeighting:
    """Base interface for weighting schemes."""

    def weight(self, selected: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError


class EqualWeight(BaseWeighting):
    """Simple equal weighting."""

    def weight(self, selected: pd.DataFrame) -> pd.DataFrame:
        if selected.empty:
            return pd.DataFrame(columns=["weight"])
        n = len(selected)
        w = np.repeat(1.0 / n, n)
        w = (w * 10000).round() / 10000
        return pd.DataFrame({"weight": w}, index=selected.index)


class ScorePropSimple(BaseWeighting):
    """Weights proportional to positive scores."""

    def __init__(self, column: str = "Sharpe") -> None:
        self.column = column

    def weight(self, selected: pd.DataFrame) -> pd.DataFrame:
        if selected.empty:
            return pd.DataFrame(columns=["weight"])
        if self.column not in selected.columns:
            raise KeyError(self.column)
        scores = selected[self.column].clip(lower=0).astype(float)
        if scores.sum() == 0:
            return EqualWeight().weight(selected)
        w = scores / scores.sum()
        return pd.DataFrame({"weight": w}, index=selected.index)


class ScorePropBayesian(BaseWeighting):
    """Score-proportional weighting with Bayesian shrinkage."""

    def __init__(self, column: str = "Sharpe", *, shrink_tau: float = 0.25) -> None:
        self.column = column
        self.tau = float(shrink_tau)

    def weight(self, selected: pd.DataFrame) -> pd.DataFrame:
        if selected.empty:
            return pd.DataFrame(columns=["weight"])
        if self.column not in selected.columns:
            raise KeyError(self.column)
        scores = selected[self.column].astype(float)
        mean = scores.mean()
        shrunk = (scores + self.tau * mean) / (1 + self.tau)
        shrunk = shrunk.clip(lower=0)
        if shrunk.sum() == 0:
            return EqualWeight().weight(selected)
        w = shrunk / shrunk.sum()
        return pd.DataFrame({"weight": w}, index=selected.index)
