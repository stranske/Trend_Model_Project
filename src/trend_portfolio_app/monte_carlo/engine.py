from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from trend_analysis._typing import FloatArray


@dataclass
class ReturnModelConfig:
    kind: str = "block_bootstrap"
    block: int = 6
    seed: int = 123


class ReturnModel:
    def fit(self, panel: pd.DataFrame) -> None:
        raise NotImplementedError

    def sample(self, n_periods: int, n_paths: int) -> FloatArray:
        raise NotImplementedError


class BlockBootstrapModel(ReturnModel):
    def __init__(self, cfg: ReturnModelConfig):
        self.cfg = cfg
        # Panel of returns used for sampling
        self.panel: pd.DataFrame | None = None
        self.rng = np.random.default_rng(cfg.seed)

    def fit(self, panel: pd.DataFrame) -> None:
        self.panel = panel.dropna(how="all")

    def sample(self, n_periods: int, n_paths: int) -> FloatArray:
        if self.panel is None:
            raise RuntimeError("Model not fit.")
        vals = self.panel.values
        T, N = vals.shape
        B = self.cfg.block
        out: FloatArray = np.zeros((n_paths, n_periods, N))
        for p in range(n_paths):
            t = 0
            while t < n_periods:
                start = self.rng.integers(0, max(1, T - B))
                seg = vals[start : start + B]
                seg_len = min(B, n_periods - t)
                out[p, t : t + seg_len, :] = seg[:seg_len, :]
                t += seg_len
        return out
