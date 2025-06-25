"""Trend analysis package."""

from . import metrics, config, data, pipeline
from .data import load_csv, identify_risk_free_fund

__all__ = [
    "metrics",
    "config",
    "data",
    "pipeline",
    "load_csv",
    "identify_risk_free_fund",
]
