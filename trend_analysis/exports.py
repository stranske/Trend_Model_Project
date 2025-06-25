"""Temporary re-export of ``trend_analysis.export`` for backward compatibility."""

from trend_analysis.export import *  # noqa: F401,F403

__all__ = [
    "export_to_excel",
    "export_to_csv",
    "export_to_json",
    "export_data",
]
