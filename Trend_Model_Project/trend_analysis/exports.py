"""Temporary re-export of ``trend_analysis.export`` for backward compatibility."""  # pragma: no cover

# ruff: noqa: F401, F403, F405

from trend_analysis.export import *  # noqa: F401,F403,F405  # pragma: no cover

__all__ = [  # pragma: no cover - re-export shim
    "register_formatter_excel",
    "reset_formatters_excel",
    "make_summary_formatter",
    "export_to_excel",
    "export_to_csv",
    "export_to_json",
    "export_data",
]
