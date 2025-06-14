import importlib.util
import pathlib
import pytest

# Dynamically load the cleanup module from its file path
module_path = pathlib.Path(__file__).resolve().parents[1] / 'Old' / 'Vol_Adj_Trend_Analysis_Cleanup.py'
spec = importlib.util.spec_from_file_location('cleanup', module_path)
cleanup = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cleanup)


def test_run_analysis_returns_none():
    assert cleanup.run_analysis(None, None, None, None, None, None, None) is None


def test_prepare_weights_missing():
    assert not hasattr(cleanup, 'prepare_weights')
