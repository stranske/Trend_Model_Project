import importlib

def test_indicator_import():
    mod = importlib.import_module('trend_analysis.core.indicator')
    assert hasattr(mod, '__all__')
