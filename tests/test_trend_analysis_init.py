import importlib


def test_trend_analysis_init_exposes_exports():
    import trend_analysis

    module = importlib.reload(trend_analysis)

    assert hasattr(module, "load_csv")
    assert hasattr(module, "export_to_csv")
    assert hasattr(module, "export_to_excel")

