import sys
import pathlib
import yaml  # type: ignore[import-untyped]

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))  # noqa: E402

from trend_analysis import config  # noqa: E402


def test_load_defaults():
    cfg = config.load()
    with open(config.DEFAULTS, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    assert cfg.version == data.get("version")
    assert "data" in cfg.model_dump()


def test_default_constants(monkeypatch):
    """Test that default constants are defined in run modules and have expected values."""
    # Import the modules that use the constants
    import sys
    from pathlib import Path
    
    # Add src to path to import modules directly using monkeypatch
    src_path = str(Path(__file__).resolve().parents[1] / "src")
    monkeypatch.syspath_prepend(src_path)
    
    from trend_analysis import run_analysis, run_multi_analysis
    
    # Check run_analysis constants
    assert hasattr(run_analysis, 'DEFAULT_OUTPUT_DIRECTORY')
    assert hasattr(run_analysis, 'DEFAULT_OUTPUT_FORMATS')
    assert run_analysis.DEFAULT_OUTPUT_DIRECTORY == "outputs"
    assert run_analysis.DEFAULT_OUTPUT_FORMATS == ["excel"]
    
    # Check run_multi_analysis constants
    assert hasattr(run_multi_analysis, 'DEFAULT_OUTPUT_DIRECTORY')
    assert hasattr(run_multi_analysis, 'DEFAULT_OUTPUT_FORMATS')
    assert run_multi_analysis.DEFAULT_OUTPUT_DIRECTORY == "outputs"
    assert run_multi_analysis.DEFAULT_OUTPUT_FORMATS == ["excel"]
    
    # Verify they are the correct types
    assert isinstance(run_analysis.DEFAULT_OUTPUT_DIRECTORY, str)
    assert isinstance(run_analysis.DEFAULT_OUTPUT_FORMATS, list)
    assert len(run_analysis.DEFAULT_OUTPUT_FORMATS) == 1
    assert run_analysis.DEFAULT_OUTPUT_FORMATS[0] == "excel"
