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


def test_config_attribute_initialization():
    """Test that Config class initializes all declared attributes even with minimal data."""
    # Create a Config with minimal data (not all attributes provided)
    minimal_data = {
        'version': '1.0',
        'data': {'csv_path': 'test.csv'},
        'run': {'mode': 'test'}
    }
    
    # This should not raise any exceptions
    cfg = config.Config(**minimal_data)
    
    # All declared attributes should be accessible without AttributeError
    assert cfg.version == '1.0'
    assert cfg.data == {'csv_path': 'test.csv'}
    assert cfg.run == {'mode': 'test'}
    
    # Attributes not provided should have default values (empty dicts)
    assert cfg.preprocessing == {}
    assert cfg.vol_adjust == {}
    assert cfg.sample_split == {}
    assert cfg.portfolio == {}
    assert cfg.metrics == {}
    assert cfg.export == {}
    
    # Attributes with explicit defaults should keep their defaults
    assert cfg.benchmarks == {}
    assert cfg.output is None
    assert cfg.multi_period is None
    assert cfg.jobs is None
    assert cfg.checkpoint_dir is None
    assert cfg.random_seed is None
