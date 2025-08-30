"""Test for Config class attribute initialization fix."""

import pytest
from trend_analysis.config import Config


def test_config_with_minimal_data():
    """Test that Config can be created with minimal data without AttributeError."""
    # This would previously fail with ValidationError due to missing required fields
    minimal_data = {"version": "1.0"}
    
    config = Config(**minimal_data)
    
    # Verify the provided data is accessible
    assert config.version == "1.0"
    
    # Verify that all declared attributes exist and can be accessed without AttributeError
    assert hasattr(config, 'data')
    assert hasattr(config, 'preprocessing') 
    assert hasattr(config, 'vol_adjust')
    assert hasattr(config, 'sample_split')
    assert hasattr(config, 'portfolio')
    assert hasattr(config, 'metrics')
    assert hasattr(config, 'export')
    assert hasattr(config, 'run')
    
    # Verify they have appropriate default values
    assert config.data == {}
    assert config.preprocessing == {}
    assert config.vol_adjust == {}
    assert config.sample_split == {}
    assert config.portfolio == {}
    assert config.metrics == {}
    assert config.export == {}
    assert config.run == {}


def test_config_with_partial_data():
    """Test that Config works properly with partial data."""
    partial_data = {
        "version": "2.0",
        "data": {"some": "value"},
        "metrics": {"another": "value"}
    }
    
    config = Config(**partial_data)
    
    # Verify provided data is preserved
    assert config.version == "2.0"
    assert config.data == {"some": "value"}
    assert config.metrics == {"another": "value"}
    
    # Verify missing fields have defaults
    assert config.preprocessing == {}
    assert config.vol_adjust == {}
    assert config.sample_split == {}
    assert config.portfolio == {}
    assert config.export == {}
    assert config.run == {}


def test_config_with_empty_data():
    """Test that Config can be created with no data at all."""
    config = Config()
    
    # All attributes should exist with default values
    assert config.version == "0.1.0"  # Default version
    assert config.data == {}
    assert config.preprocessing == {}
    assert config.vol_adjust == {}
    assert config.sample_split == {}
    assert config.portfolio == {}
    assert config.benchmarks == {}
    assert config.metrics == {}
    assert config.export == {}
    assert config.output is None
    assert config.run == {}
    assert config.multi_period is None
    assert config.jobs is None
    assert config.checkpoint_dir is None
    assert config.random_seed is None


def test_config_attribute_access_no_errors():
    """Test that accessing any declared attribute never raises AttributeError."""
    config = Config(version="test")
    
    # Test accessing each declared attribute
    declared_attrs = [
        'version', 'data', 'preprocessing', 'vol_adjust', 'sample_split',
        'portfolio', 'benchmarks', 'metrics', 'export', 'output', 'run',
        'multi_period', 'jobs', 'checkpoint_dir', 'random_seed'
    ]
    
    for attr in declared_attrs:
        # This should never raise AttributeError
        value = getattr(config, attr)
        assert value is not None or attr in ['output', 'multi_period', 'jobs', 'checkpoint_dir', 'random_seed']


def test_config_serialization_roundtrip():
    """Test that Config objects can be serialized and deserialized properly.""" 
    original_data = {
        "version": "test",
        "data": {"key": "value"},
        "portfolio": {"selection": "top"}
    }
    
    config1 = Config(**original_data)
    
    # Test model_dump
    dumped = config1.model_dump()
    config2 = Config(**dumped)
    
    assert config1.version == config2.version
    assert config1.data == config2.data
    assert config1.portfolio == config2.portfolio
    
    # Test model_dump_json
    json_str = config1.model_dump_json()
    assert isinstance(json_str, str)
    assert "test" in json_str