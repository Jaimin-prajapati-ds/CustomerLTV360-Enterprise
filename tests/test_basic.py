
import pytest
import pandas as pd
from src.utils.config import load_config
from pathlib import Path

def test_load_config():
    config = load_config("configs/config.yaml")
    assert isinstance(config, dict)
    assert 'data' in config
    assert 'models' in config

def test_config_paths():
    config = load_config("configs/config.yaml")
    assert Path(config['data']['raw_path']).parent.name == 'raw'
