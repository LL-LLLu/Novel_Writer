import pytest
from pathlib import Path
import tempfile
from pydantic import ValidationError

from novel_writer.config import Config, DataConfig, TrainingConfig

def test_default_config():
    config = Config()
    assert config.data.chunk_size == 4000
    assert config.training.epochs == 1
    assert config.model.lora_rank == 32

def test_config_from_yaml(tmp_path):
    config_file = tmp_path / "config.yaml"
    config_file.write_text("""
data:
  input_dir: custom/raw
  chunk_size: 2000
training:
  epochs: 5
""")

    config = Config.from_yaml(config_file)
    assert str(config.data.input_dir) == "custom/raw"
    assert config.data.chunk_size == 2000
    assert config.training.epochs == 5

def test_config_validation():
    with pytest.raises(ValidationError):
        Config(
            data=DataConfig(chunk_size=0)  # Must be > 0
        )

def test_config_to_yaml(tmp_path):
    config = Config(data=DataConfig(chunk_size=3000))
    output_file = tmp_path / "output.yaml"

    config.to_yaml(output_file)

    assert output_file.exists()
    loaded_config = Config.from_yaml(output_file)
    assert loaded_config.data.chunk_size == 3000
