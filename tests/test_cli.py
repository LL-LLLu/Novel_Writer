import pytest
from click.testing import CliRunner
from pathlib import Path
import tempfile
import json

from novel_writer.cli import cli

@pytest.fixture
def runner():
    return CliRunner()

@pytest.fixture
def sample_config(tmp_path):
    config_file = tmp_path / "config.yaml"
    config_file.write_text(f"""
data:
  input_dir: {tmp_path / "test_data" / "raw"}
  output_dir: {tmp_path / "test_data" / "processed"}
  temp_dir: {tmp_path / "test_data" / "temp"}
  chunk_size: 100
  overlap: 10
""")
    return config_file

def test_cli_help(runner):
    result = runner.invoke(cli, ['--help'])
    assert result.exit_code == 0
    assert 'clean' in result.output
    assert 'format' in result.output

def test_clean_command(runner, sample_config, tmp_path):
    raw_dir = tmp_path / "test_data" / "raw"
    raw_dir.mkdir(parents=True)

    # Must be >100 chars for validation and >500 chars after cleaning
    content = "This is a test novel chapter with interesting content. " * 20
    (raw_dir / "test.txt").write_text(content)

    result = runner.invoke(cli, ['-c', str(sample_config), 'clean'])
    assert result.exit_code == 0

def test_format_command(runner, sample_config, tmp_path):
    # Create test cleaned data
    temp_dir = tmp_path / "test_data" / "temp"
    temp_dir.mkdir(parents=True)
    (temp_dir / "test_cleaned.txt").write_text("Test chapter content. " * 50)

    result = runner.invoke(cli, ['-c', str(sample_config), 'format'])
    assert result.exit_code == 0

    # Check output file
    output_file = tmp_path / "test_data" / "processed" / "train.jsonl"
    assert output_file.exists()

    # Validate JSONL
    with open(output_file, 'r') as f:
        lines = f.readlines()
        assert len(lines) > 0
        entry = json.loads(lines[0])
        assert 'instruction' in entry
        assert 'output' in entry
