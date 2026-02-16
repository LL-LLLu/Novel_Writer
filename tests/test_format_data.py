import pytest
import json
from pathlib import Path

from novel_writer.processing.format import create_chunks, format_dataset
from novel_writer.config import Config, DataConfig


def test_create_chunks():
    text = "abcdefghij"
    chunks = create_chunks(text, chunk_size=4, overlap=2)
    expected = ["abcd", "cdef", "efgh", "ghij", "ij"]
    assert chunks == expected


def test_create_chunks_no_overlap():
    text = "abcdefgh"
    chunks = create_chunks(text, chunk_size=4, overlap=0)
    assert chunks == ["abcd", "efgh"]


def test_create_chunks_full_overlap():
    text = "abcdef"
    chunks = create_chunks(text, chunk_size=3, overlap=2)
    # step = chunk_size - overlap = 1
    # start=0 -> 'abc', start=1 -> 'bcd', start=2 -> 'cde',
    # start=3 -> 'def', start=4 -> 'ef', start=5 -> 'f'
    assert chunks == ["abc", "bcd", "cde", "def", "ef", "f"]


def test_format_dataset(tmp_path):
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    output_file = tmp_path / "output.jsonl"

    # Content must be >= 100 chars to not be skipped
    text = "Hello world. This is a test content that is long enough. " * 5
    (input_dir / "test.txt").write_text(text, encoding="utf-8")

    config = Config(data=DataConfig(chunk_size=4000, overlap=0))
    num = format_dataset(input_dir, output_file, config)

    assert output_file.exists()
    assert num >= 1

    with open(output_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    assert len(lines) >= 1
    entry = json.loads(lines[0])
    assert entry["instruction"] == "Continue the narrative in the established style."
    assert "input" in entry
    assert len(entry["output"]) > 0


def test_format_dataset_skips_short_chunks(tmp_path):
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    output_file = tmp_path / "output.jsonl"

    # 50 chars of content with chunk_size=200 -> one chunk of ~50 chars, skipped (< 100)
    text = "Short text." * 5
    (input_dir / "test.txt").write_text(text, encoding="utf-8")

    config = Config(data=DataConfig(chunk_size=200, overlap=0))
    num = format_dataset(input_dir, output_file, config)

    assert num == 0
