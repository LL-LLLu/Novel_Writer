import json
import pytest
from pathlib import Path
from unittest.mock import patch

from click.testing import CliRunner

from novel_writer.processing.curriculum import (
    DifficultyScore,
    compute_difficulty,
    sort_by_curriculum,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_sample_jsonl(path: Path, entries: list[dict]):
    with open(path, "w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")


def _deterministic_difficulty(text: str):
    """Return a DifficultyScore based on text length for deterministic tests."""
    score = min(len(text) / 200.0, 1.0)
    return DifficultyScore(
        vocabulary_complexity=round(score, 4),
        sentence_complexity=round(score * 0.8, 4),
        overall_difficulty=round(score, 4),
    )


# ---------------------------------------------------------------------------
# 1. DifficultyScore model validation
# ---------------------------------------------------------------------------

def test_difficulty_score_model_validation():
    score = DifficultyScore(
        vocabulary_complexity=0.75,
        sentence_complexity=0.4,
        overall_difficulty=0.61,
    )
    assert score.vocabulary_complexity == 0.75
    assert score.sentence_complexity == 0.4
    assert score.overall_difficulty == 0.61


def test_difficulty_score_model_dump():
    score = DifficultyScore(
        vocabulary_complexity=0.5,
        sentence_complexity=0.3,
        overall_difficulty=0.42,
    )
    data = score.model_dump()
    assert set(data.keys()) == {"vocabulary_complexity", "sentence_complexity", "overall_difficulty"}


# ---------------------------------------------------------------------------
# 2. compute_difficulty returns DifficultyScore with fields in [0,1]
# ---------------------------------------------------------------------------

def test_compute_difficulty_returns_difficulty_score():
    text = "The cat sat on the mat. It was a fine day."
    result = compute_difficulty(text)
    assert isinstance(result, DifficultyScore)
    assert 0.0 <= result.vocabulary_complexity <= 1.0
    assert 0.0 <= result.sentence_complexity <= 1.0
    assert 0.0 <= result.overall_difficulty <= 1.0


# ---------------------------------------------------------------------------
# 3. compute_difficulty: complex text scores higher than simple text
# ---------------------------------------------------------------------------

def test_compute_difficulty_complex_higher_than_simple():
    simple = "The cat sat. The dog ran. The bird flew."
    complex_text = (
        "The extraordinarily magnificent protagonist, overwhelmed by unfathomable "
        "circumstances, meticulously deliberated upon the philosophical implications "
        "of existential paradigms while simultaneously contemplating the ephemeral "
        "nature of consciousness and metaphysical transcendence."
    )
    simple_score = compute_difficulty(simple)
    complex_score = compute_difficulty(complex_text)
    assert complex_score.overall_difficulty > simple_score.overall_difficulty


# ---------------------------------------------------------------------------
# 4. sort_by_curriculum basic sorting (output sorted easy-to-hard)
# ---------------------------------------------------------------------------

@patch("novel_writer.processing.curriculum.compute_difficulty", side_effect=_deterministic_difficulty)
def test_sort_by_curriculum_basic_sorting(mock_diff, tmp_path):
    input_path = tmp_path / "input.jsonl"
    output_path = tmp_path / "output.jsonl"

    entries = [
        {"output": "A" * 160},  # difficulty 0.8
        {"output": "B" * 40},   # difficulty 0.2
        {"output": "C" * 100},  # difficulty 0.5
    ]
    _write_sample_jsonl(input_path, entries)

    sort_by_curriculum(input_path=input_path, output_path=output_path)

    with open(output_path, "r") as f:
        lines = [json.loads(line) for line in f if line.strip()]

    # Default: easy-to-hard ordering
    difficulties = [line["_difficulty"] for line in lines]
    assert difficulties == sorted(difficulties), "Output should be sorted easy-to-hard"


# ---------------------------------------------------------------------------
# 5. sort_by_curriculum reverse order (hard-to-easy)
# ---------------------------------------------------------------------------

@patch("novel_writer.processing.curriculum.compute_difficulty", side_effect=_deterministic_difficulty)
def test_sort_by_curriculum_reverse(mock_diff, tmp_path):
    input_path = tmp_path / "input.jsonl"
    output_path = tmp_path / "output.jsonl"

    entries = [
        {"output": "A" * 160},  # difficulty 0.8
        {"output": "B" * 40},   # difficulty 0.2
        {"output": "C" * 100},  # difficulty 0.5
    ]
    _write_sample_jsonl(input_path, entries)

    sort_by_curriculum(input_path=input_path, output_path=output_path, reverse=True)

    with open(output_path, "r") as f:
        lines = [json.loads(line) for line in f if line.strip()]

    difficulties = [line["_difficulty"] for line in lines]
    assert difficulties == sorted(difficulties, reverse=True), "Output should be sorted hard-to-easy"


# ---------------------------------------------------------------------------
# 6. sort_by_curriculum output contains _difficulty field
# ---------------------------------------------------------------------------

@patch("novel_writer.processing.curriculum.compute_difficulty", side_effect=_deterministic_difficulty)
def test_sort_by_curriculum_output_has_difficulty_field(mock_diff, tmp_path):
    input_path = tmp_path / "input.jsonl"
    output_path = tmp_path / "output.jsonl"

    entries = [
        {"output": "A" * 80},
        {"output": "B" * 120},
    ]
    _write_sample_jsonl(input_path, entries)

    sort_by_curriculum(input_path=input_path, output_path=output_path)

    with open(output_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            assert "_difficulty" in data
            assert isinstance(data["_difficulty"], float)


# ---------------------------------------------------------------------------
# 7. sort_by_curriculum statistics (total, avg, min, max, bucket_distribution)
# ---------------------------------------------------------------------------

@patch("novel_writer.processing.curriculum.compute_difficulty", side_effect=_deterministic_difficulty)
def test_sort_by_curriculum_statistics(mock_diff, tmp_path):
    input_path = tmp_path / "input.jsonl"
    output_path = tmp_path / "output.jsonl"

    entries = [
        {"output": "A" * 160},  # difficulty 0.8
        {"output": "B" * 40},   # difficulty 0.2
        {"output": "C" * 100},  # difficulty 0.5
    ]
    _write_sample_jsonl(input_path, entries)

    stats = sort_by_curriculum(input_path=input_path, output_path=output_path, num_buckets=3)

    assert stats["total"] == 3
    assert "avg_difficulty" in stats
    assert "min_difficulty" in stats
    assert "max_difficulty" in stats
    assert stats["min_difficulty"] <= stats["avg_difficulty"] <= stats["max_difficulty"]
    assert stats["sorted_order"] == "easy-to-hard"
    assert "bucket_distribution" in stats
    assert len(stats["bucket_distribution"]) == 3
    # All entries accounted for
    assert sum(stats["bucket_distribution"].values()) == 3


# ---------------------------------------------------------------------------
# 8. sort_by_curriculum with empty input
# ---------------------------------------------------------------------------

@patch("novel_writer.processing.curriculum.compute_difficulty", side_effect=_deterministic_difficulty)
def test_sort_by_curriculum_empty_input(mock_diff, tmp_path):
    input_path = tmp_path / "input.jsonl"
    output_path = tmp_path / "output.jsonl"

    # Write empty file
    input_path.write_text("")

    stats = sort_by_curriculum(input_path=input_path, output_path=output_path)

    assert stats["total"] == 0
    assert stats["avg_difficulty"] == 0.0
    assert stats["min_difficulty"] == 0.0
    assert stats["max_difficulty"] == 0.0
    assert output_path.exists()


# ---------------------------------------------------------------------------
# 9. curriculum CLI command exists in help
# ---------------------------------------------------------------------------

def test_curriculum_cli_command_exists():
    from novel_writer.cli import cli

    runner = CliRunner()
    result = runner.invoke(cli, ["curriculum", "--help"])
    assert result.exit_code == 0
    assert "Sort training data by difficulty" in result.output
    assert "--input" in result.output
    assert "--output" in result.output
    assert "--reverse" in result.output
    assert "--buckets" in result.output
