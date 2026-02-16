import json
import pytest
from pathlib import Path
from unittest.mock import patch

from click.testing import CliRunner

from novel_writer.processing.preference import (
    PreferencePair,
    generate_preference_pairs,
    score_text,
)


# ---------------------------------------------------------------------------
# 1. PreferencePair model validation
# ---------------------------------------------------------------------------

def test_preference_pair_model_validation():
    pair = PreferencePair(
        prompt="Write a story.",
        chosen="A great story with rich details.",
        rejected="bad",
        chosen_score=0.9,
        rejected_score=0.3,
    )
    assert pair.prompt == "Write a story."
    assert pair.chosen == "A great story with rich details."
    assert pair.rejected == "bad"
    assert pair.chosen_score == 0.9
    assert pair.rejected_score == 0.3


def test_preference_pair_model_dump():
    pair = PreferencePair(
        prompt="prompt",
        chosen="chosen",
        rejected="rejected",
        chosen_score=0.8,
        rejected_score=0.2,
    )
    data = pair.model_dump()
    assert set(data.keys()) == {"prompt", "chosen", "rejected", "chosen_score", "rejected_score"}


# ---------------------------------------------------------------------------
# 2. score_text returns a float (mock evaluate_text)
# ---------------------------------------------------------------------------

def test_score_text_returns_float():
    with patch("novel_writer.evaluation.metrics.evaluate_text") as mock_eval:
        from novel_writer.evaluation.metrics import EvaluationResult

        mock_eval.return_value = EvaluationResult(
            repetition_rate=0.1,
            vocabulary_diversity=0.7,
            avg_sentence_length=12.0,
            overall_score=0.75,
        )
        result = score_text("Some sample text for scoring.")
        assert isinstance(result, float)
        assert result == 0.75
        mock_eval.assert_called_once_with("Some sample text for scoring.")


# ---------------------------------------------------------------------------
# Helper: deterministic mock score based on text length
# ---------------------------------------------------------------------------

def _mock_score(text: str) -> float:
    """Return a deterministic score based on text length (longer = higher)."""
    return min(len(text) / 100.0, 1.0)


def _write_sample_jsonl(path: Path, entries: list[dict]):
    with open(path, "w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")


# ---------------------------------------------------------------------------
# 3. generate_preference_pairs with sample JSONL
# ---------------------------------------------------------------------------

@patch("novel_writer.processing.preference.score_text", side_effect=_mock_score)
def test_generate_preference_pairs_basic(mock_score, tmp_path):
    input_path = tmp_path / "input.jsonl"
    output_path = tmp_path / "output.jsonl"

    entries = [
        {"instruction": "Write a scene.", "output": "A" * 80},   # score 0.8
        {"instruction": "Write a scene.", "output": "B" * 30},   # score 0.3
        {"instruction": "Write a scene.", "output": "C" * 50},   # score 0.5
    ]
    _write_sample_jsonl(input_path, entries)

    num_pairs = generate_preference_pairs(
        input_path=input_path,
        output_path=output_path,
        min_score_diff=0.1,
    )

    assert num_pairs > 0
    assert output_path.exists()

    # Verify output lines are valid JSON
    with open(output_path, "r") as f:
        lines = [line.strip() for line in f if line.strip()]
    assert len(lines) == num_pairs

    for line in lines:
        data = json.loads(line)
        assert data["chosen_score"] > data["rejected_score"]


# ---------------------------------------------------------------------------
# 4. generate_preference_pairs respects min_score_diff
# ---------------------------------------------------------------------------

@patch("novel_writer.processing.preference.score_text", side_effect=_mock_score)
def test_generate_preference_pairs_min_score_diff(mock_score, tmp_path):
    input_path = tmp_path / "input.jsonl"
    output_path = tmp_path / "output.jsonl"

    # Two entries with very similar scores (0.50 and 0.51)
    entries = [
        {"instruction": "Write.", "output": "X" * 50},  # score 0.50
        {"instruction": "Write.", "output": "Y" * 51},  # score 0.51
    ]
    _write_sample_jsonl(input_path, entries)

    # Require at least 0.1 diff -- should produce 0 pairs
    num_pairs = generate_preference_pairs(
        input_path=input_path,
        output_path=output_path,
        min_score_diff=0.1,
    )

    assert num_pairs == 0


@patch("novel_writer.processing.preference.score_text", side_effect=_mock_score)
def test_generate_preference_pairs_large_diff_filter(mock_score, tmp_path):
    input_path = tmp_path / "input.jsonl"
    output_path = tmp_path / "output.jsonl"

    entries = [
        {"instruction": "Write.", "output": "A" * 90},   # score 0.9
        {"instruction": "Write.", "output": "B" * 60},   # score 0.6
        {"instruction": "Write.", "output": "C" * 20},   # score 0.2
    ]
    _write_sample_jsonl(input_path, entries)

    # With min_score_diff=0.5, only (0.9 vs 0.2) qualifies as >= 0.5 diff;
    # (0.9 vs 0.6) = 0.3 diff -- not enough; (0.6 vs 0.2) = 0.4 -- not enough
    num_pairs = generate_preference_pairs(
        input_path=input_path,
        output_path=output_path,
        min_score_diff=0.5,
    )

    assert num_pairs == 1


# ---------------------------------------------------------------------------
# 5. generate_preference_pairs respects max_pairs limit
# ---------------------------------------------------------------------------

@patch("novel_writer.processing.preference.score_text", side_effect=_mock_score)
def test_generate_preference_pairs_max_pairs(mock_score, tmp_path):
    input_path = tmp_path / "input.jsonl"
    output_path = tmp_path / "output.jsonl"

    # Create enough entries to generate many pairs
    entries = [
        {"instruction": "Write.", "output": "A" * (10 + i * 20)}
        for i in range(10)
    ]
    _write_sample_jsonl(input_path, entries)

    num_pairs = generate_preference_pairs(
        input_path=input_path,
        output_path=output_path,
        min_score_diff=0.05,
        max_pairs=3,
    )

    assert num_pairs == 3


# ---------------------------------------------------------------------------
# 6. Output file format (each line is valid JSON with correct keys)
# ---------------------------------------------------------------------------

@patch("novel_writer.processing.preference.score_text", side_effect=_mock_score)
def test_generate_preference_pairs_output_format(mock_score, tmp_path):
    input_path = tmp_path / "input.jsonl"
    output_path = tmp_path / "output.jsonl"

    entries = [
        {"instruction": "Describe a forest.", "output": "T" * 70},
        {"instruction": "Describe a forest.", "output": "S" * 20},
    ]
    _write_sample_jsonl(input_path, entries)

    generate_preference_pairs(
        input_path=input_path,
        output_path=output_path,
        min_score_diff=0.1,
    )

    expected_keys = {"prompt", "chosen", "rejected", "chosen_score", "rejected_score"}
    with open(output_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            assert set(data.keys()) == expected_keys
            assert isinstance(data["chosen_score"], float)
            assert isinstance(data["rejected_score"], float)
            assert data["chosen_score"] > data["rejected_score"]


# ---------------------------------------------------------------------------
# 7. generate_preference_pairs with empty input
# ---------------------------------------------------------------------------

@patch("novel_writer.processing.preference.score_text", side_effect=_mock_score)
def test_generate_preference_pairs_empty_input(mock_score, tmp_path):
    input_path = tmp_path / "input.jsonl"
    output_path = tmp_path / "output.jsonl"

    # Write empty file
    input_path.write_text("")

    num_pairs = generate_preference_pairs(
        input_path=input_path,
        output_path=output_path,
    )

    assert num_pairs == 0
    assert output_path.exists()


# ---------------------------------------------------------------------------
# 8. preference CLI command exists in help
# ---------------------------------------------------------------------------

def test_preference_cli_command_exists():
    from novel_writer.cli import cli

    runner = CliRunner()
    result = runner.invoke(cli, ["preference", "--help"])
    assert result.exit_code == 0
    assert "Generate DPO preference pairs" in result.output
    assert "--input" in result.output
    assert "--output" in result.output
    assert "--min-diff" in result.output
    assert "--max-pairs" in result.output
