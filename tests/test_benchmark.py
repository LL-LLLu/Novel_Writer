"""Comprehensive tests for novel_writer.evaluation.benchmark module."""

import json
import pytest
from unittest.mock import patch

from click.testing import CliRunner

from novel_writer.evaluation.benchmark import (
    BenchmarkResult,
    BenchmarkReport,
    run_benchmark,
)
from novel_writer.evaluation.metrics import EvaluationResult
from novel_writer.evaluation.prompts import BENCHMARK_PROMPTS
from novel_writer.cli import cli


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_evaluation():
    """A sample EvaluationResult for use in tests."""
    return EvaluationResult(
        perplexity=None,
        repetition_rate=0.1,
        vocabulary_diversity=0.8,
        avg_sentence_length=12.5,
        coherence_score=None,
        overall_score=0.72,
    )


@pytest.fixture
def sample_benchmark_result(sample_evaluation):
    """A sample BenchmarkResult for use in tests."""
    return BenchmarkResult(
        prompt_id="continuation",
        prompt_name="Story Continuation",
        category="continuation",
        generated_text="The door creaked open revealing a vast chamber.",
        evaluation=sample_evaluation,
        generation_time_seconds=1.234,
    )


@pytest.fixture
def mock_generator():
    """A mock generator function that returns deterministic text."""
    def generator_fn(prompt: str) -> str:
        return (
            "The ancient tome revealed secrets long forgotten. "
            "Shadows danced across the walls as the candle flickered. "
            "Sarah could barely contain her excitement at the discovery."
        )
    return generator_fn


@pytest.fixture
def runner():
    return CliRunner()


# ---------------------------------------------------------------------------
# BenchmarkResult model validation
# ---------------------------------------------------------------------------


class TestBenchmarkResult:
    """Tests for the BenchmarkResult model."""

    def test_valid_construction(self, sample_evaluation):
        """BenchmarkResult should accept valid data."""
        result = BenchmarkResult(
            prompt_id="test_id",
            prompt_name="Test Prompt",
            category="test",
            generated_text="Some generated text.",
            evaluation=sample_evaluation,
            generation_time_seconds=0.5,
        )
        assert result.prompt_id == "test_id"
        assert result.prompt_name == "Test Prompt"
        assert result.category == "test"
        assert result.generated_text == "Some generated text."
        assert result.evaluation == sample_evaluation
        assert result.generation_time_seconds == 0.5

    def test_serialization_roundtrip(self, sample_benchmark_result):
        """BenchmarkResult should serialize and deserialize correctly."""
        json_str = sample_benchmark_result.model_dump_json()
        restored = BenchmarkResult.model_validate_json(json_str)
        assert restored.prompt_id == sample_benchmark_result.prompt_id
        assert restored.evaluation.overall_score == sample_benchmark_result.evaluation.overall_score


# ---------------------------------------------------------------------------
# BenchmarkReport model validation
# ---------------------------------------------------------------------------


class TestBenchmarkReport:
    """Tests for the BenchmarkReport model."""

    def test_valid_construction(self, sample_benchmark_result):
        """BenchmarkReport should accept valid data."""
        report = BenchmarkReport(
            model_name="test-model",
            results=[sample_benchmark_result],
            avg_overall_score=0.72,
            avg_generation_time=1.234,
        )
        assert report.model_name == "test-model"
        assert len(report.results) == 1
        assert report.avg_overall_score == 0.72
        assert report.avg_generation_time == 1.234

    def test_empty_results(self):
        """BenchmarkReport should accept empty results list."""
        report = BenchmarkReport(
            model_name="empty-model",
            results=[],
            avg_overall_score=0.0,
            avg_generation_time=0.0,
        )
        assert len(report.results) == 0

    def test_serialization_roundtrip(self, sample_benchmark_result):
        """BenchmarkReport should serialize and deserialize correctly."""
        report = BenchmarkReport(
            model_name="test-model",
            results=[sample_benchmark_result],
            avg_overall_score=0.72,
            avg_generation_time=1.234,
        )
        json_str = report.model_dump_json()
        restored = BenchmarkReport.model_validate_json(json_str)
        assert restored.model_name == report.model_name
        assert len(restored.results) == len(report.results)


# ---------------------------------------------------------------------------
# BenchmarkReport.summary() tests
# ---------------------------------------------------------------------------


class TestBenchmarkReportSummary:
    """Tests for the BenchmarkReport.summary() method."""

    def test_summary_returns_string(self, sample_benchmark_result):
        """summary() should return a string."""
        report = BenchmarkReport(
            model_name="test-model",
            results=[sample_benchmark_result],
            avg_overall_score=0.72,
            avg_generation_time=1.234,
        )
        summary = report.summary()
        assert isinstance(summary, str)

    def test_summary_contains_model_name(self, sample_benchmark_result):
        """summary() should include the model name."""
        report = BenchmarkReport(
            model_name="my-special-model",
            results=[sample_benchmark_result],
            avg_overall_score=0.72,
            avg_generation_time=1.234,
        )
        summary = report.summary()
        assert "my-special-model" in summary

    def test_summary_contains_scores(self, sample_benchmark_result):
        """summary() should include average score and per-prompt scores."""
        report = BenchmarkReport(
            model_name="test-model",
            results=[sample_benchmark_result],
            avg_overall_score=0.72,
            avg_generation_time=1.234,
        )
        summary = report.summary()
        assert "0.720" in summary  # avg_overall_score formatted to 3 decimals
        assert "Story Continuation" in summary
        assert "[continuation]" in summary

    def test_summary_contains_prompt_count(self, sample_benchmark_result):
        """summary() should show the number of prompts evaluated."""
        report = BenchmarkReport(
            model_name="test-model",
            results=[sample_benchmark_result],
            avg_overall_score=0.72,
            avg_generation_time=1.234,
        )
        summary = report.summary()
        assert "Prompts evaluated: 1" in summary

    def test_summary_empty_results(self):
        """summary() should work with no results."""
        report = BenchmarkReport(
            model_name="empty-model",
            results=[],
            avg_overall_score=0.0,
            avg_generation_time=0.0,
        )
        summary = report.summary()
        assert "Prompts evaluated: 0" in summary


# ---------------------------------------------------------------------------
# run_benchmark tests
# ---------------------------------------------------------------------------


class TestRunBenchmark:
    """Tests for the run_benchmark function."""

    def test_with_mock_generator(self, mock_generator):
        """run_benchmark should produce a BenchmarkReport with default prompts."""
        with patch(
            "novel_writer.evaluation.benchmark.evaluate_text"
        ) as mock_eval:
            mock_eval.return_value = EvaluationResult(
                perplexity=None,
                repetition_rate=0.05,
                vocabulary_diversity=0.85,
                avg_sentence_length=15.0,
                coherence_score=None,
                overall_score=0.80,
            )

            report = run_benchmark(
                generator_fn=mock_generator,
                model_name="mock-model",
            )

        assert isinstance(report, BenchmarkReport)
        assert report.model_name == "mock-model"
        assert len(report.results) == len(BENCHMARK_PROMPTS)
        assert all(isinstance(r, BenchmarkResult) for r in report.results)

    def test_with_custom_prompts(self, mock_generator):
        """run_benchmark should accept and use custom prompts."""
        custom_prompts = [
            {
                "id": "custom1",
                "name": "Custom Prompt One",
                "prompt": "Write something interesting:\n\n",
                "category": "custom",
            },
        ]

        with patch(
            "novel_writer.evaluation.benchmark.evaluate_text"
        ) as mock_eval:
            mock_eval.return_value = EvaluationResult(
                perplexity=None,
                repetition_rate=0.1,
                vocabulary_diversity=0.75,
                avg_sentence_length=10.0,
                coherence_score=None,
                overall_score=0.70,
            )

            report = run_benchmark(
                generator_fn=mock_generator,
                model_name="custom-model",
                prompts=custom_prompts,
            )

        assert len(report.results) == 1
        assert report.results[0].prompt_id == "custom1"
        assert report.results[0].category == "custom"

    def test_saves_json_output(self, mock_generator, tmp_path):
        """run_benchmark should save JSON results when output_path is given."""
        output_file = tmp_path / "results" / "benchmark.json"

        with patch(
            "novel_writer.evaluation.benchmark.evaluate_text"
        ) as mock_eval:
            mock_eval.return_value = EvaluationResult(
                perplexity=None,
                repetition_rate=0.05,
                vocabulary_diversity=0.85,
                avg_sentence_length=15.0,
                coherence_score=None,
                overall_score=0.80,
            )

            report = run_benchmark(
                generator_fn=mock_generator,
                model_name="saved-model",
                output_path=output_file,
            )

        assert output_file.exists()

        # Validate the saved JSON is parseable and contains expected data
        with open(output_file) as f:
            data = json.load(f)
        assert data["model_name"] == "saved-model"
        assert len(data["results"]) == len(BENCHMARK_PROMPTS)

    def test_with_empty_prompts(self, mock_generator):
        """run_benchmark with empty prompts list should return empty report."""
        report = run_benchmark(
            generator_fn=mock_generator,
            model_name="empty-test",
            prompts=[],
        )

        assert isinstance(report, BenchmarkReport)
        assert len(report.results) == 0
        assert report.avg_overall_score == 0.0
        assert report.avg_generation_time == 0.0

    def test_generation_time_recorded(self, mock_generator):
        """Each result should have a non-negative generation time."""
        custom_prompts = [
            {
                "id": "time_test",
                "name": "Time Test",
                "prompt": "Test prompt.\n\n",
                "category": "test",
            },
        ]

        with patch(
            "novel_writer.evaluation.benchmark.evaluate_text"
        ) as mock_eval:
            mock_eval.return_value = EvaluationResult(
                perplexity=None,
                repetition_rate=0.0,
                vocabulary_diversity=1.0,
                avg_sentence_length=10.0,
                coherence_score=None,
                overall_score=0.85,
            )

            report = run_benchmark(
                generator_fn=mock_generator,
                model_name="time-model",
                prompts=custom_prompts,
            )

        assert report.results[0].generation_time_seconds >= 0.0

    def test_average_score_calculation(self):
        """Average scores should be correctly calculated across results."""
        def gen_fn(prompt):
            return "Generated text for testing averages."

        custom_prompts = [
            {"id": "p1", "name": "Prompt 1", "prompt": "First.\n\n", "category": "cat1"},
            {"id": "p2", "name": "Prompt 2", "prompt": "Second.\n\n", "category": "cat2"},
        ]

        scores = [0.6, 0.8]
        call_count = [0]

        def mock_evaluate(text):
            idx = call_count[0]
            call_count[0] += 1
            return EvaluationResult(
                perplexity=None,
                repetition_rate=0.1,
                vocabulary_diversity=0.8,
                avg_sentence_length=10.0,
                coherence_score=None,
                overall_score=scores[idx],
            )

        with patch(
            "novel_writer.evaluation.benchmark.evaluate_text",
            side_effect=mock_evaluate,
        ):
            report = run_benchmark(
                generator_fn=gen_fn,
                model_name="avg-model",
                prompts=custom_prompts,
            )

        assert report.avg_overall_score == pytest.approx(0.7, abs=0.001)


# ---------------------------------------------------------------------------
# BENCHMARK_PROMPTS structure tests
# ---------------------------------------------------------------------------


class TestBenchmarkPrompts:
    """Tests for the BENCHMARK_PROMPTS data structure."""

    def test_is_nonempty_list(self):
        """BENCHMARK_PROMPTS should be a non-empty list."""
        assert isinstance(BENCHMARK_PROMPTS, list)
        assert len(BENCHMARK_PROMPTS) > 0

    def test_all_have_required_keys(self):
        """Every prompt should have id, name, prompt, and category keys."""
        required_keys = {"id", "name", "prompt", "category"}
        for prompt_data in BENCHMARK_PROMPTS:
            assert isinstance(prompt_data, dict)
            assert required_keys.issubset(prompt_data.keys()), (
                f"Prompt {prompt_data.get('id', '?')} missing keys: "
                f"{required_keys - prompt_data.keys()}"
            )

    def test_ids_are_unique(self):
        """All prompt IDs should be unique."""
        ids = [p["id"] for p in BENCHMARK_PROMPTS]
        assert len(ids) == len(set(ids))

    def test_values_are_strings(self):
        """All values in each prompt dict should be strings."""
        for prompt_data in BENCHMARK_PROMPTS:
            for key in ("id", "name", "prompt", "category"):
                assert isinstance(prompt_data[key], str)
                assert len(prompt_data[key]) > 0


# ---------------------------------------------------------------------------
# CLI evaluate command tests
# ---------------------------------------------------------------------------


class TestEvaluateCLI:
    """Tests for the evaluate CLI command."""

    def test_evaluate_command_in_help(self, runner):
        """The evaluate command should appear in the CLI help output."""
        result = runner.invoke(cli, ['--help'])
        assert result.exit_code == 0
        assert 'evaluate' in result.output
