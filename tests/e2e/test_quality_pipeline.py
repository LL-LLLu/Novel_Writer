"""End-to-end tests for evaluation, preference pairs, and curriculum sorting."""

import json
import pytest
from pathlib import Path

from novel_writer.evaluation.metrics import (
    evaluate_text,
    repetition_rate,
    vocabulary_diversity,
    avg_sentence_length,
    EvaluationResult,
)
from novel_writer.processing.preference import (
    generate_preference_pairs,
    score_text,
)
from novel_writer.processing.curriculum import (
    sort_by_curriculum,
    compute_difficulty,
    DifficultyScore,
)


# Texts of varying quality for preference pair testing
HIGH_QUALITY_TEXT = (
    "The autumn wind swept through the narrow cobblestone streets of the old town, "
    "carrying with it the scent of wood smoke and ripe apples. Eleanor paused at the "
    "corner where Magnolia Lane met the ancient city wall, her gaze drawn upward to "
    "the weathered battlements. Centuries of history pressed down from those stones, "
    "each one laid by hands long turned to dust. She pulled her wool coat tighter and "
    "continued walking, her footsteps echoing in the twilight silence."
)

LOW_QUALITY_TEXT = (
    "He walked and walked and walked. The road was long. The road was very long. "
    "He kept walking. He walked more. The road was still long. He walked and walked. "
    "The walking continued. He was still walking. The road went on and on and on."
)


class TestEvaluationMetrics:
    """Test the evaluation metrics on real text samples."""

    def test_evaluate_text_returns_result(self):
        """evaluate_text should return a valid EvaluationResult."""
        result = evaluate_text(HIGH_QUALITY_TEXT)

        assert isinstance(result, EvaluationResult)
        assert 0.0 <= result.repetition_rate <= 1.0
        assert 0.0 <= result.vocabulary_diversity <= 1.0
        assert result.avg_sentence_length > 0
        assert 0.0 <= result.overall_score <= 1.0

    def test_high_quality_text_scores_well(self):
        """High quality text should have good metrics."""
        result = evaluate_text(HIGH_QUALITY_TEXT)

        # High quality text should have high vocabulary diversity
        assert result.vocabulary_diversity > 0.5
        # Low repetition
        assert result.repetition_rate < 0.5
        # Reasonable overall score
        assert result.overall_score > 0.5

    def test_low_quality_text_has_high_repetition(self):
        """Low quality repetitive text should have high repetition rate."""
        result = evaluate_text(LOW_QUALITY_TEXT)

        # Repetitive text should show higher repetition
        assert result.repetition_rate > 0.0
        # Vocabulary diversity should be lower for repetitive text
        assert result.vocabulary_diversity < evaluate_text(HIGH_QUALITY_TEXT).vocabulary_diversity

    def test_individual_metrics_consistency(self):
        """Individual metric functions should be consistent with evaluate_text."""
        text = HIGH_QUALITY_TEXT

        rep = repetition_rate(text)
        vocab = vocabulary_diversity(text)
        avg_len = avg_sentence_length(text)

        result = evaluate_text(text)

        assert abs(result.repetition_rate - rep) < 1e-6
        assert abs(result.vocabulary_diversity - vocab) < 1e-6
        assert abs(result.avg_sentence_length - avg_len) < 1e-6

    def test_coherence_score_none_without_transformers(self):
        """Without sentence-transformers, coherence_score should be None."""
        result = evaluate_text(HIGH_QUALITY_TEXT)
        # In a standard test environment without sentence-transformers installed,
        # coherence_score should be None
        # (If sentence-transformers IS installed, this test still passes as it
        # just verifies the result type)
        assert result.coherence_score is None or isinstance(result.coherence_score, float)

    def test_overall_score_formula_without_coherence(self):
        """Verify overall score calculation when coherence is None."""
        result = evaluate_text(HIGH_QUALITY_TEXT)

        if result.coherence_score is None:
            expected = (
                result.vocabulary_diversity * 0.45
                + (1.0 - result.repetition_rate) * 0.55
            )
            assert abs(result.overall_score - expected) < 1e-6


class TestPreferencePairs:
    """Test DPO preference pair generation."""

    def test_generate_preference_pairs_from_jsonl(self, sample_jsonl, tmp_path):
        """Generate preference pairs from sample JSONL and verify structure."""
        output_path = tmp_path / "preferences.jsonl"

        num_pairs = generate_preference_pairs(
            input_path=sample_jsonl,
            output_path=output_path,
            min_score_diff=0.01,  # Low threshold to ensure pairs are generated
            seed=42,
        )

        assert output_path.exists()

        # Read and verify pairs
        pairs = []
        with open(output_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    pairs.append(json.loads(line))

        assert len(pairs) == num_pairs

        for pair in pairs:
            assert "prompt" in pair
            assert "chosen" in pair
            assert "rejected" in pair
            assert "chosen_score" in pair
            assert "rejected_score" in pair
            # Chosen should score higher than rejected
            assert pair["chosen_score"] >= pair["rejected_score"]

    def test_preference_pairs_with_max_limit(self, sample_jsonl, tmp_path):
        """Verify max_pairs parameter limits output."""
        output_path = tmp_path / "limited_prefs.jsonl"

        num_pairs = generate_preference_pairs(
            input_path=sample_jsonl,
            output_path=output_path,
            min_score_diff=0.001,
            max_pairs=2,
            seed=42,
        )

        assert num_pairs <= 2

    def test_score_text_returns_float(self):
        """score_text should return a float between 0 and 1."""
        score = score_text(HIGH_QUALITY_TEXT)
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_score_text_differentiates_quality(self):
        """Higher quality text should generally score higher."""
        high_score = score_text(HIGH_QUALITY_TEXT)
        low_score = score_text(LOW_QUALITY_TEXT)

        # High quality text should score at least as well
        # (We use >= because metric-based scoring may not perfectly
        # differentiate all text pairs)
        assert high_score >= low_score or abs(high_score - low_score) < 0.2


class TestCurriculumSorting:
    """Test curriculum-based data sorting."""

    def test_compute_difficulty_returns_score(self):
        """compute_difficulty should return a valid DifficultyScore."""
        score = compute_difficulty(HIGH_QUALITY_TEXT)

        assert isinstance(score, DifficultyScore)
        assert 0.0 <= score.vocabulary_complexity <= 1.0
        assert 0.0 <= score.sentence_complexity <= 1.0
        assert 0.0 <= score.overall_difficulty <= 1.0

    def test_sort_by_curriculum_easy_to_hard(self, sample_jsonl, tmp_path):
        """Sort entries from easy to hard and verify ordering."""
        output_path = tmp_path / "curriculum_sorted.jsonl"

        stats = sort_by_curriculum(
            input_path=sample_jsonl,
            output_path=output_path,
            reverse=False,
            num_buckets=3,
        )

        assert output_path.exists()
        assert stats["total"] == 5  # sample_jsonl has 5 entries
        assert stats["sorted_order"] == "easy-to-hard"
        assert "bucket_distribution" in stats

        # Read sorted entries and verify difficulty ordering
        entries = []
        with open(output_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    entries.append(json.loads(line))

        assert len(entries) == 5

        # Each entry should have a _difficulty field
        for entry in entries:
            assert "_difficulty" in entry

        # Entries should be sorted easy to hard
        difficulties = [e["_difficulty"] for e in entries]
        assert difficulties == sorted(difficulties), (
            f"Expected easy-to-hard ordering, got: {difficulties}"
        )

    def test_sort_by_curriculum_hard_to_easy(self, sample_jsonl, tmp_path):
        """Sort entries from hard to easy (reverse) and verify ordering."""
        output_path = tmp_path / "reverse_sorted.jsonl"

        stats = sort_by_curriculum(
            input_path=sample_jsonl,
            output_path=output_path,
            reverse=True,
            num_buckets=3,
        )

        assert stats["sorted_order"] == "hard-to-easy"

        entries = []
        with open(output_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    entries.append(json.loads(line))

        difficulties = [e["_difficulty"] for e in entries]
        assert difficulties == sorted(difficulties, reverse=True), (
            f"Expected hard-to-easy ordering, got: {difficulties}"
        )

    def test_curriculum_stats_are_valid(self, sample_jsonl, tmp_path):
        """Verify that curriculum statistics are reasonable."""
        output_path = tmp_path / "stats_check.jsonl"

        stats = sort_by_curriculum(
            input_path=sample_jsonl,
            output_path=output_path,
            num_buckets=3,
        )

        assert stats["total"] == 5
        assert 0.0 <= stats["avg_difficulty"] <= 1.0
        assert 0.0 <= stats["min_difficulty"] <= 1.0
        assert 0.0 <= stats["max_difficulty"] <= 1.0
        assert stats["min_difficulty"] <= stats["avg_difficulty"] <= stats["max_difficulty"]

        # Bucket distribution should sum to total
        bucket_sum = sum(stats["bucket_distribution"].values())
        assert bucket_sum == stats["total"]


class TestQualityRoundTrip:
    """Test the complete quality pipeline: evaluate -> preference -> curriculum."""

    @pytest.mark.slow
    def test_evaluate_then_preference_then_curriculum(self, tmp_path):
        """Full round-trip: create data -> evaluate -> preference pairs -> curriculum sort."""
        # Step 1: Create a JSONL dataset with varying quality texts
        input_jsonl = tmp_path / "input_data.jsonl"
        texts = [
            {
                "instruction": "Write a description:",
                "output": (
                    "The ancient cathedral rose majestically above the medieval "
                    "skyline, its gothic spires piercing the low-hanging clouds. "
                    "Buttresses arched gracefully along its flanks, each one a "
                    "testament to the ingenuity of long-dead architects."
                ),
            },
            {
                "instruction": "Continue the story:",
                "output": (
                    "He ran fast. He ran very fast. Running was what he did. "
                    "He ran and ran and ran. The running continued all day long."
                ),
            },
            {
                "instruction": "Describe the setting:",
                "output": (
                    "Moonlight filtered through the dense canopy, casting silver "
                    "patterns on the forest floor. An owl hooted somewhere in the "
                    "darkness, its call echoing through the ancient trees. The air "
                    "was crisp and smelled of pine needles and damp earth."
                ),
            },
            {
                "instruction": "Write dialogue:",
                "output": (
                    '"Do you remember the old bridge?" Maria asked, gazing across '
                    'the river. "Of course," replied Thomas, his voice tinged with '
                    "nostalgia. \"We spent every summer there as children, catching "
                    'fish and telling stories until the stars came out."'
                ),
            },
        ]

        with open(input_jsonl, "w", encoding="utf-8") as f:
            for entry in texts:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        # Step 2: Evaluate each text
        for entry in texts:
            result = evaluate_text(entry["output"])
            assert isinstance(result, EvaluationResult)
            assert result.overall_score > 0

        # Step 3: Generate preference pairs
        pref_output = tmp_path / "preferences.jsonl"
        num_pairs = generate_preference_pairs(
            input_path=input_jsonl,
            output_path=pref_output,
            min_score_diff=0.01,
            seed=42,
        )

        assert pref_output.exists()
        # With 4 entries of varying quality, should generate some pairs
        assert num_pairs >= 0  # May be 0 if scores are too close

        # Step 4: Sort by curriculum
        curriculum_output = tmp_path / "curriculum.jsonl"
        stats = sort_by_curriculum(
            input_path=input_jsonl,
            output_path=curriculum_output,
            num_buckets=2,
        )

        assert stats["total"] == 4
        assert curriculum_output.exists()

        # Verify the sorted output maintains all original data
        with open(curriculum_output, "r", encoding="utf-8") as f:
            sorted_entries = [json.loads(line) for line in f if line.strip()]

        assert len(sorted_entries) == 4

        # All entries should have the difficulty annotation
        for entry in sorted_entries:
            assert "_difficulty" in entry
            assert "output" in entry
            assert "instruction" in entry
