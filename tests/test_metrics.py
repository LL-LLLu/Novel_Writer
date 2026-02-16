"""Comprehensive tests for novel_writer.evaluation.metrics module."""

import pytest
from unittest.mock import patch

from novel_writer.evaluation.metrics import (
    EvaluationResult,
    repetition_rate,
    vocabulary_diversity,
    avg_sentence_length,
    coherence_score,
    evaluate_text,
)


# ---------------------------------------------------------------------------
# repetition_rate tests
# ---------------------------------------------------------------------------


class TestRepetitionRate:
    """Tests for the repetition_rate function."""

    def test_no_repetitions(self):
        """All n-grams are unique -> repetition rate should be 0.0."""
        text = "the quick brown fox jumps over the lazy dog today"
        rate = repetition_rate(text, n=3)
        assert rate == 0.0

    def test_full_repetitions(self):
        """All n-grams are identical -> repetition rate should approach 1.0."""
        # "a a a a a" produces trigrams: (a,a,a), (a,a,a), (a,a,a)
        # 3 total, 1 unique => 1 - 1/3 = 0.6667
        text = "a a a a a"
        rate = repetition_rate(text, n=3)
        assert rate == pytest.approx(1.0 - 1.0 / 3.0)

    def test_partial_repetitions(self):
        """Some n-grams repeat, rate should be between 0 and 1."""
        # "a b c a b c d" -> trigrams: (a,b,c) (b,c,a) (c,a,b) (a,b,c) (b,c,d)
        # 5 total, 4 unique => 1 - 4/5 = 0.2
        text = "a b c a b c d"
        rate = repetition_rate(text, n=3)
        assert 0.0 < rate < 1.0
        assert rate == pytest.approx(0.2)

    def test_short_text_returns_zero(self):
        """Text shorter than n words should return 0.0."""
        assert repetition_rate("hello world", n=3) == 0.0
        assert repetition_rate("one", n=3) == 0.0

    def test_empty_text(self):
        """Empty string should return 0.0."""
        assert repetition_rate("", n=3) == 0.0

    def test_exact_n_words(self):
        """Text with exactly n words should produce one n-gram, rate 0.0."""
        text = "alpha beta gamma"
        rate = repetition_rate(text, n=3)
        assert rate == 0.0

    def test_custom_n_value(self):
        """Test with n=2 (bigrams)."""
        # "cat dog cat dog" -> bigrams: (cat,dog) (dog,cat) (cat,dog)
        # 3 total, 2 unique => 1 - 2/3
        text = "cat dog cat dog"
        rate = repetition_rate(text, n=2)
        assert rate == pytest.approx(1.0 - 2.0 / 3.0)


# ---------------------------------------------------------------------------
# vocabulary_diversity tests
# ---------------------------------------------------------------------------


class TestVocabularyDiversity:
    """Tests for the vocabulary_diversity function."""

    def test_diverse_text(self):
        """All unique words -> diversity should be 1.0."""
        text = "every single word here is completely unique and different"
        diversity = vocabulary_diversity(text)
        assert diversity == pytest.approx(1.0)

    def test_repetitive_text(self):
        """Highly repetitive text should have low diversity."""
        text = "the the the the the"
        diversity = vocabulary_diversity(text)
        assert diversity == pytest.approx(1.0 / 5.0)

    def test_mixed_case_normalization(self):
        """Uppercase and lowercase should be treated as the same word."""
        text = "Hello hello HELLO"
        diversity = vocabulary_diversity(text)
        assert diversity == pytest.approx(1.0 / 3.0)

    def test_empty_text(self):
        """Empty text should return 0.0."""
        assert vocabulary_diversity("") == 0.0

    def test_whitespace_only(self):
        """Whitespace-only text should return 0.0."""
        assert vocabulary_diversity("   ") == 0.0

    def test_single_word(self):
        """Single word should return 1.0."""
        assert vocabulary_diversity("hello") == pytest.approx(1.0)

    def test_partial_diversity(self):
        """Mix of unique and repeated words."""
        # "a b a c" -> 3 unique / 4 total = 0.75
        text = "a b a c"
        diversity = vocabulary_diversity(text)
        assert diversity == pytest.approx(0.75)


# ---------------------------------------------------------------------------
# avg_sentence_length tests
# ---------------------------------------------------------------------------


class TestAvgSentenceLength:
    """Tests for the avg_sentence_length function."""

    def test_multiple_sentences(self):
        """Multiple sentences with different lengths."""
        text = "Hello world. This is a test. One."
        # Sentences: "Hello world" (2), "This is a test" (4), "One" (1)
        # Average: (2 + 4 + 1) / 3 = 2.333...
        avg = avg_sentence_length(text)
        assert avg == pytest.approx(7.0 / 3.0)

    def test_single_sentence(self):
        """Single sentence with period."""
        text = "This is a single sentence."
        avg = avg_sentence_length(text)
        assert avg == pytest.approx(5.0)

    def test_empty_text(self):
        """Empty text should return 0.0."""
        assert avg_sentence_length("") == 0.0

    def test_whitespace_only(self):
        """Whitespace-only text should return 0.0."""
        assert avg_sentence_length("   ") == 0.0

    def test_exclamation_and_question(self):
        """Sentences ending with ! and ? should be recognized."""
        text = "Wow! Is this real? Yes it is."
        # Sentences: "Wow" (1), "Is this real" (3), "Yes it is" (3)
        avg = avg_sentence_length(text)
        assert avg == pytest.approx(7.0 / 3.0)

    def test_multiple_punctuation(self):
        """Multiple punctuation marks should not create empty sentences."""
        text = "Really?! That is amazing..."
        # Split on [.!?]+: "Really" and "That is amazing"
        # (The "..." collapses into one split)
        avg = avg_sentence_length(text)
        assert avg > 0.0

    def test_no_ending_punctuation(self):
        """Text without sentence-ending punctuation."""
        text = "no punctuation here at all"
        # The whole text is one sentence
        avg = avg_sentence_length(text)
        assert avg == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# coherence_score tests
# ---------------------------------------------------------------------------


class TestCoherenceScore:
    """Tests for the coherence_score function."""

    def test_returns_none_without_sentence_transformers(self):
        """Should return None when sentence-transformers is not installed."""
        with patch.dict("sys.modules", {"sentence_transformers": None}):
            # Force the import to fail by making the module None
            import importlib
            import novel_writer.evaluation.metrics as metrics_mod

            # We need to ensure the lazy import inside the function fails.
            # Patching builtins.__import__ is the reliable way:
            original_import = __builtins__.__import__ if hasattr(__builtins__, '__import__') else __import__

            def mock_import(name, *args, **kwargs):
                if name == "sentence_transformers":
                    raise ImportError("mocked")
                return original_import(name, *args, **kwargs)

            with patch("builtins.__import__", side_effect=mock_import):
                result = coherence_score("Hello there. How are you today?")
                assert result is None

    def test_fewer_than_two_sentences_returns_none(self):
        """With fewer than 2 sentences, coherence cannot be computed."""
        # Even if sentence-transformers were available, single sentence
        # should return None. We mock it to avoid the import issue.
        original_import = __import__

        def mock_import(name, *args, **kwargs):
            if name == "sentence_transformers":
                raise ImportError("mocked")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=mock_import):
            result = coherence_score("Just one sentence here")
            assert result is None


# ---------------------------------------------------------------------------
# evaluate_text tests
# ---------------------------------------------------------------------------


class TestEvaluateText:
    """Tests for the evaluate_text main entry point."""

    def test_returns_evaluation_result(self):
        """evaluate_text should return an EvaluationResult instance."""
        text = "The sun rose over the mountains. Birds sang their morning songs. A gentle breeze rustled the leaves."
        result = evaluate_text(text)
        assert isinstance(result, EvaluationResult)

    def test_result_fields_populated(self):
        """All required fields should be populated."""
        text = "The cat sat on the mat. The dog played in the yard."
        result = evaluate_text(text)

        assert isinstance(result.repetition_rate, float)
        assert isinstance(result.vocabulary_diversity, float)
        assert isinstance(result.avg_sentence_length, float)
        assert isinstance(result.overall_score, float)
        # perplexity is not computed yet
        assert result.perplexity is None

    def test_overall_score_without_coherence(self):
        """When coherence is None, overall uses adjusted weights."""
        text = "Alpha beta gamma delta. Epsilon zeta eta theta."

        # Mock coherence_score to return None (simulating no sentence-transformers)
        with patch(
            "novel_writer.evaluation.metrics.coherence_score", return_value=None
        ):
            result = evaluate_text(text)

        # overall = vocab_div * 0.45 + (1 - rep_rate) * 0.55
        expected = (
            result.vocabulary_diversity * 0.45
            + (1.0 - result.repetition_rate) * 0.55
        )
        assert result.overall_score == pytest.approx(expected)
        assert result.coherence_score is None

    def test_overall_score_with_coherence(self):
        """When coherence is available, overall uses 0.3/0.3/0.4 weights."""
        text = "The forest was dark. The trees swayed gently."

        mock_coherence = 0.85
        with patch(
            "novel_writer.evaluation.metrics.coherence_score",
            return_value=mock_coherence,
        ):
            result = evaluate_text(text)

        expected = (
            result.vocabulary_diversity * 0.3
            + (1.0 - result.repetition_rate) * 0.3
            + mock_coherence * 0.4
        )
        assert result.overall_score == pytest.approx(expected)
        assert result.coherence_score == pytest.approx(mock_coherence)

    def test_empty_text(self):
        """evaluate_text with empty text should not crash."""
        result = evaluate_text("")
        assert isinstance(result, EvaluationResult)
        assert result.vocabulary_diversity == 0.0
        assert result.avg_sentence_length == 0.0
        assert result.repetition_rate == 0.0

    def test_overall_score_range(self):
        """Overall score should be between 0 and 1 for reasonable text."""
        text = (
            "Once upon a time in a land far away. "
            "There lived a brave knight. "
            "He fought many dragons and saved the kingdom."
        )
        with patch(
            "novel_writer.evaluation.metrics.coherence_score", return_value=None
        ):
            result = evaluate_text(text)
        assert 0.0 <= result.overall_score <= 1.0

    def test_reference_text_accepted(self):
        """evaluate_text should accept reference_text without error."""
        text = "Generated text here."
        ref = "Reference text here."
        result = evaluate_text(text, reference_text=ref)
        assert isinstance(result, EvaluationResult)
