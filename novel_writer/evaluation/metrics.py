"""Automated evaluation metrics for generated novel text."""

import re
from typing import Optional

from loguru import logger
from pydantic import BaseModel


class EvaluationResult(BaseModel):
    """Container for all evaluation metric results."""

    perplexity: Optional[float] = None
    repetition_rate: float
    vocabulary_diversity: float  # type-token ratio
    avg_sentence_length: float
    coherence_score: Optional[float] = None
    overall_score: float


def repetition_rate(text: str, n: int = 3) -> float:
    """Calculate the ratio of repeated n-grams to total n-grams.

    Args:
        text: Input text to analyze.
        n: Size of n-grams to consider (default 3).

    Returns:
        Float between 0.0 (no repetition) and 1.0 (all repeated).
        Returns 0.0 if text is too short to form any n-grams.
    """
    words = text.split()
    if len(words) < n:
        return 0.0

    ngrams = [tuple(words[i : i + n]) for i in range(len(words) - n + 1)]
    total = len(ngrams)
    unique = len(set(ngrams))

    if total == 0:
        return 0.0

    # Repeated ratio: 1 - (unique / total)
    # 0 unique out of N total -> 1.0 (all repeated)
    # N unique out of N total -> 0.0 (no repetition)
    return 1.0 - (unique / total)


def vocabulary_diversity(text: str) -> float:
    """Calculate type-token ratio (unique words / total words).

    Args:
        text: Input text to analyze.

    Returns:
        Float between 0.0 and 1.0. Returns 0.0 for empty text.
    """
    words = text.lower().split()
    if not words:
        return 0.0

    unique_words = set(words)
    return len(unique_words) / len(words)


def avg_sentence_length(text: str) -> float:
    """Calculate the average number of words per sentence.

    Sentences are split on `.`, `!`, and `?` characters.

    Args:
        text: Input text to analyze.

    Returns:
        Average word count per sentence. Returns 0.0 for empty text.
    """
    if not text.strip():
        return 0.0

    sentences = re.split(r"[.!?]+", text)
    sentences = [s.strip() for s in sentences if s.strip()]

    if not sentences:
        return 0.0

    word_counts = [len(s.split()) for s in sentences]
    return sum(word_counts) / len(word_counts)


def coherence_score(text: str) -> Optional[float]:
    """Compute average cosine similarity between consecutive sentences.

    Uses sentence-transformers for embedding computation. If
    sentence-transformers is not installed, returns None.

    Args:
        text: Input text to analyze.

    Returns:
        Float between 0 and 1 representing average coherence,
        or None if sentence-transformers is not available.
    """
    try:
        from sentence_transformers import SentenceTransformer
        from sentence_transformers.util import cos_sim
    except ImportError:
        logger.debug(
            "sentence-transformers not installed; coherence_score unavailable"
        )
        return None

    sentences = re.split(r"[.!?]+", text)
    sentences = [s.strip() for s in sentences if s.strip()]

    if len(sentences) < 2:
        logger.debug("Fewer than 2 sentences; cannot compute coherence")
        return None

    logger.info("Computing coherence score with sentence-transformers")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(sentences)

    similarities = []
    for i in range(len(embeddings) - 1):
        sim = cos_sim(embeddings[i], embeddings[i + 1]).item()
        similarities.append(sim)

    avg = sum(similarities) / len(similarities)
    # Clamp to [0, 1]
    return max(0.0, min(1.0, avg))


def evaluate_text(
    text: str, reference_text: Optional[str] = None
) -> EvaluationResult:
    """Run all evaluation metrics on the given text.

    Args:
        text: The generated text to evaluate.
        reference_text: Optional reference text (reserved for future use).

    Returns:
        An EvaluationResult containing all computed metrics.
    """
    logger.info("Evaluating text ({} characters)", len(text))

    rep_rate = repetition_rate(text)
    vocab_div = vocabulary_diversity(text)
    avg_sent_len = avg_sentence_length(text)
    coh_score = coherence_score(text)

    # Compute overall score as a weighted average
    if coh_score is not None:
        overall = (
            vocab_div * 0.3 + (1.0 - rep_rate) * 0.3 + coh_score * 0.4
        )
    else:
        overall = vocab_div * 0.45 + (1.0 - rep_rate) * 0.55

    logger.info("Evaluation complete: overall_score={:.4f}", overall)

    return EvaluationResult(
        perplexity=None,
        repetition_rate=rep_rate,
        vocabulary_diversity=vocab_div,
        avg_sentence_length=avg_sent_len,
        coherence_score=coh_score,
        overall_score=overall,
    )
