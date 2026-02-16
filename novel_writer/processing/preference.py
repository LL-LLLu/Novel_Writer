"""Generate DPO preference pairs for alignment training."""

import json
import random
from pathlib import Path
from typing import Optional

from loguru import logger
from pydantic import BaseModel


class PreferencePair(BaseModel):
    """A single DPO preference pair."""
    prompt: str
    chosen: str        # preferred response
    rejected: str      # dispreferred response
    chosen_score: float
    rejected_score: float


def score_text(text: str) -> float:
    """Score a text sample using evaluation metrics."""
    from ..evaluation.metrics import evaluate_text
    result = evaluate_text(text)
    return result.overall_score


def generate_preference_pairs(
    input_path: Path,
    output_path: Path,
    min_score_diff: float = 0.1,
    max_pairs: Optional[int] = None,
    seed: int = 42,
) -> int:
    """
    Generate DPO preference pairs from a JSONL dataset.

    Reads entries from input JSONL, scores them, and creates pairs where
    one response is preferred over another based on quality scores.

    Each input line should be JSON with at least an "output" field (and optionally "instruction").

    Args:
        input_path: Path to input JSONL file.
        output_path: Path to output JSONL file.
        min_score_diff: Minimum score difference between chosen and rejected.
        max_pairs: Maximum number of pairs to generate. None for unlimited.
        seed: Random seed for reproducibility.

    Returns:
        Number of preference pairs generated.
    """
    random.seed(seed)

    # Read and score all entries
    entries = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            text = data.get("output", "")
            if not text:
                continue
            score = score_text(text)
            entries.append({
                "instruction": data.get("instruction", ""),
                "output": text,
                "score": score,
            })

    logger.info(f"Scored {len(entries)} entries from {input_path}")

    # Sort by score
    entries.sort(key=lambda x: x["score"], reverse=True)

    # Generate pairs by pairing high-score with low-score entries
    pairs = []
    n = len(entries)

    for i in range(n):
        for j in range(i + 1, n):
            if entries[i]["score"] - entries[j]["score"] >= min_score_diff:
                pair = PreferencePair(
                    prompt=entries[i]["instruction"] or entries[j]["instruction"] or "Continue writing:",
                    chosen=entries[i]["output"],
                    rejected=entries[j]["output"],
                    chosen_score=entries[i]["score"],
                    rejected_score=entries[j]["score"],
                )
                pairs.append(pair)

                if max_pairs and len(pairs) >= max_pairs:
                    break
        if max_pairs and len(pairs) >= max_pairs:
            break

    # Shuffle pairs
    random.shuffle(pairs)

    # Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for pair in pairs:
            f.write(pair.model_dump_json() + "\n")

    logger.success(f"Generated {len(pairs)} preference pairs -> {output_path}")
    return len(pairs)
