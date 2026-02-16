"""Sort training data by difficulty for curriculum learning."""

import json
from pathlib import Path
from typing import Optional

from loguru import logger
from pydantic import BaseModel


class DifficultyScore(BaseModel):
    """Difficulty assessment for a text sample."""
    vocabulary_complexity: float   # higher = more complex vocabulary
    sentence_complexity: float     # higher = longer sentences
    overall_difficulty: float      # weighted combination


def compute_difficulty(text: str) -> DifficultyScore:
    """
    Compute difficulty score for a text sample.

    Uses vocabulary diversity and sentence length as complexity proxies.
    Higher scores = more difficult text.
    """
    from ..evaluation.metrics import vocabulary_diversity, avg_sentence_length

    vocab_div = vocabulary_diversity(text)
    avg_sent_len = avg_sentence_length(text)

    # Normalize sentence length to 0-1 range (cap at 50 words per sentence)
    sent_complexity = min(avg_sent_len / 50.0, 1.0)

    # Overall difficulty: weighted combination
    overall = vocab_div * 0.6 + sent_complexity * 0.4

    return DifficultyScore(
        vocabulary_complexity=round(vocab_div, 4),
        sentence_complexity=round(sent_complexity, 4),
        overall_difficulty=round(overall, 4),
    )


def sort_by_curriculum(
    input_path: Path,
    output_path: Path,
    reverse: bool = False,
    num_buckets: int = 3,
) -> dict:
    """
    Sort training data by difficulty for curriculum learning.

    Reads JSONL, computes difficulty for each entry, sorts from easy to hard
    (or hard to easy if reverse=True), and writes sorted output.

    Also returns statistics about the difficulty distribution.

    Args:
        input_path: Input JSONL file.
        output_path: Output sorted JSONL file.
        reverse: If True, sort hard-to-easy (anti-curriculum).
        num_buckets: Number of difficulty buckets for statistics.

    Returns:
        Dict with statistics: total, bucket_counts, avg_difficulty, min/max difficulty.
    """
    entries = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            text = data.get("output", data.get("text", ""))
            if not text:
                continue

            difficulty = compute_difficulty(text)
            entries.append({
                "data": data,
                "difficulty": difficulty.overall_difficulty,
            })

    logger.info(f"Computed difficulty for {len(entries)} entries")

    # Sort by difficulty
    entries.sort(key=lambda x: x["difficulty"], reverse=reverse)

    # Write sorted output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for entry in entries:
            row = entry["data"].copy()
            row["_difficulty"] = entry["difficulty"]
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # Compute statistics
    difficulties = [e["difficulty"] for e in entries]
    stats = {
        "total": len(entries),
        "avg_difficulty": round(sum(difficulties) / len(difficulties), 4) if difficulties else 0.0,
        "min_difficulty": round(min(difficulties), 4) if difficulties else 0.0,
        "max_difficulty": round(max(difficulties), 4) if difficulties else 0.0,
        "sorted_order": "hard-to-easy" if reverse else "easy-to-hard",
    }

    # Bucket distribution
    if difficulties and num_buckets > 0:
        bucket_size = 1.0 / num_buckets
        buckets = {f"bucket_{i+1}": 0 for i in range(num_buckets)}
        for d in difficulties:
            idx = min(int(d / bucket_size), num_buckets - 1)
            buckets[f"bucket_{idx+1}"] += 1
        stats["bucket_distribution"] = buckets

    logger.success(f"Sorted {len(entries)} entries by difficulty -> {output_path}")
    logger.info(f"  Avg difficulty: {stats['avg_difficulty']}, Range: [{stats['min_difficulty']}, {stats['max_difficulty']}]")

    return stats
