import re
import json
from pathlib import Path
from typing import List, Tuple, Dict
import math

from loguru import logger

class QualityFilter:
    """Score and filter text chunks by quality metrics."""

    def __init__(
        self,
        min_length: int = 500,
        max_length: int = 10000,
        min_dialogue_ratio: float = 0.0,
        min_variety: float = 0.3
    ):
        """
        Args:
            min_length: Minimum character count
            max_length: Maximum character count
            min_dialogue_ratio: Min ratio of dialogue (quotes/chars)
            min_variety: Min word variety (unique words / total words)
        """
        self.min_length = min_length
        self.max_length = max_length
        self.min_dialogue_ratio = min_dialogue_ratio
        self.min_variety = min_variety

    def score_length(self, text: str) -> float:
        """Score based on length (0-1)."""
        if len(text) < self.min_length:
            return 0.0
        if len(text) > self.max_length:
            return 0.5  # Penalize too long

        # Ideal length is in middle of range
        ideal = (self.min_length + self.max_length) / 2
        diff = abs(len(text) - ideal)
        return max(0, 1 - diff / ideal)

    def score_dialogue(self, text: str) -> float:
        """Score based on dialogue density (0-1)."""
        if len(text) == 0:
            return 0.0

        quote_count = text.count('"')
        ratio = quote_count / len(text)

        # Novel should have some dialogue
        if ratio < self.min_dialogue_ratio:
            return 0.0

        # Cap at reasonable maximum
        return min(1.0, ratio * 10)

    def score_variety(self, text: str) -> float:
        """Score based on vocabulary variety (0-1)."""
        words = text.lower().split()
        if len(words) == 0:
            return 0.0

        unique_words = set(words)
        variety = len(unique_words) / len(words)

        return max(0, variety - self.min_variety) / (1 - self.min_variety)

    def score_structure(self, text: str) -> float:
        """Score based on text structure (0-1)."""
        score = 1.0

        # Penalize all uppercase
        if text.upper() == text:
            score *= 0.3

        # Penalize too much whitespace
        whitespace_ratio = text.count('\n') / max(1, len(text))
        if whitespace_ratio > 0.3:
            score *= 0.5

        # Check for sentence structure (periods)
        sentences = text.count('.')
        if sentences < 2 and len(text) > 500:
            score *= 0.4

        return score

    def score(self, text: str) -> Tuple[float, Dict[str, float]]:
        """
        Calculate overall quality score.

        Returns:
            (overall_score, component_scores)
        """
        component_scores = {
            'length': self.score_length(text),
            'dialogue': self.score_dialogue(text),
            'variety': self.score_variety(text),
            'structure': self.score_structure(text),
        }

        # Weighted average
        weights = {
            'length': 0.2,
            'dialogue': 0.3,
            'variety': 0.3,
            'structure': 0.2,
        }

        overall = sum(
            component_scores[k] * weights[k]
            for k in component_scores
        )

        return overall, component_scores

    def filter_entries(
        self,
        entries: List[Dict],
        keep_ratio: float = 0.8
    ) -> List[Dict]:
        """
        Filter entries keeping top quality.

        Args:
            entries: List of JSON entries with 'output' field
            keep_ratio: Fraction of entries to keep

        Returns:
            Filtered list of entries
        """
        scored_entries = []

        for entry in entries:
            text = entry.get('output', '')
            score, components = self.score(text)
            scored_entries.append({
                'entry': entry,
                'score': score,
                'components': components
            })

        # Sort by score (descending)
        scored_entries.sort(key=lambda x: x['score'], reverse=True)

        # Keep top percentage
        keep_count = int(len(scored_entries) * keep_ratio)
        filtered = scored_entries[:keep_count]

        logger.info(
            f"Filtered {len(scored_entries)} → {len(filtered)} entries "
            f"(kept top {keep_ratio*100:.0f}%)"
        )

        # Log score distribution
        scores = [e['score'] for e in scored_entries]
        logger.info(
            f"Score range: {min(scores):.3f} - {max(scores):.3f}, "
            f"mean: {sum(scores)/len(scores):.3f}"
        )

        return [e['entry'] for e in filtered]

    def filter_jsonl(
        self,
        input_file: Path,
        output_file: Path = None,
        keep_ratio: float = 0.8
    ) -> int:
        """
        Filter JSONL dataset by quality.

        Returns:
            Number of filtered entries
        """
        entries = []

        # Read entries
        logger.info(f"Reading: {input_file}")
        with open(input_file, 'r', encoding='utf-8') as f:
            for line in f:
                entry = json.loads(line)
                entries.append(entry)

        # Filter
        logger.info(f"Filtering {len(entries)} entries...")
        filtered = self.filter_entries(entries, keep_ratio)

        # Write
        output_file = output_file or input_file.parent / f"{input_file.stem}_filtered.jsonl"
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w', encoding='utf-8') as f:
            for entry in filtered:
                json.dump(entry, f)
                f.write('\n')

        logger.success(f"Wrote {len(filtered)} filtered entries")

        return len(filtered)

def filter_dataset(
    input_file: Path,
    output_file: Path = None,
    keep_ratio: float = 0.8
) -> int:
    """
    Convenience function to filter dataset by quality.

    Returns:
        Number of filtered entries
    """
    filter = QualityFilter()
    return filter.filter_jsonl(input_file, output_file, keep_ratio)
