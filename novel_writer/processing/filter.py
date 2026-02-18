import re
import json
from pathlib import Path
from typing import List, Tuple, Dict
import math

from loguru import logger


def _is_chinese(text: str) -> bool:
    """Detect if text is primarily Chinese."""
    sample = text[:500]
    cjk_count = sum(1 for c in sample if '\u4e00' <= c <= '\u9fff')
    total_alpha = max(1, sum(1 for c in sample if c.isalpha() or '\u4e00' <= c <= '\u9fff'))
    return (cjk_count / total_alpha) > 0.3


class QualityFilter:
    """Score and filter text chunks by quality metrics."""

    def __init__(
        self,
        min_length: int = 500,
        max_length: int = 10000,
        min_dialogue_ratio: float = 0.0,
        min_variety: float = 0.3
    ):
        self.min_length = min_length
        self.max_length = max_length
        self.min_dialogue_ratio = min_dialogue_ratio
        self.min_variety = min_variety

    def score_length(self, text: str) -> float:
        """Score based on length (0-1)."""
        if len(text) < self.min_length:
            return 0.0
        if len(text) > self.max_length:
            return 0.5

        ideal = (self.min_length + self.max_length) / 2
        diff = abs(len(text) - ideal)
        return max(0, 1 - diff / ideal)

    def score_dialogue(self, text: str) -> float:
        """Score based on dialogue density (0-1). Supports CJK quotes."""
        if len(text) == 0:
            return 0.0

        # Count all dialogue quote styles
        quote_count = (
            text.count('"') +          # ASCII double quote
            text.count('\u201c') +     # left curly double quote "
            text.count('\u201d') +     # right curly double quote "
            text.count('\u300c') +     # CJK left corner bracket 「
            text.count('\u300d') +     # CJK right corner bracket 」
            text.count('\u300e') +     # CJK left double corner bracket 『
            text.count('\u300f')       # CJK right double corner bracket 』
        )
        ratio = quote_count / len(text)

        if ratio < self.min_dialogue_ratio:
            return 0.0

        return min(1.0, ratio * 10)

    def score_variety(self, text: str) -> float:
        """Score based on vocabulary variety (0-1). CJK-aware."""
        is_zh = _is_chinese(text)

        if is_zh:
            # For Chinese: use character bigrams instead of space-split words
            chars = [c for c in text if '\u4e00' <= c <= '\u9fff']
            if len(chars) < 2:
                return 0.0
            bigrams = [chars[i] + chars[i+1] for i in range(len(chars) - 1)]
            total = len(bigrams)
            unique = len(set(bigrams))
        else:
            words = text.lower().split()
            if len(words) == 0:
                return 0.0
            total = len(words)
            unique = len(set(words))

        variety = unique / total
        return max(0, variety - self.min_variety) / (1 - self.min_variety)

    def score_structure(self, text: str) -> float:
        """Score based on text structure (0-1). Supports CJK punctuation."""
        score = 1.0

        # Penalize all uppercase (only relevant for non-CJK)
        if text.upper() == text and not _is_chinese(text):
            score *= 0.3

        # Penalize too much whitespace
        whitespace_ratio = text.count('\n') / max(1, len(text))
        if whitespace_ratio > 0.3:
            score *= 0.5

        # Check for sentence structure: count both English and Chinese sentence endings
        sentences = (
            text.count('.') +
            text.count('\u3002') +   # Chinese period 。
            text.count('\uff01') +   # Chinese exclamation ！
            text.count('\uff1f') +   # Chinese question mark ？
            text.count('!') +
            text.count('?')
        )
        if sentences < 2 and len(text) > 500:
            score *= 0.4

        return score

    def score(self, text: str) -> Tuple[float, Dict[str, float]]:
        """Calculate overall quality score."""
        component_scores = {
            'length': self.score_length(text),
            'dialogue': self.score_dialogue(text),
            'variety': self.score_variety(text),
            'structure': self.score_structure(text),
        }

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
        """Filter entries keeping top quality."""
        scored_entries = []

        for entry in entries:
            # Score based on output (the main text), but also consider input if present
            text = entry.get('output', '')
            input_text = entry.get('input', '')
            if input_text:
                # For continuation pairs, score the combined text for better assessment
                text = input_text + '\n' + text

            score, components = self.score(text)
            scored_entries.append({
                'entry': entry,
                'score': score,
                'components': components
            })

        scored_entries.sort(key=lambda x: x['score'], reverse=True)

        keep_count = int(len(scored_entries) * keep_ratio)
        filtered = scored_entries[:keep_count]

        logger.info(
            f"Filtered {len(scored_entries)} → {len(filtered)} entries "
            f"(kept top {keep_ratio*100:.0f}%)"
        )

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
        """Filter JSONL dataset by quality."""
        entries = []

        logger.info(f"Reading: {input_file}")
        with open(input_file, 'r', encoding='utf-8') as f:
            for line in f:
                entry = json.loads(line)
                entries.append(entry)

        logger.info(f"Filtering {len(entries)} entries...")
        filtered = self.filter_entries(entries, keep_ratio)

        output_file = output_file or input_file.parent / f"{input_file.stem}_filtered.jsonl"
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w', encoding='utf-8') as f:
            for entry in filtered:
                json.dump(entry, f, ensure_ascii=False)
                f.write('\n')

        logger.success(f"Wrote {len(filtered)} filtered entries")

        return len(filtered)


def filter_dataset(
    input_file: Path,
    output_file: Path = None,
    keep_ratio: float = 0.8
) -> int:
    """Convenience function to filter dataset by quality."""
    filter = QualityFilter()
    return filter.filter_jsonl(input_file, output_file, keep_ratio)
