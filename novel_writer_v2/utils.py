"""Utility functions for the multi-agent story generator."""

import json
import re


def detect_language(text: str) -> str:
    """Detect if text is primarily Chinese or English.

    Returns "zh" or "en".
    """
    sample = text[:500]
    cjk_count = sum(1 for c in sample if '\u4e00' <= c <= '\u9fff')
    total_alpha = max(
        1, sum(1 for c in sample if c.isalpha() or '\u4e00' <= c <= '\u9fff')
    )
    return "zh" if (cjk_count / total_alpha) > 0.3 else "en"


def parse_json_response(text: str) -> dict:
    """Extract JSON from an LLM response that may contain markdown fences."""
    # Try to find JSON block in markdown code fence
    m = re.search(r'```(?:json)?\s*\n(.*?)\n```', text, re.DOTALL)
    if m:
        return json.loads(m.group(1))
    # Try parsing the whole text as JSON
    cleaned = text.strip()
    if cleaned.startswith('{') or cleaned.startswith('['):
        return json.loads(cleaned)
    # Last resort: find first { to last }
    start = cleaned.find('{')
    end = cleaned.rfind('}')
    if start != -1 and end != -1:
        return json.loads(cleaned[start:end + 1])
    raise ValueError(f"Could not parse JSON from response: {text[:200]}...")


def truncate_text(text: str, max_chars: int, from_end: bool = False) -> str:
    """Truncate text at sentence boundaries.

    Args:
        text: The text to truncate.
        max_chars: Maximum number of characters.
        from_end: If True, keep the end of the text instead of the beginning.
    """
    if len(text) <= max_chars:
        return text

    if from_end:
        # Keep the last max_chars, break at sentence boundary
        chunk = text[-(max_chars + 200):]
        # Find first sentence boundary in the chunk
        for sep in ['. ', '。', '！', '？', '! ', '? ', '\n']:
            idx = chunk.find(sep)
            if idx != -1 and idx < 200:
                return chunk[idx + len(sep):]
        return "..." + text[-max_chars:]
    else:
        # Keep the first max_chars, break at sentence boundary
        chunk = text[:max_chars + 200]
        # Find last sentence boundary in the chunk
        best = max_chars
        for sep in ['. ', '。', '！', '？', '! ', '? ', '\n']:
            idx = chunk.rfind(sep, 0, max_chars + 200)
            if idx != -1 and idx >= max_chars - 200:
                best = min(best, idx + len(sep))
        if best > max_chars:
            best = max_chars
        return text[:best] + "..."
