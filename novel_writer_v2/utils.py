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


def _repair_truncated_json(text: str) -> str:
    """Attempt to repair truncated JSON by finding the last complete top-level element."""
    if not text or text[0] not in ('{', '['):
        return text

    # For arrays: find the last complete object `}` followed by `,` or at array level
    # Strategy: progressively try parsing longer prefixes ending at `}` or `]`
    # But that's slow. Instead: find the last `}` that makes valid JSON when we close.
    if text[0] == '[':
        # Find positions of all top-level-ish `},` or `}]` patterns
        # Walk backwards from the end looking for a `}` that completes an element
        last_brace = len(text)
        while True:
            last_brace = text.rfind('}', 0, last_brace)
            if last_brace <= 0:
                break
            # Try closing the array after this brace
            candidate = text[:last_brace + 1].rstrip().rstrip(',') + ']'
            try:
                return candidate
            except Exception:
                pass
            # Try parsing it
            try:
                json.loads(candidate)
                return candidate
            except json.JSONDecodeError:
                continue

    # For objects: find the last complete key-value pair
    if text[0] == '{':
        last_end = len(text)
        for ch in ('}', ']', '"'):
            while True:
                last_end_pos = text.rfind(ch, 0, last_end)
                if last_end_pos <= 0:
                    break
                candidate = text[:last_end_pos + 1]
                # Count open/close to figure out what to append
                stack = []
                in_str = False
                esc = False
                for c in candidate:
                    if esc:
                        esc = False
                        continue
                    if c == '\\' and in_str:
                        esc = True
                        continue
                    if c == '"':
                        in_str = not in_str
                        continue
                    if in_str:
                        continue
                    if c in ('{', '['):
                        stack.append('}' if c == '{' else ']')
                    elif c in ('}', ']') and stack:
                        stack.pop()
                closing = ''.join(reversed(stack))
                try:
                    json.loads(candidate + closing)
                    return candidate + closing
                except json.JSONDecodeError:
                    last_end = last_end_pos
                    continue
                break

    return text


def parse_json_response(text: str) -> dict | list:
    """Extract JSON from an LLM response that may contain markdown fences."""
    # Try to find JSON block in markdown code fence
    m = re.search(r'```(?:json)?\s*\n(.*?)\n```', text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    # Try parsing the whole text as JSON
    cleaned = text.strip()
    if cleaned.startswith('{') or cleaned.startswith('['):
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass
    # Find the outermost JSON structure
    for open_ch, close_ch in [('[', ']'), ('{', '}')]:
        start = cleaned.find(open_ch)
        if start == -1:
            continue
        end = cleaned.rfind(close_ch)
        if end > start:
            try:
                return json.loads(cleaned[start:end + 1])
            except json.JSONDecodeError:
                pass
    # Last resort: try to repair truncated JSON
    for open_ch in ('[', '{'):
        start = cleaned.find(open_ch)
        if start != -1:
            try:
                repaired = _repair_truncated_json(cleaned[start:])
                return json.loads(repaired)
            except (json.JSONDecodeError, ValueError):
                pass
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
