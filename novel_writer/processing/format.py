import json
import random
import re
from pathlib import Path
from typing import List, Tuple

from loguru import logger
from ..config import Config

# Diverse instruction pool for training data variety
_ZH_INSTRUCTIONS = [
    "续写这段叙事，保持原文的风格和节奏。",
    "以相同的文风继续这个故事。",
    "根据已有的情节和人物设定，续写下一段。",
    "保持叙事视角不变，继续推进故事发展。",
    "用生动的细节描写续写这个场景。",
    "通过对话和动作描写推进下面的情节。",
    "延续当前的叙事氛围，写出接下来发生的事。",
    "以细腻的笔触续写这段文字。",
    "按照原文的叙事节奏，写出故事的下一部分。",
    "继续描绘这个场景中的人物和事件。",
    "用符合原文风格的语言续写故事。",
    "展开叙述，让故事自然地向前发展。",
    "保持文风一致，续写接下来的情节。",
    "以沉浸式的叙事方式继续这段故事。",
    "描绘接下来的场景，注意环境和人物的刻画。",
    "用简洁有力的文字续写这段叙事。",
    "继续讲述这个故事，注意情感的表达。",
    "以自然流畅的文笔续写下一段。",
    "延续原文的基调，推进故事走向。",
    "用丰富的感官描写续写这个场景。",
]

_EN_INSTRUCTIONS = [
    "Continue the narrative in the established style.",
    "Write the next passage, maintaining the existing voice and tone.",
    "Advance the story using vivid sensory details.",
    "Continue this scene with natural dialogue and action.",
    "Extend the narrative, preserving the point of view and pacing.",
    "Write what happens next, staying true to the characters.",
    "Continue the story with concrete, immersive description.",
    "Carry the narrative forward in the same literary register.",
    "Write the next segment, matching the established rhythm.",
    "Develop this scene further with authentic detail.",
    "Push the story forward through action and dialogue.",
    "Continue in the same voice, advancing the plot naturally.",
    "Write the following passage in the style of the preceding text.",
    "Extend this scene with attention to atmosphere and character.",
    "Continue the narrative arc with engaging prose.",
    "Write what comes next, maintaining tension and pacing.",
    "Advance the story, weaving in environmental detail.",
    "Continue with prose that matches the tone and texture of the original.",
    "Develop the next beat of the story with precise language.",
    "Carry the scene forward, balancing action with description.",
]

# Sentence-ending patterns for boundary-aware splitting
# CJK: 。！？… don't require trailing whitespace (Chinese has no spaces between sentences)
# English: .!? require trailing whitespace to avoid matching abbreviations
_SENTENCE_END_CJK_RE = re.compile(
    r'[。！？…][""\'\u201d\u300d）\)]*'  # CJK sentence-enders + optional closing quotes
)
_SENTENCE_END_EN_RE = re.compile(
    r'[\.\!\?][""\'\u201d）\)]*'  # English sentence-enders + optional closing quotes
    r'(?:\s|\n)'                  # must be followed by whitespace/newline
)


def _is_chinese(text: str) -> bool:
    """Detect if text is primarily Chinese."""
    sample = text[:500]
    cjk_count = sum(1 for c in sample if '\u4e00' <= c <= '\u9fff')
    total_alpha = max(1, sum(1 for c in sample if c.isalpha() or '\u4e00' <= c <= '\u9fff'))
    return (cjk_count / total_alpha) > 0.3


def _pick_instruction(text: str) -> str:
    """Pick a contextually appropriate instruction based on text content."""
    pool = _ZH_INSTRUCTIONS if _is_chinese(text) else _EN_INSTRUCTIONS
    return random.choice(pool)


def _find_sentence_boundary(text: str, target_pos: int, search_range: int = 300) -> int:
    """Find the nearest sentence boundary near target_pos.

    Searches within search_range characters around target_pos for a sentence-ending
    punctuation mark. Returns the position right after the sentence end (start of
    next sentence). Falls back to paragraph boundary, then target_pos.
    """
    # Search window
    start = max(0, target_pos - search_range)
    end = min(len(text), target_pos + search_range)
    window = text[start:end]

    # Find all sentence boundaries in the window (both CJK and English)
    best_pos = None
    best_dist = search_range + 1

    for pattern in (_SENTENCE_END_CJK_RE, _SENTENCE_END_EN_RE):
        for m in pattern.finditer(window):
            boundary = start + m.end()
            dist = abs(boundary - target_pos)
            if dist < best_dist:
                best_dist = dist
                best_pos = boundary

    if best_pos is not None:
        return best_pos

    # Fallback: look for paragraph boundary (\n\n)
    para_start = max(0, target_pos - search_range)
    para_end = min(len(text), target_pos + search_range)
    for offset in range(0, search_range):
        for pos in [target_pos + offset, target_pos - offset]:
            if 0 <= pos < len(text) - 1 and text[pos:pos+2] == '\n\n':
                return pos + 2
    return target_pos


def create_chunks(text: str, chunk_size: int, overlap: int) -> List[str]:
    """Create sentence-boundary-aware chunks from text."""
    chunks = []
    text_len = len(text)
    if text_len == 0:
        return chunks

    start = 0
    while start < text_len:
        # Find end position, snapping to sentence boundary
        raw_end = start + chunk_size
        if raw_end >= text_len:
            # Last chunk — take everything remaining
            chunk = text[start:]
            if chunk.strip():
                chunks.append(chunk.strip())
            break

        end = _find_sentence_boundary(text, raw_end)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        # Advance, accounting for overlap
        next_start = end - overlap
        if next_start <= start:
            # Avoid infinite loop: force advance
            next_start = start + chunk_size - overlap
        start = next_start

    return chunks


def create_continuation_pairs(
    text: str,
    chunk_size: int,
    context_ratio: float = 0.4,
) -> List[Tuple[str, str]]:
    """Split text into (context, continuation) pairs for training.

    Each pair has the preceding text as context (input) and the following
    text as continuation (output). This teaches the model actual continuation.

    Args:
        text: Full text to split
        chunk_size: Total size of each context+continuation pair
        context_ratio: Fraction of chunk_size to use as context (input)

    Returns:
        List of (context, continuation) tuples
    """
    pairs = []
    text_len = len(text)
    if text_len < 200:
        return pairs

    context_size = int(chunk_size * context_ratio)
    continuation_size = chunk_size - context_size
    # Step forward by continuation_size so each continuation is unique
    step = continuation_size

    pos = 0
    while pos + context_size + 100 < text_len:
        # Snap the split point (between context and continuation) to a sentence boundary
        raw_split = pos + context_size
        split = _find_sentence_boundary(text, raw_split, search_range=200)

        # Snap the end to a sentence boundary too
        raw_end = split + continuation_size
        if raw_end >= text_len:
            end = text_len
        else:
            end = _find_sentence_boundary(text, raw_end, search_range=200)

        context = text[pos:split].strip()
        continuation = text[split:end].strip()

        if len(context) >= 100 and len(continuation) >= 100:
            pairs.append((context, continuation))

        # Advance
        next_pos = pos + step
        if next_pos <= pos:
            next_pos = pos + max(500, step)
        pos = next_pos

    return pairs


def format_dataset(input_dir: Path, output_file: Path, config: Config) -> int:
    """Format cleaned data into JSONL with context/continuation pairs."""
    input_path = input_dir
    data = []

    # Use rglob to also pick up files in chapter subdirectories
    files = list(input_path.rglob('*.txt'))
    logger.info(f"Found {len(files)} cleaned files to format")

    from ..utils.progress import process_with_progress

    def process_func(file_path):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()

            if len(text) < 200:
                return

            is_zh = _is_chinese(text)

            # Generate context/continuation pairs
            pairs = create_continuation_pairs(
                text,
                config.data.chunk_size,
                context_ratio=0.4,
            )

            for context, continuation in pairs:
                entry = {
                    "instruction": _pick_instruction(continuation),
                    "input": context,
                    "output": continuation,
                }
                data.append(entry)

            # Also add some standalone chunks for variety (pure completion training)
            # Use ~20% of the text for this
            if len(text) > config.data.chunk_size:
                standalone_chunks = create_chunks(
                    text,
                    config.data.chunk_size,
                    config.data.overlap,
                )
                # Sample a subset to avoid overwhelming the continuation pairs
                n_standalone = max(1, len(standalone_chunks) // 5)
                for chunk in random.sample(standalone_chunks, min(n_standalone, len(standalone_chunks))):
                    if len(chunk) < 100:
                        continue
                    entry = {
                        "instruction": _pick_instruction(chunk),
                        "input": "",
                        "output": chunk,
                    }
                    data.append(entry)

        except Exception as e:
            logger.error(f"Failed to format {file_path}: {e}")

    process_with_progress(
        files,
        process_func,
        description="Formatting dataset...",
        total=len(files)
    )

    # Shuffle to mix continuation pairs with standalone chunks
    random.shuffle(data)

    # Write to JSONL
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        for entry in data:
            json.dump(entry, f, ensure_ascii=False)
            f.write('\n')

    n_pairs = sum(1 for e in data if e['input'])
    n_standalone = len(data) - n_pairs
    logger.success(f"Saved {len(data)} entries to {output_file} "
                   f"({n_pairs} continuation pairs + {n_standalone} standalone)")
    return len(data)


def format_data(config: Config) -> int:
    """Main format function using config."""
    output_file = config.data.output_dir / "train.jsonl"
    return format_dataset(config.data.temp_dir, output_file, config)
