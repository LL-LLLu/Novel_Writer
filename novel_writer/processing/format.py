import json
import random
from pathlib import Path
from typing import List

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


def _pick_instruction(text: str) -> str:
    """Pick a contextually appropriate instruction based on text content."""
    # Detect language: if >30% CJK characters, use Chinese instructions
    cjk_count = sum(1 for c in text[:200] if '\u4e00' <= c <= '\u9fff')
    total_alpha = max(1, sum(1 for c in text[:200] if c.isalpha() or '\u4e00' <= c <= '\u9fff'))
    is_chinese = (cjk_count / total_alpha) > 0.3

    pool = _ZH_INSTRUCTIONS if is_chinese else _EN_INSTRUCTIONS
    return random.choice(pool)


def create_chunks(text: str, chunk_size: int, overlap: int) -> List[str]:
    """Create overlapping chunks from text."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += (chunk_size - overlap)
    return chunks

def format_dataset(input_dir: Path, output_file: Path, config: Config) -> int:
    """Format cleaned data into JSONL."""
    input_path = input_dir
    data = []

    files = list(input_path.glob('*.txt'))
    logger.info(f"Found {len(files)} cleaned files to format")

    from ..utils.progress import process_with_progress

    def process_func(file_path):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
                chunks = create_chunks(
                    text,
                    config.data.chunk_size,
                    config.data.overlap
                )

                for chunk in chunks:
                    if len(chunk) < 100:
                        continue

                    entry = {
                        "instruction": _pick_instruction(chunk),
                        "input": "",
                        "output": chunk
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

    # Write to JSONL
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        for entry in data:
            json.dump(entry, f)
            f.write('\n')

    logger.success(f"Saved {len(data)} entries to {output_file}")
    return len(data)

def format_data(config: Config) -> int:
    """Main format function using config."""
    output_file = config.data.output_dir / "train.jsonl"
    return format_dataset(config.data.temp_dir, output_file, config)
