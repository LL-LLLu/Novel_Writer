import json
from pathlib import Path
from typing import List

from loguru import logger
from ..config import Config

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
                        "instruction": "Continue the narrative in the established style.",
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
