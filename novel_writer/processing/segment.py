import re
from pathlib import Path
from typing import List, Tuple
import unicodedata

from loguru import logger

class ChapterSegmenter:
    """Intelligent chapter segmentation for novels."""

    # Patterns for chapter detection
    CHAPTER_PATTERNS = [
        # English patterns
        r"Chapter\s+\d+",
        r"Chapter\s+[IVXLCDM]+",
        r"Part\s+\d+",
        r"Book\s+\d+",
        # Chinese/Japanese patterns
        r"第\d+章",
        r"第\d+回",
        # Common novel headers
        r"Prologue\b",
        r"Epilogue\b",
        r"Volume\s+\d+",
    ]

    def __init__(self, min_chapter_length: int = 1000):
        """
        Args:
            min_chapter_length: Minimum character count for a valid chapter
        """
        self.min_chapter_length = min_chapter_length
        self.pattern = re.compile(
            "|".join(self.CHAPTER_PATTERNS),
            re.IGNORECASE | re.MULTILINE
        )

    def find_chapter_boundaries(self, text: str) -> List[Tuple[str, int]]:
        """
        Find all chapter boundaries in text.

        Returns:
            List of (chapter_title, position) tuples
        """
        matches = []
        for match in self.pattern.finditer(text):
            title = match.group().strip()
            position = match.start()
            matches.append((title, position))
        return matches

    def segment_text(self, text: str) -> List[Tuple[str, str]]:
        """
        Segment text into chapters.

        Returns:
            List of (chapter_title, chapter_content) tuples
        """
        boundaries = self.find_chapter_boundaries(text)

        if not boundaries:
            logger.warning("No chapters detected, treating as single chapter")
            return [("Full Text", text)]

        chapters = []

        # Add first chapter (before first boundary)
        if boundaries[0][1] > 0:
            title, pos = boundaries[0]
            preface = text[:pos].strip()
            if len(preface) > 200:  # Keep if substantial
                chapters.append(("Preface", preface))

        # Extract chapters
        for i, (title, start_pos) in enumerate(boundaries):
            # Find next boundary
            if i + 1 < len(boundaries):
                _, next_pos = boundaries[i + 1]
                content = text[start_pos:next_pos].strip()
            else:
                # Last chapter
                content = text[start_pos:].strip()

            # Filter short chapters (likely false positives)
            if len(content) >= self.min_chapter_length:
                # Clean title (remove extra whitespace)
                clean_title = re.sub(r'\s+', ' ', title).strip()
                chapters.append((clean_title, content))
            else:
                logger.debug(f"Skipping short chapter: {title} ({len(content)} chars)")

        logger.info(f"Segmented into {len(chapters)} chapters")
        return chapters

    def segment_file(self, file_path: Path) -> List[Path]:
        """
        Segment a file and save chapters to separate files.

        Returns:
            List of paths to chapter files
        """
        logger.info(f"Segmenting file: {file_path.name}")

        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()

        chapters = self.segment_text(text)
        output_dir = file_path.parent / f"{file_path.stem}_chapters"
        output_dir.mkdir(exist_ok=True)

        chapter_paths = []
        for i, (title, content) in enumerate(chapters, 1):
            # Create filename from title (safe for filesystem)
            safe_title = re.sub(r'[^\w\s-]', '', title)
            safe_title = safe_title.replace(' ', '_')[:50]
            chapter_file = output_dir / f"{i:03d}_{safe_title}.txt"

            with open(chapter_file, 'w', encoding='utf-8') as f:
                f.write(content)

            chapter_paths.append(chapter_file)
            logger.debug(f"Saved: {chapter_file.name} ({len(content)} chars)")

        return chapter_paths

def segment_directory(
    input_dir: Path,
    min_chapter_length: int = 1000
) -> List[Path]:
    """
    Segment all files in a directory.

    Returns:
        List of all chapter file paths
    """
    segmenter = ChapterSegmenter(min_chapter_length=min_chapter_length)
    all_chapters = []

    files = list(input_dir.glob('*.txt'))
    logger.info(f"Segmenting {len(files)} files...")

    for file_path in files:
        try:
            chapters = segmenter.segment_file(file_path)
            all_chapters.extend(chapters)
        except Exception as e:
            logger.error(f"Failed to segment {file_path}: {e}")
            continue

    logger.success(f"Created {len(all_chapters)} chapter files")
    return all_chapters
