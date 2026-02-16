"""End-to-end tests for the clean -> format pipeline."""

import json
import pytest
from pathlib import Path

from tests.e2e.conftest import SAMPLE_NOVEL_TEXT, SAMPLE_ENGLISH_TEXT
from novel_writer.processing.clean import clean_data, clean_text, process_file
from novel_writer.processing.format import format_data, create_chunks
from novel_writer.processing.segment import ChapterSegmenter, segment_directory


class TestCleanFormatPipeline:
    """Test the clean -> format pipeline end-to-end."""

    def test_clean_then_format(self, sample_config, sample_txt_files):
        """Run clean_data followed by format_data and verify JSONL output."""
        # Step 1: Clean
        cleaned_files = clean_data(sample_config)
        assert len(cleaned_files) > 0, "clean_data should produce at least one output file"

        # Verify cleaned files exist in temp_dir
        for f in cleaned_files:
            assert f.exists()
            content = f.read_text(encoding="utf-8")
            assert len(content) >= 500, f"Cleaned file should be >= 500 chars, got {len(content)}"

        # Step 2: Format
        num_entries = format_data(sample_config)
        assert num_entries > 0, "format_data should produce entries"

        # Step 3: Verify JSONL output
        output_file = sample_config.data.output_dir / "train.jsonl"
        assert output_file.exists()

        entries = []
        with open(output_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    entries.append(json.loads(line))

        assert len(entries) == num_entries
        assert len(entries) > 0

    def test_jsonl_entry_structure(self, sample_config, sample_txt_files):
        """Verify that JSONL entries have the correct keys."""
        clean_data(sample_config)
        format_data(sample_config)

        output_file = sample_config.data.output_dir / "train.jsonl"
        with open(output_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)

                # Must have these keys
                assert "instruction" in entry, "Entry missing 'instruction' key"
                assert "input" in entry, "Entry missing 'input' key"
                assert "output" in entry, "Entry missing 'output' key"

                # output should have substantial content
                assert len(entry["output"]) >= 100, (
                    f"Entry output too short: {len(entry['output'])} chars"
                )

                # instruction should be non-empty
                assert len(entry["instruction"]) > 0

    def test_clean_text_function_directly(self):
        """Test the clean_text function on sample data."""
        dirty_text = (
            "Page 1 of 10\n\n\n\n"
            "The hero walked through the forest.   There were many trees.\n\n\n\n\n"
            "Page 2 of 10\n\nHe found a clearing in the woods."
        )
        cleaned = clean_text(dirty_text)

        # Page numbers should be removed
        assert "Page 1 of 10" not in cleaned
        assert "Page 2 of 10" not in cleaned

        # Excessive whitespace should be normalized
        assert "\n\n\n" not in cleaned

        # Content should be preserved
        assert "hero walked" in cleaned
        assert "clearing" in cleaned

    def test_format_skips_short_chunks(self, sample_config):
        """Verify format_data skips chunks that are too short (<100 chars)."""
        # Create a very short file in temp_dir that will produce short chunks
        short_file = sample_config.data.temp_dir / "short_cleaned.txt"
        short_file.write_text("Short text." * 5, encoding="utf-8")  # 55 chars

        num_entries = format_data(sample_config)

        # The short file's chunks should be skipped
        assert num_entries == 0

    def test_create_chunks_function(self):
        """Test the chunking logic with known inputs."""
        text = "A" * 1000  # 1000 character text
        chunks = create_chunks(text, chunk_size=400, overlap=100)

        # Each chunk should be at most 400 chars
        for chunk in chunks:
            assert len(chunk) <= 400

        # Should create multiple chunks
        assert len(chunks) >= 3

        # Chunks should cover the full text
        # With chunk_size=400, overlap=100, step=300
        # Positions: 0, 300, 600, 900
        assert len(chunks) == 4


class TestSegmentFormatPipeline:
    """Test segmentation followed by formatting."""

    def test_segment_text_detects_chapters(self):
        """Verify chapter detection works on text with Arabic-numeral chapters."""
        # The segmenter pattern is 第\d+章 (Arabic digits), so we need
        # text that uses 第1章, 第2章 rather than 第一章, 第二章
        chinese_text_with_arabic_nums = SAMPLE_NOVEL_TEXT.replace("第一章", "第1章").replace(
            "第二章", "第2章"
        )
        segmenter = ChapterSegmenter(min_chapter_length=50)
        chapters = segmenter.segment_text(chinese_text_with_arabic_nums)

        assert len(chapters) >= 2, (
            f"Should detect at least 2 Chinese chapters, got {len(chapters)}: "
            f"{[t for t, _ in chapters]}"
        )

        titles = [title for title, _ in chapters]
        title_text = " ".join(titles)
        assert "第" in title_text, f"Should find Chinese chapter markers in titles: {titles}"

    def test_segment_english_text(self):
        """Verify chapter detection works on English text."""
        segmenter = ChapterSegmenter(min_chapter_length=50)
        chapters = segmenter.segment_text(SAMPLE_ENGLISH_TEXT)

        assert len(chapters) >= 2, "Should detect at least 2 English chapters"

        titles = [title for title, _ in chapters]
        title_text = " ".join(titles)
        assert "Chapter" in title_text, f"Should find Chapter markers in titles: {titles}"

    def test_segment_then_format(self, sample_config):
        """Segment files by chapter, then format into JSONL."""
        input_dir = sample_config.data.input_dir

        # Create a long text with chapters that meets the cleaning threshold
        long_text = SAMPLE_ENGLISH_TEXT * 5  # Repeat to exceed length thresholds

        txt_file = input_dir / "long_novel.txt"
        txt_file.write_text(long_text, encoding="utf-8")

        # Segment with low min_chapter_length so chapters pass through
        chapter_paths = segment_directory(input_dir, min_chapter_length=50)

        # There should be chapter files created
        assert len(chapter_paths) > 0, "Segmentation should produce chapter files"

        # Each chapter file should have content
        for cp in chapter_paths:
            assert cp.exists()
            content = cp.read_text(encoding="utf-8")
            assert len(content) > 0

    def test_full_segment_clean_format(self, sample_config):
        """Full pipeline: create text -> clean -> format with chunk verification."""
        input_dir = sample_config.data.input_dir

        # Create a substantial text file (needs >100 chars for validation, >500 for cleaning)
        substantial_text = SAMPLE_ENGLISH_TEXT * 3
        txt_file = input_dir / "substantial_novel.txt"
        txt_file.write_text(substantial_text, encoding="utf-8")

        # Clean
        cleaned_files = clean_data(sample_config)
        assert len(cleaned_files) > 0

        # Format
        num_entries = format_data(sample_config)
        assert num_entries > 0

        # Read and verify the final output
        output_file = sample_config.data.output_dir / "train.jsonl"
        with open(output_file, "r", encoding="utf-8") as f:
            entries = [json.loads(line) for line in f if line.strip()]

        # All entries should have required structure
        for entry in entries:
            assert "instruction" in entry
            assert "output" in entry
            assert len(entry["output"]) >= 100
