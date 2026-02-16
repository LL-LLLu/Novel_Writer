import pytest
from pathlib import Path
import tempfile

from novel_writer.processing.segment import ChapterSegmenter, segment_directory

def test_segmentation():
    text = (
        "Chapter 1: The Beginning\n"
        + "This is the first section with some content. " * 20 + "\n\n"
        + "Chapter 2: The Middle\n"
        + "This is the second section with some content. " * 20 + "\n\n"
        + "Chapter 3: The End\n"
        + "This is the third section with some content. " * 20
    )

    segmenter = ChapterSegmenter(min_chapter_length=10)
    chapters = segmenter.segment_text(text)

    assert len(chapters) == 3
    assert chapters[0][0] == "Chapter 1"
    assert "first section" in chapters[0][1]

def test_chinese_segmentation():
    text = (
        "第1章 开始\n"
        + "这是第一章的内容。" * 50 + "\n\n"
        + "第2章 继续\n"
        + "这是第二章的内容。" * 50
    )

    segmenter = ChapterSegmenter(min_chapter_length=5)
    chapters = segmenter.segment_text(text)

    assert len(chapters) == 2
    assert "第1章" in chapters[0][0]

def test_segment_directory(tmp_path):
    # Create test file
    test_file = tmp_path / "test.txt"
    test_file.write_text("Chapter 1\n" + "Content. " * 100)

    chapters = segment_directory(tmp_path, min_chapter_length=10)

    assert len(chapters) == 1
    assert chapters[0].exists()
