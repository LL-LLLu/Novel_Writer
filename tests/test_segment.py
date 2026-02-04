import pytest
from pathlib import Path
import tempfile

from novel_writer.processing.segment import ChapterSegmenter, segment_directory

def test_segmentation():
    text = """
    Chapter 1: The Beginning
    This is the first chapter content.

    Chapter 2: The Middle
    This is the second chapter content.

    Chapter 3: The End
    This is the third chapter content.
    """

    segmenter = ChapterSegmenter(min_chapter_length=10)
    chapters = segmenter.segment_text(text)

    assert len(chapters) == 3
    assert chapters[0][0] == "Chapter 1: The Beginning"
    assert "first chapter" in chapters[0][1]

def test_chinese_segmentation():
    text = """
    第一章 开始
    这是第一章的内容。

    第二章 继续
    这是第二章的内容。
    """

    segmenter = ChapterSegmenter(min_chapter_length=5)
    chapters = segmenter.segment_text(text)

    assert len(chapters) == 2
    assert "第一章" in chapters[0][0]

def test_segment_directory(tmp_path):
    # Create test file
    test_file = tmp_path / "test.txt"
    test_file.write_text("Chapter 1\n" + "Content. " * 100)

    chapters = segment_directory(tmp_path, min_chapter_length=10)

    assert len(chapters) == 1
    assert chapters[0].exists()
