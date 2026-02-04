import pytest
import json
import tempfile
from pathlib import Path

from novel_writer.processing.deduplicate import TextDeduplicator

def test_exact_duplicate():
    deduplicator = TextDeduplicator()

    text = "This is a test text with some content."
    texts = [("1", text), ("2", text), ("3", text)]

    unique = deduplicator.deduplicate_texts(texts)

    assert len(unique) == 1
    assert unique[0][1] == text

def test_near_duplicate():
    deduplicator = TextDeduplicator(threshold=0.85)

    text = "This is a test text with some content."
    similar = "This is a test text with extra content."  # 93% similar

    texts = [("1", text), ("2", similar)]
    unique = deduplicator.deduplicate_texts(texts)

    assert len(unique) == 1

def test_no_duplicate():
    deduplicator = TextDeduplicator()

    text1 = "First completely different text."
    text2 = "Second very different content here."

    texts = [("1", text1), ("2", text2)]
    unique = deduplicator.deduplicate_texts(texts)

    assert len(unique) == 2
