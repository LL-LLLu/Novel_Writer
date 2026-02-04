import pytest

from novel_writer.processing.filter import QualityFilter

def test_quality_scoring():
    filter_obj = QualityFilter()

    # Good quality text
    good_text = '"Hello," she said. "How are you today?"\n\n' * 20
    score, components = filter_obj.score(good_text)

    assert score > 0.5
    assert 'dialogue' in components
    assert 'variety' in components

def test_short_text():
    filter_obj = QualityFilter(min_length=1000)

    short_text = "Too short."
    score, components = filter_obj.score(short_text)

    assert score == 0.0

def test_filtering():
    filter_obj = QualityFilter()

    entries = [
        {"output": "Good " * 500 + '"dialogue"'},
        {"output": "bad" * 10},  # Too short
        {"output": "GOOD " * 500 + '"dialogue"'},
    ]

    filtered = filter_obj.filter_entries(entries, keep_ratio=1.0)

    # Should filter out short text
    assert len(filtered) == 2
