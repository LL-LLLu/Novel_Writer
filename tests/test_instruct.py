import pytest
import json
import tempfile
from pathlib import Path

from novel_writer.processing.instruct import InstructionGenerator

def test_rule_based_generation():
    generator = InstructionGenerator(use_llm=False)
    text = "This is a test text. It has some content. " * 20

    entries = generator.generate_from_chunk(text, num_instructions=2)

    assert len(entries) == 2
    for entry in entries:
        assert 'instruction' in entry
        assert 'output' in entry
        assert entry['output'] == text

def test_summary_generation():
    generator = InstructionGenerator(use_llm=False)
    text = "Once upon a time, there was a hero. He fought many battles."

    summary = generator.generate_summary(text)

    assert "hero" in summary.lower() or "battle" in summary.lower()
    assert len(summary) <= 200
