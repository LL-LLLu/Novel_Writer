import pytest
from pathlib import Path
import tempfile
from unittest.mock import Mock, patch

def test_exporter_initialization():
    from novel_writer.export import ModelExporter

    with patch('novel_writer.export.exporter.AutoModelForCausalLM'), \
         patch('novel_writer.export.exporter.AutoTokenizer'), \
         patch('novel_writer.export.exporter.PeftModel'):
        
        exporter = ModelExporter(
            base_model_path="test/model",
            lora_path=None
        )

        assert exporter is not None
