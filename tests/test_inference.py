import pytest
from unittest.mock import Mock, patch
from pathlib import Path

def test_generator_initialization():
    from novel_writer.inference import NovelGenerator

    # Mock model loading (since we don't have actual models)
    with patch('novel_writer.inference.AutoModelForCausalLM') as mock_model:
        mock_model.from_pretrained.return_value = Mock()

        with patch('novel_writer.inference.AutoTokenizer') as mock_tokenizer:
            mock_tokenizer.from_pretrained.return_value = Mock()

            generator = NovelGenerator(
                base_model_path="test/model",
                lora_path=None,
                device="cpu"
            )

            assert generator.model is not None
            assert generator.tokenizer is not None
