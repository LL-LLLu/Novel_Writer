import pytest
from pathlib import Path

def test_mixer_initialization():
    from novel_writer.processing.mix import StyleMixer

    with pytest.raises(Exception):  # Will fail to load actual model
        mixer = StyleMixer("fake/model/path")

def test_weight_validation():
    from novel_writer.processing.mix import StyleMixer

    # Mocking init to bypass model loading
    class MockMixer(StyleMixer):
        def __init__(self):
            self.base_model_path = "test"

    mixer = MockMixer()

    with pytest.raises(ValueError):
        mixer.merge_loras(
            [Path("lora1"), Path("lora2")],
            [0.5]  # Wrong number of weights
        )

    with pytest.raises(ValueError):
        mixer.merge_loras(
            [Path("lora1"), Path("lora2")],
            [0.3, 0.3]  # Doesn't sum to 1.0
        )
