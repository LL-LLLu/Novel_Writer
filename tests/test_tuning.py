import pytest
from pathlib import Path
import tempfile

from novel_writer.tuning import HyperparameterTuner

def test_hyperparameter_suggestion():
    tuner = HyperparameterTuner(study_name="test")

    # Mock trial
    class MockTrial:
        def __init__(self):
            self._trial_id = 0
            self.number = 0

        def suggest_float(self, name, low, high, log=False):
            return low

        def suggest_int(self, name, low, high):
            return low

        def suggest_categorical(self, name, choices):
            return choices[0]

    trial = MockTrial()
    params = tuner.suggest_hyperparameters(trial)

    assert 'learning_rate' in params
    assert 'lora_rank' in params
    assert params['lora_rank'] >= 8
