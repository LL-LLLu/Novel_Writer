from pathlib import Path
from typing import Optional
from pydantic import BaseModel, Field, validator
import yaml

class DataConfig(BaseModel):
    input_dir: Path = Field(default=Path("data/raw"))
    output_dir: Path = Field(default=Path("data/processed"))
    temp_dir: Path = Field(default=Path("data/processed/temp_cleaned"))
    chunk_size: int = Field(default=4000, gt=0)
    overlap: int = Field(default=500, ge=0)

class TrainingConfig(BaseModel):
    base_model: str = Field(default="unsloth/llama-3-8b-instruct-bnb-4bit")
    epochs: int = Field(default=1, gt=0)
    learning_rate: float = Field(default=2e-4, gt=0)
    max_steps: Optional[int] = Field(default=60)
    batch_size: int = Field(default=2, gt=0)
    gradient_accumulation_steps: int = Field(default=4, gt=0)

class ModelConfig(BaseModel):
    context_length: int = Field(default=8192, gt=0)
    lora_rank: int = Field(default=32, gt=0)
    lora_alpha: int = Field(default=64, gt=0)
    lora_dropout: float = Field(default=0.0, ge=0, le=1)

class Config(BaseModel):
    data: DataConfig = Field(default_factory=DataConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    log_level: str = Field(default="INFO")

    @classmethod
    def from_yaml(cls, path: Path) -> "Config":
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return cls(**data)

    def to_yaml(self, path: Path):
        with open(path, "w") as f:
            yaml.dump(self.dict(), f, default_flow_style=False)
