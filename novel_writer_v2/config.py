"""Configuration dataclasses for the multi-agent story generator."""

from dataclasses import dataclass, field


@dataclass
class GeminiConfig:
    api_key: str = ""
    model: str = "gemini-3.1-pro-preview"
    temperature: float = 0.7
    max_output_tokens: int = 4096


@dataclass
class QwenConfig:
    api_key: str = ""
    model: str = "qwen3.5-plus"
    base_url: str = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
    temperature: float = 0.8
    max_tokens: int = 4096


@dataclass
class GenerationConfig:
    max_debate_rounds: int = 3
    chapter_min_chars: int = 5000
    chapter_max_chars: int = 8000
    language: str = "auto"


@dataclass
class AppConfig:
    gemini: GeminiConfig = field(default_factory=GeminiConfig)
    qwen: QwenConfig = field(default_factory=QwenConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
