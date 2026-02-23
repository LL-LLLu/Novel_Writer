from functools import lru_cache

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables and .env file."""

    GEMINI_API_KEY: str = ""
    QWEN_API_KEY: str = ""
    GEMINI_MODEL: str = "gemini-3.1-pro-preview"
    QWEN_MODEL: str = "qwen3.5-plus"
    QWEN_BASE_URL: str = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
    DATABASE_URL: str = "sqlite:///./novel_writer.db"
    GEMINI_TEMPERATURE: float = 0.7
    QWEN_TEMPERATURE: float = 0.8
    MAX_OUTPUT_TOKENS: int = 4096
    MAX_DEBATE_ROUNDS: int = 3
    CHAPTER_MIN_CHARS: int = 5000
    CHAPTER_MAX_CHARS: int = 8000

    model_config = {"env_file": ".env"}


@lru_cache
def get_settings() -> Settings:
    """Return cached application settings."""
    return Settings()
