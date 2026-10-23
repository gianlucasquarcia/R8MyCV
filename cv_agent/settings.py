import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    model: str = os.getenv("OLLAMA_MODEL", "qwen2.5vl")
    base_url: str = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    temperature: float = 0.2
    num_ctx: int = 8_192
    num_predict: int = 2_048
    repeat_penalty: float = 1.15
    max_pages: int = 3
    min_text_chars: int = 200
    max_text_chars: int = 12_000
    llm_retries: int = 2


settings = Settings()
