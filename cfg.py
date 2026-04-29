from dataclasses import dataclass
from typing import Optional


@dataclass
class LLMConfig:
    """Конфигурация для LLM"""
    provider: str = "deepseek"
    model: str = "deepseek-chat"
    api_key: Optional[str] = None
    base_url: str = "https://api.deepseek.com"
    temperature: float = 0.0
    max_tokens: int = 2000  # Добавляем поле