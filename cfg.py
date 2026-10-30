"""Application settings and LLM credentials loaded from the environment."""

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(PROJECT_ROOT / ".env", override=False)


@dataclass(frozen=True)
class Settings:
    """Keep uploaded documents and generated files outside the source tree."""

    data_dir: Path = field(
        default_factory=lambda: Path(
            os.getenv("BACHRAG_DATA_DIR", str(PROJECT_ROOT / "runtime"))
        ).expanduser().resolve()
    )

    @property
    def upload_dir(self) -> Path:
        return self.data_dir / "uploads"

    @property
    def index_path(self) -> Path:
        return self.data_dir / "brand.index"

    @property
    def data_path(self) -> Path:
        return self.data_dir / "chunks.json"

    @property
    def state_file(self) -> Path:
        return self.data_dir / "uploaded_files_state.json"


settings = Settings()


@dataclass
class LLMConfig:
    """Configuration for an OpenAI-compatible chat completion endpoint."""

    provider: str = "deepseek"
    model: str = "deepseek-chat"
    api_key: str | None = field(default=None, repr=False)
    base_url: str = "https://api.deepseek.com"
    temperature: float = 0.0
    max_tokens: int = 2000

    @classmethod
    def from_env(cls) -> "LLMConfig":
        provider = os.getenv("LLM_PROVIDER", "deepseek").strip().lower()
        defaults = {
            "deepseek": ("deepseek-chat", "https://api.deepseek.com", "DEEPSEEK_API_KEY"),
            "openai": ("gpt-4o-mini", "https://api.openai.com/v1", "OPENAI_API_KEY"),
        }
        model, base_url, key_name = defaults.get(provider, defaults["deepseek"])
        return cls(
            provider=provider,
            model=os.getenv("LLM_MODEL", model),
            api_key=os.getenv("LLM_API_KEY") or os.getenv(key_name) or None,
            base_url=os.getenv("LLM_BASE_URL", base_url),
        )
