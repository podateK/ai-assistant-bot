from __future__ import annotations

import enum
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class AIProvider(str, enum.Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    LOCAL = "local"


class ImageProvider(str, enum.Enum):
    DALLE = "dalle"
    STABLE_DIFFUSION = "stable_diffusion"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    discord_token: str = Field(default="")
    discord_application_id: str = Field(default="")
    default_prefix: str = Field(default="!")
    default_ai_model: str = Field(default="openai")
    ai_admin_ids: list[int] = Field(default_factory=list)

    openai_api_key: str = Field(default="")
    openai_org_id: str = Field(default="")
    openai_model: str = Field(default="gpt-4o")
    openai_vision_model: str = Field(default="gpt-4o")
    openai_max_tokens: int = Field(default=4096)
    openai_temperature: float = Field(default=0.7)

    anthropic_api_key: str = Field(default="")
    anthropic_model: str = Field(default="claude-sonnet-4-20250514")
    anthropic_max_tokens: int = Field(default=4096)
    anthropic_temperature: float = Field(default=0.7)

    local_api_url: str = Field(default="http://localhost:11434/api/generate")
    local_model: str = Field(default="llama3")
    local_api_key: str = Field(default="")

    dall_e_model: str = Field(default="dall-e-3")
    dall_e_size: str = Field(default="1024x1024")
    dall_e_quality: str = Field(default="standard")
    image_gen_provider: ImageProvider = Field(default=ImageProvider.DALLE)

    database_path: str = Field(default="data/bot.db")
    max_conversation_history: int = Field(default=50)
    conversation_summary_threshold: int = Field(default=30)
    rate_limit_per_minute: int = Field(default=10)
    rate_limit_burst: int = Field(default=3)
    token_tracking_enabled: bool = Field(default=True)
    log_level: str = Field(default="INFO")

    @property
    def database_url(self) -> str:
        return f"sqlite+aiosqlite:///{self.database_path}"

    @property
    def data_dir(self) -> Path:
        return Path(self.database_path).parent

    @property
    def available_providers(self) -> list[AIProvider]:
        providers = []
        if self.openai_api_key:
            providers.append(AIProvider.OPENAI)
        if self.anthropic_api_key:
            providers.append(AIProvider.ANTHROPIC)
        if self.local_api_url:
            providers.append(AIProvider.LOCAL)
        return providers


def parse_admin_ids(raw: str) -> list[int]:
    if not raw:
        return []
    return [int(uid.strip()) for uid in raw.split(",") if uid.strip().isdigit()]


def load_settings() -> Settings:
    settings = Settings()
    raw = Settings.model_fields.get("ai_admin_ids")
    if isinstance(settings.ai_admin_ids, str):
        settings.ai_admin_ids = parse_admin_ids(settings.ai_admin_ids)
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    return settings
