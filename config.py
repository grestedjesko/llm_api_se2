from __future__ import annotations

import logging
from functools import lru_cache

try:
    from pydantic_settings import BaseSettings, SettingsConfigDict
except ImportError:
    from pydantic import BaseSettings
    SettingsConfigDict = None

from pydantic import Field


class Settings(BaseSettings):
    if SettingsConfigDict:
        model_config = SettingsConfigDict(
            env_file=".env",
            env_file_encoding="utf-8",
            env_prefix="",
        )
    else:
        class Config:
            env_file = ".env"
            env_file_encoding = "utf-8"

    model_name: str = Field(
        default="Qwen/Qwen3-4B",
        validation_alias="LLM_MODEL_NAME",
    )
    max_tokens_default: int = Field(
        default=256,
        validation_alias="LLM_MAX_TOKENS",
    )
    temperature_default: float = Field(
        default=0.7,
        validation_alias="LLM_TEMPERATURE",
    )
    top_p_default: float = Field(
        default=0.9,
        validation_alias="LLM_TOP_P",
    )
    log_level: str = Field(
        default="INFO",
        validation_alias="LOG_LEVEL",
    )


@lru_cache()
def get_settings() -> Settings:
    settings = Settings()
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    logging.getLogger(__name__).info("Settings loaded (model=%s)", settings.model_name)
    return settings


settings = get_settings()


