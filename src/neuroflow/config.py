"""Configuration management for NeuroFlow AI using pydantic-settings."""

import os
from pathlib import Path
from typing import Literal
from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
ENV_FILE = REPO_ROOT / ".env"


class Settings(BaseSettings):
    """Application settings loaded from environment variables and .env file."""

    # LLM Settings
    llm_provider: Literal["fake", "ollama", "openai", "anthropic", "openrouter"] = Field(
        default="fake",
        description="LLM provider backend (fake for deterministic testing/dev, openrouter/ollama/openai/anthropic for live models)",
        alias="LLM_PROVIDER",
    )
    llm_model: str = Field(
        default="mock-model",
        description="Model name/identifier to use with the selected provider",
        alias="LLM_MODEL",
    )
    llm_temperature: float = Field(
        default=0.0,
        ge=0.0,
        le=2.0,
        description="Sampling temperature (0.0 for deterministic planning)",
        alias="LLM_TEMPERATURE",
    )
    openrouter_api_key: str | None = Field(
        default=None,
        description="API key for OpenRouter models",
        alias="OPENROUTER_API_KEY",
    )
    openrouter_base_url: str = Field(
        default="https://openrouter.ai/api/v1",
        description="Base URL for OpenRouter API",
        alias="OPENROUTER_BASE_URL",
    )
    openrouter_reasoning: bool = Field(
        default=True,
        description="Whether to enable reasoning for OpenRouter models that support it",
        alias="OPENROUTER_REASONING",
    )
    openai_api_key: str | None = Field(
        default=None,
        description="API key for OpenAI models",
        alias="OPENAI_API_KEY",
    )
    anthropic_api_key: str | None = Field(
        default=None,
        description="API key for Anthropic models",
        alias="ANTHROPIC_API_KEY",
    )
    ollama_base_url: str = Field(
        default="http://localhost:11434",
        description="Base URL for local Ollama instance",
        alias="OLLAMA_BASE_URL",
    )

    # Persistence
    sqlite_db_path: str = Field(
        default="data/neuroflow.db",
        description="Path to SQLite database file",
        alias="SQLITE_DB_PATH",
    )

    # UI Settings (Milestone 4+)
    gradio_server_port: int = Field(
        default=7860,
        description="Port for Gradio UI server",
        alias="GRADIO_SERVER_PORT",
    )
    gradio_server_name: str = Field(
        default="127.0.0.1",
        description="Host address for Gradio UI server",
        alias="GRADIO_SERVER_NAME",
    )

    @model_validator(mode="after")
    def infer_provider(self) -> "Settings":
        """Auto-select openrouter when an OpenRouter API key is provided and not forced to fake."""
        if self.openrouter_api_key and self.llm_provider in ("ollama", "fake"):
            if os.environ.get("LLM_PROVIDER") != "fake":
                object.__setattr__(self, "llm_provider", "openrouter")
                if self.llm_model == "mock-model":
                    object.__setattr__(self, "llm_model", "google/gemma-4-26b-a4b-it:free")
        return self

    model_config = SettingsConfigDict(
        env_file=ENV_FILE,
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )


def get_settings(reload: bool = False) -> Settings:
    """Return application settings, always reflecting current .env and environment."""
    return Settings()
