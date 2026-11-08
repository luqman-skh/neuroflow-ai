"""Configuration management for NeuroFlow AI using pydantic-settings."""

from typing import Literal
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables and .env file."""

    # LLM Settings
    llm_provider: Literal["fake", "ollama", "openai", "anthropic"] = Field(
        default="fake",
        description="LLM provider backend (fake for deterministic testing/dev, ollama/openai/anthropic for live models)",
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

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )


# Singleton settings instance
settings = Settings()


def get_settings() -> Settings:
    """Return the application settings instance."""
    return settings
