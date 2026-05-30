"""Env-driven configuration. See .env.example for the full list."""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """All runtime config in one place. Loaded from env / .env file."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Storage
    db_path: str = "./data/orders.db"
    index_path: str = "./data/orders.idx"

    # Models
    embed_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    llm_model_name: str = "gpt-4o-mini"

    # LLM client selection: "openai" or "fixture"
    llm_client: str = "openai"
    openai_api_key: str | None = None

    # Observability
    log_level: str = "INFO"
    prompt_version: str = "v1"

    # Hard limits
    sql_timeout_seconds: float = 5.0
    max_question_length: int = 500


def get_settings() -> Settings:
    """Singleton accessor. Importable from app.py and etl.py."""
    return Settings()  # type: ignore[call-arg]
