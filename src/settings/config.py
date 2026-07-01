from functools import lru_cache
from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # Anthropic (preferred)
    anthropic_api_key: SecretStr = SecretStr("")

    # Azure OpenAI (fallback)
    azure_openai_key: SecretStr = SecretStr("")
    azure_openai_endpoint: str = ""
    azure_openai_deployment: str = "gpt-4o"
    azure_openai_api_version: str = "2024-08-01-preview"

    # Anthropic model to use
    anthropic_model: str = "claude-sonnet-4-6"

    # Langfuse
    langfuse_public_key: SecretStr = SecretStr("")
    langfuse_secret_key: SecretStr = SecretStr("")
    langfuse_host: str = "https://cloud.langfuse.com"

    # Dataset
    dataset_raw_dir: str = "dataset"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()  # type: ignore[call-arg]
