from functools import lru_cache
from pydantic import AliasChoices, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Anthropic (preferred)
    anthropic_api_key: SecretStr = Field(
        default=SecretStr(""),
        validation_alias=AliasChoices("ANTHROPIC_API_KEY", "anthropic_api_key"),
    )

    # Azure OpenAI (fallback)
    azure_openai_key: SecretStr = Field(
        default=SecretStr(""),
        validation_alias=AliasChoices("AZURE_OPENAI_KEY", "azure_openai_key", "OPENAI_API_KEY"),
    )
    azure_openai_endpoint: str = Field(
        default="",
        validation_alias=AliasChoices("AZURE_OPENAI_ENDPOINT", "azure_openai_endpoint", "OPENAI_BASE_URL"),
    )
    azure_openai_deployment: str = Field(
        default="gpt-4o",
        validation_alias=AliasChoices("AZURE_OPENAI_DEPLOYMENT", "azure_openai_deployment"),
    )
    azure_openai_api_version: str = Field(
        default="2024-08-01-preview",
        validation_alias=AliasChoices("AZURE_OPENAI_API_VERSION", "azure_openai_api_version"),
    )

    # Generic OpenAI-compatible backend
    openai_api_key: SecretStr = Field(
        default=SecretStr(""),
        validation_alias=AliasChoices("OPENAI_API_KEY", "openai_api_key"),
    )
    openai_base_url: str = Field(
        default="",
        validation_alias=AliasChoices("OPENAI_BASE_URL", "openai_base_url"),
    )
    openai_model: str = Field(
        default="gpt-4o-mini",
        validation_alias=AliasChoices("OPENAI_MODEL", "openai_model"),
    )

    # Anthropic model to use
    anthropic_model: str = Field(
        default="claude-sonnet-4-6",
        validation_alias=AliasChoices("ANTHROPIC_MODEL", "anthropic_model"),
    )

    # Langfuse
    langfuse_public_key: SecretStr = Field(
        default=SecretStr(""),
        validation_alias=AliasChoices("LANGFUSE_PUBLIC_KEY", "langfuse_public_key"),
    )
    langfuse_secret_key: SecretStr = Field(
        default=SecretStr(""),
        validation_alias=AliasChoices("LANGFUSE_SECRET_KEY", "langfuse_secret_key"),
    )
    langfuse_host: str = Field(
        default="https://cloud.langfuse.com",
        validation_alias=AliasChoices("LANGFUSE_HOST", "langfuse_host"),
    )

    # Dataset
    dataset_raw_dir: str = Field(
        default="dataset",
        validation_alias=AliasChoices("DATASET_RAW_DIR", "dataset_raw_dir"),
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()  # type: ignore[call-arg]
