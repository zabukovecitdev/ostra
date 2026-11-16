from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource, SettingsConfigDict

from ostra.constants import DEFAULT_CHUNK_OVERLAP_TOKENS, DEFAULT_DATABASE_PATH
from ostra.enums.embedding_provider_name import EmbeddingProviderName
from ostra.models.embedding_model_registry import DEFAULT_EMBEDDING_MODEL, EMBEDDING_MODELS


class AppConfig(BaseSettings):
    """Hold the application settings.

    Each setting comes from the first of these that sets it: an environment variable with the OSTRA_ prefix, a
    keyword argument (the values of the config file), the default value. For example OSTRA_EMBEDDING_MODEL=bge-m3
    selects that model for one run. A list setting such as OSTRA_DIRECTORIES takes a JSON list. Paths accept a leading ~
    for the home directory.
    """

    model_config = SettingsConfigDict(env_prefix="OSTRA_", frozen=True, extra="ignore")

    database_path: Path = DEFAULT_DATABASE_PATH
    directories: list[Path] = Field(default_factory=list)
    # None means the default chunk size of the embedding model.
    chunk_size_tokens: int | None = None
    chunk_overlap_tokens: int = DEFAULT_CHUNK_OVERLAP_TOKENS
    embedding_provider: EmbeddingProviderName = EmbeddingProviderName.OLLAMA
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    # None means the OLLAMA_BASE_URL environment variable or the local default applies.
    ollama_base_url: str | None = None

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """Let an environment variable override the values that the config file passes as keyword arguments."""
        return env_settings, init_settings

    @field_validator("database_path")
    @classmethod
    def _expand_database_path(cls, database_path: Path) -> Path:
        """Expand a leading ~ in the database path to the home directory."""
        return database_path.expanduser()

    @field_validator("directories")
    @classmethod
    def _expand_directories(cls, directories: list[Path]) -> list[Path]:
        """Expand a leading ~ in each folder to the home directory."""
        return [directory.expanduser() for directory in directories]

    @field_validator("chunk_size_tokens", "chunk_overlap_tokens", mode="before")
    @classmethod
    def _reject_bool(cls, value: object) -> object:
        """Reject true and false, which would otherwise count as the integers 1 and 0."""
        if isinstance(value, bool):
            raise ValueError("must be an integer, not a boolean")

        return value

    @field_validator("embedding_model")
    @classmethod
    def _check_embedding_model(cls, embedding_model: str) -> str:
        """Accept only a model that the model registry knows."""
        if embedding_model not in EMBEDDING_MODELS:
            raise ValueError(f"must be one of {', '.join(EMBEDDING_MODELS)}")

        return embedding_model
