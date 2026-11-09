import json
import logging
import os
from pathlib import Path

from platformdirs import user_config_path, user_data_path
from pydantic import Field, ValidationError, field_validator
from pydantic_settings import BaseSettings, EnvSettingsSource, SettingsConfigDict

from ostra.models.provider import EmbeddingProviderName
from ostra.models.registry import DEFAULT_EMBEDDING_MODEL, EMBEDDING_MODELS
from ostra.store.database_backend import DatabaseBackend

APP_NAME = "ostra"
# Earlier versions kept the config file here. An existing file at this path stays in use.
LEGACY_CONFIG_PATH = Path.home() / "ostra" / "config.json"
DEFAULT_CONFIG_PATH = user_config_path(APP_NAME) / "config.json"
DEFAULT_DATABASE_PATH = user_data_path(APP_NAME) / "ostra.db"
DEFAULT_DATABASE_BACKEND = DatabaseBackend.SQLITE
DEFAULT_EMBEDDING_PROVIDER = EmbeddingProviderName.OLLAMA
DEFAULT_CHUNK_OVERLAP_TOKENS = 50
OWNER_ONLY_DIRECTORY_MODE = 0o700
OWNER_ONLY_FILE_MODE = 0o600

NO_DIRECTORIES_MESSAGE = "No folders are configured. Add folders to the directories list in the config file."

logger = logging.getLogger(__name__)


def default_config_path() -> Path:
    """Return the location of the config file that the application reads by default.

    Returns:
        The legacy config file at ~/ostra/config.json when it exists. Otherwise the config file in the
        user config folder of the current operating system.
    """
    if LEGACY_CONFIG_PATH.exists():
        return LEGACY_CONFIG_PATH

    return DEFAULT_CONFIG_PATH


class AppConfig(BaseSettings):
    """Hold the application settings.

    Each setting comes from the first of these that sets it: a keyword argument, an environment variable with
    the OSTRA_ prefix, the config file, the default value. For example OSTRA_DATABASE_BACKEND=chroma selects
    Chroma for one run. A list setting such as OSTRA_DIRECTORIES takes a JSON list. Paths accept a leading ~
    for the home directory.
    """

    model_config = SettingsConfigDict(env_prefix="OSTRA_", frozen=True, extra="ignore")

    database_path: Path = DEFAULT_DATABASE_PATH
    directories: list[Path] = Field(default_factory=list)
    database_backend: DatabaseBackend = DEFAULT_DATABASE_BACKEND
    # None means the default chunk size of the embedding model.
    chunk_size_tokens: int | None = None
    chunk_overlap_tokens: int = DEFAULT_CHUNK_OVERLAP_TOKENS
    embedding_provider: EmbeddingProviderName = DEFAULT_EMBEDDING_PROVIDER
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    # None means the OLLAMA_BASE_URL environment variable or the local default applies.
    ollama_base_url: str | None = None

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


def load_config(config_path: Path | None = None) -> AppConfig:
    """Read the application settings from the JSON config file and the OSTRA_ environment variables.

    Create the config file with the default settings when it does not exist. Each missing key uses its default
    value. An environment variable overrides the key of the same name in the file.

    Args:
        config_path: Location of the JSON config file. Defaults to the path from default_config_path.

    Returns:
        The application settings.

    Raises:
        ValueError: When the config file does not contain a valid JSON object or when a setting is not valid.
    """
    if config_path is None:
        config_path = default_config_path()

    if not config_path.exists():
        logger.info("Config file not found. Creating it. Path=%s", config_path)
        _write_default_config(config_path)

    try:
        file_values = json.loads(config_path.read_text())
    except json.JSONDecodeError as error:
        raise ValueError(f"Config file is not valid JSON. Path={config_path}") from error

    if not isinstance(file_values, dict):
        raise ValueError(f"Config file must contain a JSON object. Path={config_path}")

    # Keyword arguments outrank environment variables, so merge the environment over the file values here.
    environment_values = EnvSettingsSource(AppConfig)()
    try:
        return AppConfig(**{**file_values, **environment_values})
    except ValidationError as error:
        raise ValueError(f"Config file has invalid settings. Path={config_path}\n{error}") from error


def save_config(config: AppConfig, config_path: Path | None = None) -> None:
    """Write the application settings to the JSON config file with owner-only permissions.

    Replace the existing file. Create the parent directory when it is missing.

    Args:
        config: The settings to write.
        config_path: Location of the JSON config file. Defaults to the path from default_config_path.
    """
    if config_path is None:
        config_path = default_config_path()

    # Leave out the unset optional keys, so that a model change takes its own chunk size.
    values = config.model_dump(mode="json", exclude_none=True)
    _write_config_file(config_path, values)


def _write_default_config(config_path: Path) -> None:
    """Write the default settings to a new config file with owner-only permissions.

    Args:
        config_path: Location of the JSON config file to create.
    """
    # model_construct skips the environment variables, so that they are not written into the file.
    save_config(AppConfig.model_construct(), config_path)


def _write_config_file(config_path: Path, values: dict) -> None:
    """Write values as JSON to the config file with owner-only permissions.

    Create the parent directory when it is missing. Set owner-only permissions on the directory and the file.

    Args:
        config_path: Location of the JSON config file.
        values: The JSON object to write.
    """
    config_path.parent.mkdir(mode=OWNER_ONLY_DIRECTORY_MODE, parents=True, exist_ok=True)
    config_path.parent.chmod(OWNER_ONLY_DIRECTORY_MODE)

    file_descriptor = os.open(config_path, os.O_CREAT | os.O_WRONLY | os.O_TRUNC, OWNER_ONLY_FILE_MODE)
    with os.fdopen(file_descriptor, "w") as config_file:
        json.dump(values, config_file, indent=2)
        config_file.write("\n")
