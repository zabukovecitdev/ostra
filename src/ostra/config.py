import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

from platformdirs import user_config_path, user_data_path, user_documents_path

from ostra.connectors.database_facade import DatabaseBackend

APP_NAME = "ostra"
# Earlier versions kept the config file here. An existing file at this path stays in use.
LEGACY_CONFIG_PATH = Path.home() / "ostra" / "config.json"
DEFAULT_CONFIG_PATH = user_config_path(APP_NAME) / "config.json"
DEFAULT_DATABASE_PATH = user_data_path(APP_NAME) / "ostra.db"
# The model accepts 512 tokens. Two of them are the special tokens [CLS] and [SEP].
DEFAULT_DATABASE_BACKEND = DatabaseBackend.SQLITE
DEFAULT_CHUNK_SIZE_TOKENS = 510
DEFAULT_CHUNK_OVERLAP_TOKENS = 50
OWNER_ONLY_DIRECTORY_MODE = 0o700
OWNER_ONLY_FILE_MODE = 0o600

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


def default_directories() -> list[Path]:
    """Return the default directories that the RAG service processes.

    Returns:
        A list that contains the user Documents directory of the current operating system.
    """
    return [user_documents_path()]


@dataclass(frozen=True)
class AppConfig:
    """Hold the application settings read from the config file."""

    database_path: Path
    directories: list[Path] = field(default_factory=default_directories)
    database_backend: DatabaseBackend = DEFAULT_DATABASE_BACKEND
    chunk_size_tokens: int = DEFAULT_CHUNK_SIZE_TOKENS
    chunk_overlap_tokens: int = DEFAULT_CHUNK_OVERLAP_TOKENS


def load_config(config_path: Path | None = None) -> AppConfig:
    """Read the application settings from the JSON config file.

    Create the config file with the default settings when it does not exist.
    Each missing key uses its default value. Paths accept a leading ~ for the home directory.

    Args:
        config_path: Location of the JSON config file. Defaults to the path from default_config_path.

    Returns:
        The application settings.

    Raises:
        ValueError: When the config file does not contain valid JSON or when directories is not a list.
    """
    if config_path is None:
        config_path = default_config_path()

    if not config_path.exists():
        logger.info("Config file not found. Creating it. Path=%s", config_path)
        _write_default_config(config_path)
        return AppConfig(
            database_path=DEFAULT_DATABASE_PATH,
            directories=default_directories(),
            database_backend=DEFAULT_DATABASE_BACKEND,
        )

    try:
        raw_config = json.loads(config_path.read_text())
    except json.JSONDecodeError as error:
        raise ValueError(f"Config file is not valid JSON. Path={config_path}") from error

    database_path = Path(raw_config.get("database_path", str(DEFAULT_DATABASE_PATH))).expanduser()
    raw_directories = raw_config.get("directories", [str(directory) for directory in default_directories()])
    if not isinstance(raw_directories, list):
        raise ValueError(f"Config key directories must be a list. Path={config_path}")

    directories = [Path(raw_directory).expanduser() for raw_directory in raw_directories]
    chunk_size_tokens = _read_int(raw_config, "chunk_size_tokens", DEFAULT_CHUNK_SIZE_TOKENS, config_path)
    chunk_overlap_tokens = _read_int(raw_config, "chunk_overlap_tokens", DEFAULT_CHUNK_OVERLAP_TOKENS, config_path)
    database_backend = _read_database_backend(raw_config.get("database_backend", DEFAULT_DATABASE_BACKEND.value))
    return AppConfig(
        database_path=database_path,
        directories=directories,
        database_backend=database_backend,
        chunk_size_tokens=chunk_size_tokens,
        chunk_overlap_tokens=chunk_overlap_tokens,
    )


def _read_database_backend(raw_backend: str) -> DatabaseBackend:
    """Convert the database_backend config value to a backend.

    Args:
        raw_backend: The value of the database_backend key.

    Returns:
        The database backend with that name.

    Raises:
        ValueError: When the value does not name a supported backend.
    """
    try:
        return DatabaseBackend(raw_backend)
    except ValueError as error:
        supported = ", ".join(backend.value for backend in DatabaseBackend)
        raise ValueError(f"Config key database_backend must be one of {supported}. Backend={raw_backend}") from error


def _read_int(raw_config: dict, key: str, default: int, config_path: Path) -> int:
    """Read an integer setting from the parsed config file.

    Args:
        raw_config: The parsed JSON object of the config file.
        key: The name of the setting.
        default: The value to return when the key is missing.
        config_path: Location of the JSON config file, used in the error message.

    Returns:
        The integer value of the setting.

    Raises:
        ValueError: When the value is not an integer.
    """
    value = raw_config.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"Config key {key} must be an integer. Path={config_path}")

    return value


def _write_default_config(config_path: Path) -> None:
    """Write the default settings to a new config file with owner-only permissions.

    Create the parent directory when it is missing. Set owner-only permissions on the directory and the file.

    Args:
        config_path: Location of the JSON config file to create.
    """
    config_path.parent.mkdir(mode=OWNER_ONLY_DIRECTORY_MODE, parents=True, exist_ok=True)
    config_path.parent.chmod(OWNER_ONLY_DIRECTORY_MODE)

    default_config = {
        "database_path": str(DEFAULT_DATABASE_PATH),
        "directories": [str(directory) for directory in default_directories()],
        "database_backend": DEFAULT_DATABASE_BACKEND.value,
        "chunk_size_tokens": DEFAULT_CHUNK_SIZE_TOKENS,
        "chunk_overlap_tokens": DEFAULT_CHUNK_OVERLAP_TOKENS,
    }
    file_descriptor = os.open(config_path, os.O_CREAT | os.O_WRONLY | os.O_TRUNC, OWNER_ONLY_FILE_MODE)
    with os.fdopen(file_descriptor, "w") as config_file:
        json.dump(default_config, config_file, indent=2)
        config_file.write("\n")
