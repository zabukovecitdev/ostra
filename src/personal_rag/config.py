import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

from platformdirs import user_documents_path

CONFIG_PATH = Path.home() / "personal-rag" / "config.json"
DEFAULT_DATABASE_PATH_TEXT = "~/personal-rag/personal_rag.db"
DEFAULT_DATABASE_PATH = Path(DEFAULT_DATABASE_PATH_TEXT).expanduser()
OWNER_ONLY_DIRECTORY_MODE = 0o700
OWNER_ONLY_FILE_MODE = 0o600

logger = logging.getLogger(__name__)


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


def load_config(config_path: Path = CONFIG_PATH) -> AppConfig:
    """Read the application settings from the JSON config file.

    Create the config file with the default settings when it does not exist.
    Each missing key uses its default value. Paths accept a leading ~ for the home directory.

    Args:
        config_path: Location of the JSON config file.

    Returns:
        The application settings.

    Raises:
        ValueError: When the config file does not contain valid JSON or when directories is not a list.
    """
    if not config_path.exists():
        logger.info("Config file not found. Creating it. Path=%s", config_path)
        _write_default_config(config_path)
        return AppConfig(database_path=DEFAULT_DATABASE_PATH, directories=default_directories())

    try:
        raw_config = json.loads(config_path.read_text())
    except json.JSONDecodeError as error:
        raise ValueError(f"Config file is not valid JSON. Path={config_path}") from error

    database_path = Path(raw_config.get("database_path", DEFAULT_DATABASE_PATH_TEXT)).expanduser()
    raw_directories = raw_config.get("directories", [str(directory) for directory in default_directories()])
    if not isinstance(raw_directories, list):
        raise ValueError(f"Config key directories must be a list. Path={config_path}")

    directories = [Path(raw_directory).expanduser() for raw_directory in raw_directories]
    return AppConfig(database_path=database_path, directories=directories)


def _write_default_config(config_path: Path) -> None:
    """Write the default settings to a new config file with owner-only permissions.

    Create the parent directory when it is missing. Set owner-only permissions on the directory and the file.

    Args:
        config_path: Location of the JSON config file to create.
    """
    config_path.parent.mkdir(mode=OWNER_ONLY_DIRECTORY_MODE, parents=True, exist_ok=True)
    config_path.parent.chmod(OWNER_ONLY_DIRECTORY_MODE)

    default_config = {
        "database_path": DEFAULT_DATABASE_PATH_TEXT,
        "directories": [str(directory) for directory in default_directories()],
    }
    file_descriptor = os.open(config_path, os.O_CREAT | os.O_WRONLY | os.O_TRUNC, OWNER_ONLY_FILE_MODE)
    with os.fdopen(file_descriptor, "w") as config_file:
        json.dump(default_config, config_file, indent=2)
        config_file.write("\n")
