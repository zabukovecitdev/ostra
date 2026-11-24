import json
import os
from pathlib import Path

from platformdirs import user_config_path
from pydantic import ValidationError

from ostra.constants import APP_NAME
from ostra.errors.config_not_found_error import ConfigNotFoundError
from ostra.models.app_config import AppConfig
from ostra.utils.owner_only import ensure_owner_only_directory, open_owner_only_file

# Earlier versions kept the config file here. An existing file at this path stays in use.
LEGACY_CONFIG_PATH = Path.home() / "ostra" / "config.json"
DEFAULT_CONFIG_PATH = user_config_path(APP_NAME) / "config.json"


def default_config_path() -> Path:
    """Return the location of the config file that the application reads by default.

    Returns:
        The legacy config file at ~/ostra/config.json when it exists. Otherwise the config file in the
        user config folder of the current operating system.
    """
    if LEGACY_CONFIG_PATH.exists():
        return LEGACY_CONFIG_PATH

    return DEFAULT_CONFIG_PATH


def load_config(config_path: Path | None = None) -> AppConfig:
    """Read the application settings from the JSON config file and the OSTRA_ environment variables.

    Each missing key uses its default value. An environment variable overrides the key of the same name in the file.

    Args:
        config_path: Location of the JSON config file. Defaults to the path from default_config_path.

    Returns:
        The application settings.

    Raises:
        ConfigNotFoundError: When the config file does not exist. Nothing is created.
        ValueError: When the config file does not contain a valid JSON object or when a setting is not valid.
    """
    if config_path is None:
        config_path = default_config_path()

    if not config_path.exists():
        raise ConfigNotFoundError(config_path)

    file_values = _read_file_values(config_path)
    try:
        return AppConfig(**file_values)
    except ValidationError as error:
        raise ValueError(f"Config file has invalid settings. Path={config_path}\n{error}") from error


def read_config_file(config_path: Path) -> AppConfig | None:
    """Read the settings of the config file alone, without the OSTRA_ environment variables.

    Unlike load_config, it does not create a missing file.

    Args:
        config_path: Location of the JSON config file.

    Returns:
        The settings of the file, or None when the file does not exist.

    Raises:
        ValueError: When the config file does not contain a valid JSON object or when a setting is not valid.
    """
    if not config_path.exists():
        return None

    file_values = _read_file_values(config_path)
    try:
        return AppConfig.from_file_values(file_values)
    except ValidationError as error:
        raise ValueError(f"Config file has invalid settings. Path={config_path}\n{error}") from error


def _read_file_values(config_path: Path) -> dict:
    """Read the JSON object of the config file.

    Args:
        config_path: Location of the JSON config file.

    Returns:
        The keys and values of the file.

    Raises:
        ValueError: When the file does not contain a valid JSON object.
    """
    try:
        file_values = json.loads(config_path.read_text())
    except json.JSONDecodeError as error:
        raise ValueError(f"Config file is not valid JSON. Path={config_path}") from error

    if not isinstance(file_values, dict):
        raise ValueError(f"Config file must contain a JSON object. Path={config_path}")

    return file_values


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


def _write_config_file(config_path: Path, values: dict) -> None:
    """Write values as JSON to the config file with owner-only permissions.

    Create the parent directory when it is missing. Set owner-only permissions on the directory and the file.

    Args:
        config_path: Location of the JSON config file.
        values: The JSON object to write.
    """
    ensure_owner_only_directory(config_path.parent)
    file_descriptor = open_owner_only_file(config_path, truncate=True)
    with os.fdopen(file_descriptor, "w") as config_file:
        json.dump(values, config_file, indent=2)
        config_file.write("\n")
