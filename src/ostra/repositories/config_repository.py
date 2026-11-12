import json
import logging
import os
from pathlib import Path

from platformdirs import user_config_path
from pydantic import ValidationError
from pydantic_settings import EnvSettingsSource

from ostra.constants import APP_NAME, OWNER_ONLY_DIRECTORY_MODE, OWNER_ONLY_FILE_MODE
from ostra.models.app_config import AppConfig

# Earlier versions kept the config file here. An existing file at this path stays in use.
LEGACY_CONFIG_PATH = Path.home() / "ostra" / "config.json"
DEFAULT_CONFIG_PATH = user_config_path(APP_NAME) / "config.json"

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
