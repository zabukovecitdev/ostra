import json
from pathlib import Path

from platformdirs import user_data_path

from ostra.constants import APP_NAME, OWNER_ONLY_DIRECTORY_MODE

STATUS_PATH = user_data_path(APP_NAME) / "status.json"


def write_status(status_path: Path, status: dict) -> None:
    """Write the status of the last run to a JSON file, replacing the previous file.

    Args:
        status_path: Location of the JSON status file.
        status: The values to write.
    """
    status_path.parent.mkdir(mode=OWNER_ONLY_DIRECTORY_MODE, parents=True, exist_ok=True)
    temporary_path = status_path.with_suffix(".tmp")
    temporary_path.write_text(json.dumps(status, indent=2) + "\n")
    temporary_path.replace(status_path)


def read_status(status_path: Path = STATUS_PATH) -> dict | None:
    """Read the status of the last background run.

    Args:
        status_path: Location of the JSON status file.

    Returns:
        The values of the last run, or None when the background indexer has not run yet.
    """
    if not status_path.exists():
        return None

    return json.loads(status_path.read_text())
