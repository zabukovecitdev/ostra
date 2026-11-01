import argparse
import json
import logging
import time
from datetime import UTC, datetime
from pathlib import Path

from platformdirs import user_data_path

from ostra.config import APP_NAME, NO_DIRECTORIES_MESSAGE, AppConfig, load_config
from ostra.connectors.database_facade import DatabaseFacade
from ostra.indexing import build_ingestion_service
from ostra.models.factory import build_embedding_model
from ostra.service_manager import install_service, uninstall_service
from ostra.services.database_service import DatabaseService

DEFAULT_INTERVAL_MINUTES = 15
STATUS_PATH = user_data_path(APP_NAME) / "status.json"

logger = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> None:
    """Run the background indexer, or install or remove its user service.

    Args:
        argv: Command line arguments without the program name. Read sys.argv when None.
    """
    logging.basicConfig(level=logging.INFO)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    arguments = _build_parser().parse_args(argv)
    if arguments.command == "install":
        print(install_service(interval_minutes=arguments.interval))
    elif arguments.command == "uninstall":
        print(uninstall_service())
    else:
        run_forever(interval_minutes=arguments.interval)


def run_forever(interval_minutes: int) -> None:
    """Index the configured folders, wait, and repeat until the process stops.

    A failed run does not stop the loop. The next run starts after the interval.

    Args:
        interval_minutes: Minutes to wait between the end of one run and the start of the next.
    """
    while True:
        index_once()
        time.sleep(interval_minutes * 60)


def index_once(status_path: Path = STATUS_PATH) -> None:
    """Index the configured folders one time and write the result to the status file.

    The config file is read again on each run, so setting changes apply to the next run.
    The error is logged and saved in the status file. It is not raised.

    Args:
        status_path: Location of the JSON status file.
    """
    started_at = datetime.now(UTC)
    processed_files: list[Path] = []
    error = None
    message = None
    try:
        config = load_config()
        if config.directories:
            _index_directories(config, processed_files)
        else:
            message = NO_DIRECTORIES_MESSAGE
    except Exception as exception:
        logger.exception("Indexing run failed.")
        error = str(exception)

    _write_status(
        status_path,
        {
            "started_at": started_at.isoformat(),
            "finished_at": datetime.now(UTC).isoformat(),
            "files_processed": len(processed_files),
            "error": error,
            "message": message,
        },
    )


def _index_directories(config: AppConfig, processed_files: list[Path]) -> None:
    """Embed and store the chunks of the configured folders.

    Args:
        config: The application settings.
        processed_files: List that receives the path of each processed file.
    """
    connector = DatabaseFacade(backend=config.database_backend, database_path=config.database_path)
    try:
        DatabaseService(connector=connector).initialize()
        build_ingestion_service(
            config=config,
            connector=connector,
            embedding_model=build_embedding_model(config),
            on_file_processed=processed_files.append,
        ).process_files()
    finally:
        connector.close()


def _write_status(status_path: Path, status: dict) -> None:
    """Write the status of the last run to a JSON file, replacing the previous file.

    Args:
        status_path: Location of the JSON status file.
        status: The values to write.
    """
    status_path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    temporary_path = status_path.with_suffix(".tmp")
    temporary_path.write_text(json.dumps(status, indent=2) + "\n")
    temporary_path.replace(status_path)


def _build_parser() -> argparse.ArgumentParser:
    """Build the command line parser with the run, install and uninstall commands.

    Returns:
        The parser. The run command is the default.
    """
    parser = argparse.ArgumentParser(prog="ostra-daemon", description="Index text files in the background.")
    parser.set_defaults(command="run", interval=DEFAULT_INTERVAL_MINUTES)
    commands = parser.add_subparsers(dest="command", metavar="COMMAND")
    for name, text in (
        ("run", "Index in a loop until stopped. This is the default."),
        ("install", "Install and start a systemd user service that runs this loop."),
    ):
        command = commands.add_parser(name, help=text, description=text)
        command.add_argument(
            "--interval",
            type=int,
            default=DEFAULT_INTERVAL_MINUTES,
            metavar="MINUTES",
            help=f"Minutes between runs. Default is {DEFAULT_INTERVAL_MINUTES}.",
        )
    commands.add_parser("uninstall", help="Stop and remove the systemd user service.")
    return parser


if __name__ == "__main__":
    main()
