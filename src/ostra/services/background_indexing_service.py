import logging
import time
from datetime import UTC, datetime
from pathlib import Path

from ostra.constants import NO_DIRECTORIES_MESSAGE
from ostra.models.app_config import AppConfig
from ostra.repositories.config_repository import load_config
from ostra.repositories.open_chunk_store import open_chunk_store
from ostra.repositories.status_repository import STATUS_PATH, write_status
from ostra.services.embedding_model_factory import build_embedding_model
from ostra.services.indexing_service import run_indexing

logger = logging.getLogger(__name__)


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
    processed_files: list[str] = []
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

    write_status(
        {
            "started_at": started_at.isoformat(),
            "finished_at": datetime.now(UTC).isoformat(),
            "files_processed": len(processed_files),
            "error": error,
            "message": message,
        },
        status_path,
    )


def _index_directories(config: AppConfig, processed_files: list[str]) -> None:
    """Embed and store the chunks of the configured folders.

    Args:
        config: The application settings.
        processed_files: List that receives the ID of each processed document.
    """
    embedding_model = build_embedding_model(config)
    with open_chunk_store(config.database_backend, config.database_path) as store:
        run_indexing(
            config=config,
            model_index=store.model_index(embedding_model.spec, create=True),
            embedding_model=embedding_model,
            on_document_processed=processed_files.append,
        )
