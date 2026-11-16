import logging
import time

from ostra.constants import NO_DIRECTORIES_MESSAGE
from ostra.models.app_config import AppConfig
from ostra.repositories.config_repository import load_config
from ostra.services.index_session import open_index
from ostra.services.ingestion.indexing_service import run_indexing
from ostra.services.ingestion.source_factory import build_sources

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


def index_once() -> None:
    """Index the configured folders one time and log the result.

    The config file is read again on each run, so setting changes apply to the next run.
    The error is logged. It is not raised.
    """
    processed_files: list[str] = []
    try:
        config = load_config()
        if not config.directories:
            logger.warning(NO_DIRECTORIES_MESSAGE)
            return

        _index_directories(config, processed_files)
    except Exception:
        logger.exception("Indexing run failed.")
        return

    logger.info("Indexing run finished. FilesProcessed=%d", len(processed_files))


def _index_directories(config: AppConfig, processed_files: list[str]) -> None:
    """Embed and store the chunks of the configured folders.

    Args:
        config: The application settings.
        processed_files: List that receives the ID of each processed document.
    """
    with open_index(config, create=True) as session:
        run_indexing(
            config=config,
            model_index=session.model_index,
            embedding_model=session.embedding_model,
            sources=build_sources(config),
            on_document_processed=processed_files.append,
        )
