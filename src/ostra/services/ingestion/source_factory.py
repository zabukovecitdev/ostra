from pathlib import Path

from ostra.models.app_config import AppConfig
from ostra.providers.sources.document_source import DocumentSource
from ostra.providers.sources.file_system_source import FileSystemSource


def build_sources(
    config: AppConfig, directories: list[Path] | None = None, recursive: bool = False
) -> list[DocumentSource]:
    """Build the sources of documents to index.

    Add a new kind of source here. The indexing code takes the list and does not know the kinds.

    Args:
        config: The application settings.
        directories: Folders to index. Use the configured folders when None or empty.
        recursive: Index the files in the subfolders of the folders too.

    Returns:
        The sources, which are the file system source of the folders.
    """
    return [FileSystemSource(directories=directories or config.directories, recursive=recursive)]
