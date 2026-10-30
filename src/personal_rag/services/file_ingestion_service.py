from pathlib import Path

from personal_rag.hashing import hash_file, hash_line
from personal_rag.repositories.file_chunk_repository import FileChunkRepository
from personal_rag.repositories.file_repository import FileRepository
from personal_rag.services.file_service import FileService


class FileIngestionService:
    """Process the text files and skip the lines that a previous run already processed."""

    def __init__(
        self,
        file_service: FileService,
        file_repository: FileRepository,
        file_chunk_repository: FileChunkRepository,
        reindex: bool,
    ):
        self._file_service = file_service
        self._file_repository = file_repository
        self._file_chunk_repository = file_chunk_repository
        self._reindex = reindex

    def process_files(self) -> None:
        """Process each text file in the configured directories, one file at a time."""
        for file_path in self._file_service.stream_files():
            self._process_file(file_path)

    def _process_file(self, file_path: Path) -> None:
        """Save the file digest, then process each line whose digest is not stored.

        A line is skipped when its digest is stored for the file, whatever its line number.
        When reindex is set, no stored digest is skipped, so every line is processed again.
        Save the digest of each processed line after the line is processed. Delete the stored digests that
        no line of the file has. Process the line by printing it.

        Args:
            file_path: Location of the file to process.
        """
        file_key = str(file_path)
        self._file_repository.save_file(file_path=file_key, digest=hash_file(file_path))

        stored_digests = self._file_chunk_repository.get_digests(file_key)
        skipped_digests = set() if self._reindex else set(stored_digests)
        current_digests = set()
        for line in self._file_service.stream_lines(file_path):
            line_digest = hash_line(line)
            current_digests.add(line_digest)
            if line_digest in skipped_digests:
                continue

            print(line.rstrip("\n"))
            self._file_chunk_repository.save_chunk(file_key, line_digest)
            skipped_digests.add(line_digest)

        self._file_chunk_repository.delete_chunks(file_key, stored_digests - current_digests)
