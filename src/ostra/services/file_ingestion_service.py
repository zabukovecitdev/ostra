from pathlib import Path

from ostra.embeddings.base import EmbeddingProvider
from ostra.hashing import hash_file, hash_text
from ostra.repositories.file_chunk_repository import FileChunkRepository
from ostra.repositories.file_repository import FileRepository
from ostra.services.file_service import FileService
from ostra.services.text_chunker import TextChunker


class FileIngestionService:
    """Process the text files in chunks and skip the chunks that a previous run already processed."""

    def __init__(
        self,
        file_service: FileService,
        file_repository: FileRepository,
        file_chunk_repository: FileChunkRepository,
        chunker: TextChunker,
        embedding_provider: EmbeddingProvider,
        reindex: bool,
    ):
        self._file_service = file_service
        self._file_repository = file_repository
        self._file_chunk_repository = file_chunk_repository
        self._chunker = chunker
        self._embedding_provider = embedding_provider
        self._reindex = reindex

    def process_files(self) -> None:
        """Process each text file in the configured directories, one file at a time."""
        for file_path in self._file_service.stream_files():
            self._process_file(file_path)

    def _process_file(self, file_path: Path) -> None:
        """Save the file digest, then embed and store each chunk whose digest is not stored.

        A chunk is skipped when its digest is stored for the file, whatever its position.
        When reindex is set, no stored digest is skipped, so every chunk is embedded again.
        Embed each processed chunk and save it with its embedding, then print it.
        Delete the stored digests that no chunk of the file has.

        Args:
            file_path: Location of the file to process.
        """
        file_key = str(file_path)
        self._file_repository.save_file(file_path=file_key, digest=hash_file(file_path))

        stored_digests = self._file_chunk_repository.get_digests(file_key)
        skipped_digests = set() if self._reindex else set(stored_digests)
        current_digests = set()
        for chunk in self._chunker.stream_chunks(self._file_service.stream_words(file_path)):
            chunk_digest = hash_text(chunk)
            current_digests.add(chunk_digest)
            if chunk_digest in skipped_digests:
                continue

            embedding = self._embedding_provider.get_embedding(chunk)
            self._file_chunk_repository.save_chunk(file_key, chunk_digest, chunk, embedding)
            print(chunk)
            skipped_digests.add(chunk_digest)

        self._file_chunk_repository.delete_chunks(file_key, stored_digests - current_digests)
