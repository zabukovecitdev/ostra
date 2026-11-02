import sqlite3
from collections.abc import Iterable, Set

import sqlite_vec

from ostra.embeddings.vector import Vector
from ostra.store.chunk_match import ChunkMatch
from ostra.store.document_record import DocumentRecord
from ostra.store.embedded_chunk import EmbeddedChunk
from ostra.store.model_index import ModelIndex


class SqliteModelIndex(ModelIndex):
    """Keep the chunks of one embedding model in a chunk table and a vec0 vector table.

    The vector table uses the chunk ID of the chunk table as its row ID.
    """

    def __init__(
        self, connection: sqlite3.Connection, model_id: str, dimensions: int, chunks_table: str, vectors_table: str
    ):
        self._connection = connection
        self._model_id = model_id
        self._dimensions = dimensions
        self._chunks_table = chunks_table
        self._vectors_table = vectors_table

    def stored_digests(self, document_id: str) -> set[str]:
        """Return the digests of the chunks that are stored for a document.

        Args:
            document_id: The ID of the document.

        Returns:
            The SHA-256 hex digests of the stored chunks.
        """
        rows = self._connection.execute(
            f"SELECT digest FROM {self._chunks_table} WHERE document_id = ?", (document_id,)
        ).fetchall()
        return {row[0] for row in rows}

    def save_document(
        self, document: DocumentRecord, new_chunks: Iterable[EmbeddedChunk], current_digests: Set[str]
    ) -> None:
        """Save a document and its new chunks, then delete the chunks that the document no longer has.

        Commit after the document row and after each chunk. The text and the vector of a chunk are written in one
        transaction.

        Args:
            document: The ID, source and title of the document.
            new_chunks: The chunks to save, with their embeddings.
            current_digests: The digests of all chunks that the document has now.
        """
        self._connection.execute(
            "INSERT INTO documents (document_id, source, title) VALUES (?, ?, ?) "
            "ON CONFLICT (document_id) DO UPDATE SET "
            "source = excluded.source, title = excluded.title, processed_at = CURRENT_TIMESTAMP",
            (document.document_id, document.source, document.title),
        )
        self._connection.commit()

        for chunk in new_chunks:
            self._save_chunk(document.document_id, chunk)

        self._delete_stale_chunks(document.document_id, current_digests)

    def search(self, query: Vector, limit: int) -> list[ChunkMatch]:
        """Return the stored chunks that are closest to a query vector.

        Args:
            query: The embedding vector of the query.
            limit: The number of chunks to return.

        Returns:
            The closest chunks, nearest first. Each match has a cosine distance, where 0 means the same direction.
        """
        query_values = sqlite_vec.serialize_float32(self._checked(query))
        rows = self._connection.execute(
            "SELECT c.document_id, COALESCE(d.source, ''), COALESCE(d.title, c.document_id), c.digest, "
            f"c.chunk_text, v.distance FROM {self._vectors_table} v "
            f"JOIN {self._chunks_table} c ON c.chunk_id = v.rowid "
            "LEFT JOIN documents d ON d.document_id = c.document_id "
            "WHERE v.embedding MATCH ? AND k = ? ORDER BY v.distance",
            (query_values, limit),
        ).fetchall()
        return [
            ChunkMatch(document_id=document_id, source=source, title=title, digest=digest, text=text, distance=distance)
            for document_id, source, title, digest, text, distance in rows
        ]

    def _checked(self, vector: Vector) -> list[float]:
        """Check that a vector belongs to the model of this index and has its size.

        Args:
            vector: The vector to check.

        Returns:
            The values of the vector.

        Raises:
            ValueError: When the vector comes from another model or has the wrong size.
        """
        if vector.model_id != self._model_id or len(vector.values) != self._dimensions:
            raise ValueError(
                f"Vector does not fit the index. Index={self._model_id}/{self._dimensions} "
                f"Vector={vector.model_id}/{len(vector.values)}"
            )

        return list(vector.values)

    def _save_chunk(self, document_id: str, chunk: EmbeddedChunk) -> None:
        """Insert a chunk and its vector, or replace both when the digest is already stored.

        Args:
            document_id: The ID of the document.
            chunk: The chunk with its embedding.
        """
        embedding = sqlite_vec.serialize_float32(self._checked(chunk.embedding))
        connection = self._connection
        connection.execute(
            f"INSERT INTO {self._chunks_table} (document_id, digest, chunk_text) VALUES (?, ?, ?) "
            "ON CONFLICT (document_id, digest) DO UPDATE SET "
            "processed_at = CURRENT_TIMESTAMP, chunk_text = excluded.chunk_text",
            (document_id, chunk.digest, chunk.text),
        )
        chunk_id = connection.execute(
            f"SELECT chunk_id FROM {self._chunks_table} WHERE document_id = ? AND digest = ?",
            (document_id, chunk.digest),
        ).fetchone()[0]
        connection.execute(f"DELETE FROM {self._vectors_table} WHERE rowid = ?", (chunk_id,))
        connection.execute(
            f"INSERT INTO {self._vectors_table} (rowid, embedding) VALUES (?, ?)",
            (chunk_id, embedding),
        )
        connection.commit()

    def _delete_stale_chunks(self, document_id: str, current_digests: Set[str]) -> None:
        """Delete the stored chunks and vectors of a document whose digest is not current.

        Args:
            document_id: The ID of the document.
            current_digests: The digests that the document has now.
        """
        stale = [
            (chunk_id, digest)
            for chunk_id, digest in self._connection.execute(
                f"SELECT chunk_id, digest FROM {self._chunks_table} WHERE document_id = ?", (document_id,)
            ).fetchall()
            if digest not in current_digests
        ]
        if not stale:
            return

        self._connection.executemany(f"DELETE FROM {self._vectors_table} WHERE rowid = ?", [(row[0],) for row in stale])
        self._connection.executemany(
            f"DELETE FROM {self._chunks_table} WHERE chunk_id = ?", [(row[0],) for row in stale]
        )
        self._connection.commit()
