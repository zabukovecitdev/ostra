-- Give each embedding model its own index, and name the stored items by document instead of by file.
-- The chunks of the existing index move to chunks_1 with the same row ids, so the vectors in chunk_vectors stay valid.
-- The existing index belongs to mxbai-embed-large. Its fingerprint is set on the first write.
BEGIN;
CREATE TABLE documents (
    document_id TEXT PRIMARY KEY,
    source TEXT NOT NULL,
    title TEXT NOT NULL,
    processed_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
INSERT INTO documents (document_id, source, title, processed_at)
SELECT file_path, 'filesystem', file_path, processed_at FROM files;

CREATE TABLE model_indexes (
    model_index_id INTEGER PRIMARY KEY,
    model_id TEXT NOT NULL UNIQUE,
    dimensions INTEGER NOT NULL,
    fingerprint TEXT,
    chunks_table TEXT NOT NULL,
    vectors_table TEXT NOT NULL
);

CREATE TABLE chunks_1 (
    chunk_id INTEGER PRIMARY KEY,
    document_id TEXT NOT NULL,
    digest TEXT NOT NULL,
    chunk_text TEXT,
    processed_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (document_id, digest)
);
INSERT INTO chunks_1 (chunk_id, document_id, digest, chunk_text, processed_at)
SELECT rowid, file_path, digest, chunk_text, processed_at FROM file_chunks;

INSERT INTO model_indexes (model_index_id, model_id, dimensions, fingerprint, chunks_table, vectors_table)
VALUES (1, 'mxbai-embed-large', 1024, NULL, 'chunks_1', 'chunk_vectors');

DROP TABLE file_chunks;
DROP TABLE files;
COMMIT;
