DROP TABLE IF EXISTS file_chunks;

CREATE TABLE file_chunks (
    file_path TEXT NOT NULL,
    digest TEXT NOT NULL,
    processed_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (file_path, digest)
);
