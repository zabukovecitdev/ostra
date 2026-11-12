CREATE TABLE IF NOT EXISTS file_chunks (
    file_path TEXT NOT NULL,
    line_number INTEGER NOT NULL,
    digest TEXT NOT NULL,
    processed_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (file_path, line_number)
);
