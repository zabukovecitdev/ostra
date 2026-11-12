-- Move the chunk vectors into a vec0 virtual table, so that the database can run nearest-neighbour search.
-- The vec0 table uses the same rowid as file_chunks. Each vector is 1024 values, the size of mxbai-embed-large.
CREATE VIRTUAL TABLE chunk_vectors USING vec0(embedding float[1024] distance_metric=cosine);
INSERT INTO chunk_vectors (rowid, embedding) SELECT rowid, embedding FROM file_chunks WHERE embedding IS NOT NULL;
ALTER TABLE file_chunks DROP COLUMN embedding;
-- Chunks stored before this migration have no text. Run with --reindex once to store their text.
ALTER TABLE file_chunks ADD COLUMN chunk_text TEXT;
