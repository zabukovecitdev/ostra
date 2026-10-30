-- Store the embedding of each chunk as float32 bytes.
-- Rows written before this migration have no embedding, so they are deleted.
-- The next run processes those chunks again and stores their embeddings.
ALTER TABLE file_chunks ADD COLUMN embedding BLOB;
DELETE FROM file_chunks;
