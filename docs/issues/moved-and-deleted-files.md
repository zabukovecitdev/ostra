# Issue: moved and deleted files

## Summary

The app keys stored chunks by file path. When a file moves, the app treats it as a new file and embeds every chunk again. When a file is deleted, its chunks stay in the database and can still appear in search results.

## Steps to reproduce

1. Run `ostra` on a folder that contains `notes.txt`.
2. Move `notes.txt` to another folder inside the configured directories.
3. Run `ostra` again.

## Current behavior

- Every chunk of the moved file is embedded again, even though the text has not changed.
- The rows for the old path stay in `files`, `file_chunks` and `chunk_vectors`.
- A search can return chunks from the old path, which no longer exists.

## Expected behavior

- A moved file reuses its stored vectors. The model does not run again.
- Chunks of a deleted file are removed, so a search does not return them.

## Cause

- Chunk rows use the key `(file_path, digest)`. A new path has no matching rows.
- The ingestion step only visits files that exist. Nothing removes rows for files that are gone.

## Proposed fix

1. **Move detection.** When a new path has the same file digest as a stored file that no longer exists, move the stored rows to the new path. Do not embed again. The schema does not change.
2. **Cleanup.** After each run, delete the rows of stored files that are no longer in the processed folders.

## Alternative considered

Store each vector once, keyed by its chunk digest only, with a separate table that maps each file to its chunks. Files with the same text would share one vector. This removes the duplicate work in more cases, but it changes the schema of both backends.

## Open questions

- Should cleanup run only for the configured folders, or also for files that were moved out of them?
- Do two identical files at different paths need separate rows?
