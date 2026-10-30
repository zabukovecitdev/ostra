# Embedding timing report

Model: `mxbai-embed-large:latest` on the local Ollama server (1024 dimensions).
Tokenizer: `mixedbread-ai/mxbai-embed-large-v1`. Chunks: 510 tokens, 50 tokens overlap.

## 1. App run on `~/Documents`

Command: `uv run ostra`

| Item | Value |
|---|---|
| Wall time, whole run | 3.3 s |
| Files in scope | 8 top-level text files (`.txt`, `.md`, `.csv`) |
| Files with chunks | 7 (`empty.txt` has no text) |
| Chunks | 7 |
| Tokens across all chunks | 428 |
| Largest chunk | 93 tokens |

Most of the 3.3 s went to starting Python and loading the tokenizer. The run printed every chunk and stored the file and chunk digests in SQLite.

**Embedding and storage in the app:** the app does not embed or store vectors yet. It only prints chunks and stores digests. The times below come from a separate script that calls the same chunker and the same Ollama provider.

## 2. Embedding latency on the Documents chunks

| Measure | Value |
|---|---|
| Chunking, all files | 3 ms |
| First embedding call (includes loading the model in Ollama) | 1.28 s |
| Later calls, n = 6 | median 44 ms, max 55 ms, total 0.25 s |

## 3. Embedding latency on full-size chunks

The two longer sample files in `tests/data/` produce five chunks: four at 484 to 510 tokens and one at 133 tokens. Each chunk was embedded three times.

| Measure | Value |
|---|---|
| Calls | 15 |
| Median | 114 ms |
| Mean | 104 ms |
| Min / max | 59 ms / 117 ms |

## 4. Estimate for larger folders

A full-size chunk takes about 0.11 s to embed. Storage time is not measured because storage is not built yet.

| Input size | Chunks (approx.) | Embedding time (approx.) |
|---|---|---|
| 1,000 words | 2 to 3 | 0.2 to 0.3 s |
| 1,000,000 tokens | 1,960 | about 3.7 min |

## 5. Scope and limits

- The app reads only top-level text files. It skips `work-setup/`, `rag-nested/`, and `work-setup.tar.gz`. Those folders hold most of the 3.5 GB in `~/Documents`.
- Each measurement uses the short Documents files or the two longer sample files. They are too few for a precise benchmark.
- Vector storage is not built yet. The current database holds only digests, so the storage time is not in this report.
