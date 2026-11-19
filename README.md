<div align="center">

# ostra

**Private semantic search for your own text files.**

[![CI](https://github.com/zabukovecitdev/ostra/actions/workflows/ci.yml/badge.svg)](https://github.com/zabukovecitdev/ostra/actions/workflows/ci.yml)
![Python 3.14](https://img.shields.io/badge/python-3.14-blue)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Ruff](https://img.shields.io/badge/code%20style-ruff-261230)](https://github.com/astral-sh/ruff)
[![Checked with mypy](https://img.shields.io/badge/types-mypy-2a6db2)](https://mypy-lang.org)

*The Greek word ὄστρακον (ostrakon) means "shell" or "potsherd". Athenians used ostraka, pottery fragments, to vote in ostracism. The word also gave us "ostracize".*

</div>

---

ostra is a local retrieval pipeline for personal text files. It splits each file into token-sized chunks, embeds every chunk with a local model, stores the vectors, and returns the closest chunks for a query.

Everything runs on your machine. No text leaves it.

## Contents

- [Features](#features)
- [How it works](#how-it-works)
- [Requirements](#requirements)
- [Installation](#installation)
- [Quick start](#quick-start)
- [Configuration](#configuration)
- [Usage](#usage)
- [Background indexing (Linux)](#background-indexing-linux)
- [Architecture](#architecture)
- [Development](#development)
- [Evals](#evals)
- [Performance](#performance)
- [Known limitations](#known-limitations)
- [Roadmap](#roadmap)
- [License](#license)

## Features

- **Token-based chunking** with configurable size and overlap. Token counts come from the embedding model's own tokenizer, so each chunk fits the model's input limit.
- **Resumable ingestion.** Each chunk is stored under its content digest. A rerun skips chunks that are already embedded, so an interrupted run continues where it stopped.
- **Local embeddings** through [Ollama](https://ollama.com), with `mxbai-embed-large` (default) or `bge-m3` for multilingual text.
- **One index for each model.** Vectors of two models never mix, so you can compare models on the same files.
- **Vector search** with cosine distance, from the command line, with plain text or JSON output.
- **SQLite storage with `sqlite-vec`.** One database holds an index for each embedding model, with versioned migrations.
- **Background indexing** as a systemd user service.
- **Owner-only file permissions** on the database, the config file and their folders. Linux and macOS apply these modes. On Windows they have no effect.

## How it works

```
 ┌──────────────┐    ┌──────────┐    ┌───────────┐    ┌──────────────┐
 │ Text files   │───►│ Chunker  │───►│  Ollama   │───►│ Vector store │
 │ .txt .md .csv│    │ (tokens) │    │ embedding │    │    SQLite    │
 └──────────────┘    └──────────┘    └───────────┘    └──────┬───────┘
                                                             │
 ┌──────────────┐    ┌──────────┐    ┌───────────┐           │
 │ Ranked chunks│◄───│  Search  │◄───│  Ollama   │◄── query  │
 │ with sources │    │ (cosine) │    │ embedding │           │
 └──────────────┘    └────┬─────┘    └───────────┘           │
                          └──────────────────────────────────┘
```

1. A source reads your files and puts them in a bounded queue.
2. The pipeline splits each file into chunks with the model's tokenizer.
3. Ollama embeds each new chunk. Chunks with a known digest are skipped.
4. The store saves the chunk text and its vector in the index of that model.
5. A search embeds the query with the same model and returns the closest chunks.

## Requirements

- Python 3.14
- [uv](https://docs.astral.sh/uv/)
- [Ollama](https://ollama.com) running locally or on your network, with the embedding model pulled:

  ```bash
  ollama pull mxbai-embed-large
  ollama pull bge-m3          # optional, for multilingual text
  ```

- `make` (optional, for the shortcuts below)

## Installation

Install the command line tools from GitHub:

```bash
uv tool install 'ostra @ git+https://github.com/zabukovecitdev/ostra'
```


For development, clone the repository instead:

```bash
git clone https://github.com/zabukovecitdev/ostra.git
cd ostra
uv sync
```

## Quick start

```bash
uv sync
# add your folders to "directories" in the config file
make run                      # process the configured folders
make search QUERY="solar system" LIMIT=3
```

## Configuration

The config file is `config.json` in your user config folder, for example `~/.config/ostra/config.json` on Linux. It is created with defaults on the first run. An existing `~/ostra/config.json` from an earlier version is still used when it exists.

| Key | Default | Meaning |
|---|---|---|
| `database_path` | `ostra.db` in your user data folder, for example `~/.local/share/ostra/ostra.db` on Linux | SQLite database. |
| `directories` | none | Folders to process. Add at least one before the first run. The top-level `.txt`, `.md` and `.csv` files are read. `ostra index --recursive` reads the subfolders too. |
| `chunk_size_tokens` | not set | Tokens per chunk. When not set, the default of the embedding model applies (510 for mxbai-embed-large). The value must fit the model input after its special tokens. |
| `chunk_overlap_tokens` | `50` | Tokens shared by consecutive chunks |
| `embedding_provider` | `ollama` | The service that creates embeddings. Ollama is the only provider now. |
| `embedding_model` | `mxbai-embed-large` | The embedding model: `mxbai-embed-large` or `bge-m3`. Use `bge-m3` for non-English text such as Slovene. Each model has its own index in the database. `ostra index --model ID` and `ostra search --model ID` override it for one run. |
| `ollama_base_url` | not set | Address of the Ollama server, for example `http://192.168.1.20:11434`. It wins over `OLLAMA_BASE_URL`. The text of your files is sent to this server. |

The Ollama server address comes from the `OLLAMA_BASE_URL` environment variable (default `http://localhost:11434`) unless `ollama_base_url` is set. The model name comes from the model registry. The old `OLLAMA_MODEL` variable is not used any more, and ostra stops with an error when it is set.

## Usage

| Command | Effect |
|---|---|
| `ostra index` | Embed and store the new or changed chunks of the configured folders |
| `ostra index --directory PATH` | Index one or more folders instead of the configured ones. Repeat the option for more folders. |
| `ostra index --reindex` | Embed every chunk again, even those already stored |
| `ostra search "query" --limit 5` | Print the chunks closest to the query, with a similarity percentage and a short preview |
| `ostra search "query" --max-distance 0.4` | Hide chunks whose cosine distance is above 0.4 (0 is the same direction, 2 is opposite) |
| `ostra search "query" --full` | Print the whole text of each chunk |
| `ostra search "query" --json` | Print the chunks as a JSON list with the full text |
| `ostra index --database PATH` / `ostra search "query" --database PATH` | Use another SQLite database file for one run instead of `database_path` from the config |

Run `ostra --help` or `ostra index --help` for the full list of options.

Makefile shortcuts:

```bash
make run DIRECTORY=~/Documents/notes
make reindex
make search QUERY="transformer attention" LIMIT=3
make help             # list every target
```

## Background indexing (Linux)

`ostra-daemon` indexes the configured folders in a loop. It reads the config file again on each run. It logs the result of each run.

| Command | Effect |
|---|---|
| `ostra-daemon` | Run the loop in the foreground. Default interval is 15 minutes. |
| `ostra-daemon install --interval 10` | Install and start a systemd user service |
| `ostra-daemon uninstall` | Stop and remove the service |

The SQLite database uses write-ahead logging, so `ostra search` works while the daemon writes.

## Architecture

```
src/ostra/
├── cli/                   command line: parser, command handlers, search output, progress bar; ostra-daemon
├── services/
│   ├── ingestion/         pipeline, runner, text chunker, source factory, run_indexing
│   ├── embedding/         EmbeddingModel bundle, provider registry and model factory
│   ├── daemon/            background indexer loop and systemd service
│   └── ...                search service, index session
├── providers/             adapters to outside systems
│   ├── sources/           DocumentSource (abstract) and the file system source
│   ├── embeddings/        EmbeddingProvider (abstract, applies query and document prefixes) and the Ollama provider
│   └── tokenizers/        Tokenizer (abstract) and the tokenizer-file implementation
├── repositories/          storage: SqliteChunkStore with one SqliteModelIndex for each embedding model
│   └── sqlite/            SQLite + sqlite-vec, versioned migrations, one index for each model
├── models/                data types and the model registry (one spec for each model)
└── enums/, errors/, utils/, constants.py
```

A source reads documents in its own thread and puts them in a bounded queue. The pipeline takes them from the queue, and it does not know where they come from. A new source, such as notes or email, is one new `DocumentSource` class. The document ID is the key of the stored chunks. The file system source uses the file path.

The services depend on abstract classes (`DocumentSource`, `EmbeddingProvider`, `ChunkSearchIndex`, `ChunkWriteIndex`), not on SQLite or Ollama. A new embedding provider is one builder in `PROVIDER_BUILDERS` of `services/embedding/embedding_model_factory.py`. A new embedding model is one `EmbeddingModelSpec` in `models/embedding_model_registry.py` plus its `tokenizer.json` file under `providers/tokenizers/data/`.

## Development

```bash
uv sync
make hooks            # install git hooks: checks on commit, tests on push
make check            # lint, format check, type check and tests
```

| Command | Effect |
|---|---|
| `make test` | Run the test suite |
| `make lint` | Check the code with ruff |
| `make format` | Format the code with ruff |
| `make typecheck` | Check the types with mypy |
| `make check` | Run all of the checks above |
| `make evals v1` | Run the retrieval evals of the version folder `evals/v1` (needs Ollama) |

The suite runs 219 tests. The tests do not call Ollama: they use a fake embedding model and a local tokenizer file.

CI runs the same checks on Linux, macOS and Windows for each push to `master` and for each pull request.

## Evals

`make evals <version>` indexes the documents of `evals/<version>/data/` into a new database in that folder, asks every question of `tasks.json` and writes the rank of the expected file to `output.json`. Each version folder keeps its own documents, tasks, settings and results, so two runs compare side by side. See [evals/README.md](evals/README.md).

## Performance

Measured on the development machine with a local Ollama server:

| Measure | Result |
|---|---|
| Embedding one full-size chunk (510 tokens) | median 115 ms |
| Embedding and storing 1 MB of text (408 chunks) | 48 s |
| Reindex of a 440 KB novel (243 chunks) | 28 s |
| Search | 1.4 ms (query embedding adds about 25 ms) |

More detail is in [reports/embedding-timing.md](reports/embedding-timing.md).

## Known limitations

- Subfolders are read only with `ostra index --recursive`. The background indexer reads the top-level files only.
- Only plain text formats are read. PDF and Office files are skipped.
- A moved file is embedded again, and the rows of deleted files are not removed.
- The chunk digests are SHA-256 of the chunk text. Identical text in two files is stored once per file path.

## Roadmap

- Move detection and cleanup for moved and deleted files
- Recursive folder scanning with an option to turn it on
- Answer generation that cites the source file of each chunk
- A content-addressed chunk store that shares vectors between identical text

## License

ostra is released under the [MIT License](LICENSE).

The `mxbai-embed-large-v1` tokenizer file has its own license. See [src/ostra/providers/tokenizers/data/mxbai-embed-large-v1/LICENSE.txt](src/ostra/providers/tokenizers/data/mxbai-embed-large-v1/LICENSE.txt).
