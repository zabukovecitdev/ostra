<div align="center">

# ostra

*The Greek word ὄστρακον (ostrakon) means "shell" or "potsherd". Athenians used ostraka, pottery fragments, to vote in ostracism. The word also gave us "ostracize".*

</div>

A local retrieval pipeline for personal text files. It splits each file into token-sized chunks, embeds every chunk with a local model, stores the vectors, and returns the closest chunks for a query. Everything runs on your machine. No text leaves it.

## Features

- **Token-based chunking** with configurable size and overlap. Token counts come from the embedding model's own tokenizer, so each chunk fits the model's input limit.
- **Resumable ingestion.** Each chunk is stored under its content digest. A rerun skips chunks that are already embedded, so an interrupted run continues where it stopped.
- **Local embeddings** through [Ollama](https://ollama.com), using `mxbai-embed-large` (1024 dimensions).
- **Vector search** with cosine distance, from the command line.
- **Two storage backends behind one interface:** SQLite with `sqlite-vec` (default) and Chroma. Switch with a flag or a config key.
- **Owner-only file permissions** on the database, the config file and their folders. Linux and macOS apply these modes. On Windows they have no effect.

## Architecture

```
main.py                    command line entry point and wiring
├── sources/               DocumentSource (abstract) and the file system source
├── pipeline/              ingestion pipeline (chunk, embed, store) and the queue runner
├── services/              chunking, search, database setup
├── store/                 ChunkStore and ModelIndex (abstract), one index for each embedding model
│   ├── sqlite/            SQLite + sqlite-vec, versioned migrations
│   └── chroma/            Chroma persistent client
├── embeddings/            EmbeddingProvider (abstract, applies query and document prefixes), Vector, Ollama provider
├── tokenization/          Tokenizer (abstract) and the tokenizer-file implementation
└── models/                model registry (one spec for each model), EmbeddingModel bundle, factory
```

A source reads documents in its own thread and puts them in a bounded queue. The pipeline takes them from the queue, and it does not know where they come from. A new source, such as notes or email, is one new `DocumentSource` class. The document ID is the key of the stored chunks. The file system source uses the file path.

The layers depend on abstract classes, so a new embedding model or storage backend means one new class and a change in one place. A new model is one `EmbeddingModelSpec` in `models/registry.py` plus its `tokenizer.json` file under `tokenization/data/`.

## Requirements

- Python 3.14
- [uv](https://docs.astral.sh/uv/)
- [Ollama](https://ollama.com) running locally, with the embedding model pulled:

  ```bash
  ollama pull mxbai-embed-large
  ollama pull bge-m3          # optional, for multilingual text
  ```

- `make` (optional, for the shortcuts below)

## Quick start

```bash
uv sync
# add your folders to "directories" in the config file, or use the config page of ostra-app
make run                      # process the configured folders
make search QUERY="solar system" LIMIT=3
```

## Configuration

The config file is `config.json` in your user config folder, for example `~/.config/ostra/config.json` on Linux. It is created with defaults on the first run. An existing `~/ostra/config.json` from an earlier version is still used when it exists.

| Key | Default | Meaning |
|---|---|---|
| `database_path` | `ostra.db` in your user data folder, for example `~/.local/share/ostra/ostra.db` on Linux | SQLite database. The Chroma store is a `chroma` folder beside it. |
| `database_backend` | `sqlite` | `sqlite` or `chroma`. Chroma needs the extra: `pip install 'ostra[chroma]'` |
| `directories` | none | Folders to process. Add at least one before the first run. Only top-level `.txt`, `.md` and `.csv` files are read. |
| `chunk_size_tokens` | not set | Tokens per chunk. When not set, the default of the embedding model applies (510 for mxbai-embed-large). The value must fit the model input after its special tokens. |
| `chunk_overlap_tokens` | `50` | Tokens shared by consecutive chunks |
| `embedding_provider` | `ollama` | The service that creates embeddings. Ollama is the only provider now. |
| `embedding_model` | `mxbai-embed-large` | The embedding model: `mxbai-embed-large` or `bge-m3`. Use `bge-m3` for non-English text such as Slovene. Each model has its own index in the database, so vectors of two models never mix. `ostra index --model ID` and `ostra search --model ID` override it for one run. |
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
| `ostra --database chroma index` | Use the Chroma backend for this run. The option also works after the command name. |

Run `ostra --help` or `ostra index --help` for the full list of options.

Makefile shortcuts:

```bash
make run DIRECTORY=~/Documents/notes
make reindex
make search QUERY="transformer attention" LIMIT=3
make check            # lint, format check, type check and tests
make hooks            # install git hooks: checks on commit, tests on push
make help             # list every target
```

## Background indexing (Linux)

`ostra-daemon` indexes the configured folders in a loop. It reads the config file again on each run. It writes the result of the last run to `status.json` in the user data folder.

| Command | Effect |
|---|---|
| `ostra-daemon` | Run the loop in the foreground. Default interval is 15 minutes. |
| `ostra-daemon install --interval 10` | Install and start a systemd user service |
| `ostra-daemon uninstall` | Stop and remove the service |

The SQLite database uses write-ahead logging, so `ostra search` works while the daemon writes.

## Desktop interface

A local web interface has a search page and a config page. The config page edits the folders, database path, backend and chunk sizes. It also shows the result of the last background run. The **Index now** button starts one indexing run at once with the saved settings, and it shows the result when the run ends.

```bash
uv tool install 'ostra[gui]'
ostra-app                     # opens http://127.0.0.1:8765
```

On the first start, with no folders set, a welcome page shows one button that adds folders. It also shows whether Ollama and the model are ready, with a link to install them when they are not. The config page has a folder picker. The interface listens on the local machine only, so the picker lists the folders of the computer that runs it.

## Evaluating retrieval

`ostra eval` measures how well the search finds the right chunks. It reads a JSON test set. Each case has a query and the chunks that the search must return:

```json
[
  {"query": "Which bird walks along the stream bottom?",
   "relevant": [{"file": "river_ecosystems_field_guide.txt", "text": "Dippers walk along the bed"}]}
]
```

A result matches when its file path ends with `file`. When `text` is set, the chunk must also contain it. Keep `text` short, so that it does not cross a chunk boundary.

| Command | Effect |
|---|---|
| `ostra eval cases.json --limit 5` | Print the rank of each case, the hit rate, recall and MRR at 5, and the mean distance of relevant and other results |
| `ostra eval cases.json --query-prefix "TEXT "` | Put a query instruction in front of each query, to compare instructions of the model |
| `ostra eval cases.json --json` | Print the report as JSON |
| `make eval-sample` | Index `tests/data` into a scratch folder and run `evals/sample_cases.json`. It needs a running Ollama. |

Use the mean distances to choose `--max-distance`: set it between the relevant and the other distance. Run the same test set before and after you change the chunk size, the model or the search code.

## Testing

```bash
make test
```

The suite runs 214 tests. The storage contract tests and the ingestion tests run once for each backend. The tests do not call Ollama: they use a fake embedding model and a local tokenizer file.

## Performance

Measured on the development machine with a local Ollama server:

| Measure | Result |
|---|---|
| Embedding one full-size chunk (510 tokens) | median 115 ms |
| Embedding and storing 1 MB of text (408 chunks) | 48 s |
| Reindex of a 440 KB novel (243 chunks), SQLite / Chroma | 28 s / 31 s |
| Search, SQLite / Chroma | 1.4 ms / 2.9 ms (query embedding adds about 25 ms) |

More detail is in [reports/embedding-timing.md](reports/embedding-timing.md).

## Known limitations

- Only top-level files are read. Subfolders are skipped.
- Only plain text formats are read. PDF and Office files are skipped.
- A moved file is embedded again, and the rows of deleted files are not removed. See [docs/issues/moved-and-deleted-files.md](docs/issues/moved-and-deleted-files.md).
- The vector size is fixed by the migration (1024). A model with another size needs a new migration.
- The chunk digests are SHA-256 of the chunk text. Identical text in two files is stored once per file path.

## Project layout

```
src/ostra/   application code
tests/              pytest suite and sample text files (tests/data)
reports/            performance measurements
docs/issues/        tracked issues with analysis
```

## Roadmap (planned)

- Move detection and cleanup for moved and deleted files
- Recursive folder scanning with an option to turn it on
- Answer generation that cites the source file of each chunk
- A content-addressed chunk store that shares vectors between identical text
