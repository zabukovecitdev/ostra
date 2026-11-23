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
- [Upgrading from a version with ostra-daemon](#upgrading-from-a-version-with-ostra-daemon)
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
- **Local embeddings with no setup.** The default model, `potion-multilingual-128m`, runs inside ostra on the CPU and supports more than 100 languages. It needs no GPU and no server. Switch to [Ollama](https://ollama.com) with `mxbai-embed-large` or `bge-m3` for larger models.
- **One index for each model.** Vectors of two models never mix, so you can compare models on the same files.
- **Vector search** with cosine distance, from the command line, with plain text or JSON output.
- **SQLite storage with `sqlite-vec`.** One database holds an index for each embedding model, with versioned migrations.
- **Owner-only file permissions** on the database, the config file and their folders. Linux and macOS apply these modes. On Windows they have no effect.

## How it works

```
 ┌──────────────┐    ┌──────────┐    ┌───────────┐    ┌──────────────┐
 │ Text files   │───►│ Chunker  │───►│ Embedding │───►│ Vector store │
 │ .txt .md .csv│    │ (tokens) │    │   model   │    │    SQLite    │
 └──────────────┘    └──────────┘    └───────────┘    └──────┬───────┘
                                                             │
 ┌──────────────┐    ┌──────────┐    ┌───────────┐           │
 │ Ranked chunks│◄───│  Search  │◄───│ Embedding │◄── query  │
 │ with sources │    │ (cosine) │    │   model   │           │
 └──────────────┘    └────┬─────┘    └───────────┘           │
                          └──────────────────────────────────┘
```

1. A source reads your files and puts them in a bounded queue.
2. The pipeline splits each file into chunks with the model's tokenizer.
3. The embedding model embeds each new chunk, in the ostra process or on an Ollama server. Chunks with a known digest are skipped.
4. The store saves the chunk text and its vector in the index of that model.
5. A search embeds the query with the same model and returns the closest chunks.

## Requirements

- Python 3.14
- [uv](https://docs.astral.sh/uv/)
- `make` (optional, for the shortcuts below)

Nothing else is needed for the default model. On the first index or search, ostra downloads the model from Hugging Face (about 500 MB) into your user cache folder, for example `~/.cache/ostra/models` on Linux. Later runs work offline.

The default model runs on ONNX Runtime, which has wheels for Linux (x86_64 and arm64), macOS on Apple silicon and Windows. Intel Macs can use the Ollama models.

### Optional: Ollama models

For the larger Ollama models, run [Ollama](https://ollama.com) locally or on your network, pull the model and set `embedding_model` in the config file:

```bash
ollama pull mxbai-embed-large
ollama pull bge-m3          # multilingual
```

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
ostra init                    # choose the model, the folders and the indexing settings step by step
ostra index                   # process the configured folders
ostra search "solar system" --limit 3
```

`ostra init` asks one question at a time, with the current value as the default:

1. The embedding model.
2. For an Ollama model, the server address. ostra checks that the server answers and has the model, and prints the `ollama pull` command when it does not.
3. The folders to index, and whether to include their subfolders.
4. The chunk settings: the model defaults, or your own chunk size and overlap.
5. The database location.

It shows a summary and saves the settings only when you confirm. Press Ctrl-C at any time to stop without saving. Run `ostra init` again later to change the settings. `ostra init --defaults --directory ~/notes` saves the settings without questions, for scripts.

ostra indexes only when you run `ostra index`. A change of the model, the folders, the subfolder setting, the chunk settings or the database needs a run of `ostra index` to reach the index. The summary of `ostra init` lists these changes, and after saving it offers to index right away. A plain `ostra index` is enough: it embeds the new chunks and removes the outdated chunks of each file it reads. `--reindex` is only needed to embed unchanged chunks again. After a model change, searches find nothing until `ostra index` has filled the index of the new model.

## Configuration

The config file is `config.json` in your user config folder, for example `~/.config/ostra/config.json` on Linux. It is created with defaults on the first run. An existing `~/ostra/config.json` from an earlier version is still used when it exists.

| Key | Default | Meaning |
|---|---|---|
| `database_path` | `ostra.db` in your user data folder, for example `~/.local/share/ostra/ostra.db` on Linux | SQLite database. |
| `directories` | none | Folders to process. Add at least one before the first run. The top-level `.txt`, `.md` and `.csv` files are read, and the subfolders too when `recursive` is set. |
| `recursive` | `false` | Read the files in the subfolders of each folder too. Hidden subfolders, such as `.git`, are skipped. `ostra index --recursive` and `--no-recursive` override it for one run. |
| `chunk_size_tokens` | not set | Tokens per chunk. When not set, the default of the embedding model applies (510 for every supported model). The value must fit the model input after its special tokens. |
| `chunk_overlap_tokens` | `50` | Tokens shared by consecutive chunks |
| `embedding_model` | `potion-multilingual-128m` | The embedding model. The model decides where it runs; see the table below. Each model has its own index in the database. `ostra index --model ID` and `ostra search --model ID` override it for one run. |
| `ollama_base_url` | not set | Address of the Ollama server, for example `http://192.168.1.20:11434`. It wins over `OLLAMA_BASE_URL`. The text of your files is sent to this server. Only the Ollama models use it. |

| Model | Runs in | Languages | Dimensions | Notes |
|---|---|---|---|---|
| `potion-multilingual-128m` (default) | the ostra process, CPU ([FastEmbed](https://github.com/qdrant/fastembed)) | 100+ | 256 | Static embeddings: hundreds of chunks per second on a laptop CPU. Downloaded once on first use. |
| `mxbai-embed-large` | Ollama | English | 1024 | Needs `ollama pull mxbai-embed-large`. |
| `bge-m3` | Ollama | 100+ | 1024 | Needs `ollama pull bge-m3`. |

Config files from earlier versions name `mxbai-embed-large` and keep using it and its existing index. The old `embedding_provider` key is ignored, because the model decides the provider. To move to the default model, set `embedding_model` to `potion-multilingual-128m` and run `ostra index`.

For the Ollama models, the server address comes from the `OLLAMA_BASE_URL` environment variable (default `http://localhost:11434`) unless `ollama_base_url` is set. The model name comes from the model registry. The old `OLLAMA_MODEL` variable is not used any more, and ostra stops with an error when it is set.

## Usage

| Command | Effect |
|---|---|
| `ostra init` | Set up the model, the Ollama connection, the folders and the indexing settings step by step |
| `ostra init --defaults` | Save the current settings, or the defaults, without questions. Add `--directory PATH` to add a folder |
| `ostra index` | Embed and store the new or changed chunks of the configured folders |
| `ostra index --directory PATH` | Index one or more folders instead of the configured ones. Repeat the option for more folders. |
| `ostra index --reindex` | Embed every chunk again, even those already stored |
| `ostra search "query"` | In a terminal, list the matching files and open the one you pick. See [Opening results](#opening-results) |
| `ostra search "query" --limit 5` | Show at most 5 files in the list, or print 5 chunks when the output is not a terminal |
| `ostra search "query" --no-interactive` | Print the chunks closest to the query, with a similarity percentage and a short preview, instead of the list |
| `ostra search "query" --max-distance 0.4` | Hide chunks whose cosine distance is above 0.4 (0 is the same direction, 2 is opposite) |
| `ostra search "query" --full` | Print the whole text of each chunk |
| `ostra search "query" --json` | Print the chunks as a JSON list with the full text |
| `ostra index --database PATH` / `ostra search "query" --database PATH` | Use another SQLite database file for one run instead of `database_path` from the config |

Run `ostra --help` or `ostra index --help` for the full list of options.

### Opening results

When you run `ostra search` in a terminal, it lists each matching file once, nearest first, with its similarity and its number of matching chunks. The nearest chunk of the highlighted file shows below the list.

- Press Enter to open the file in its default application. The list stays open, so you can open more files. Pick Quit or press Ctrl-C to stop.
- When no application opens the file, ostra shows it in its folder in the file manager. When the file was deleted after indexing, ostra opens its folder.
- On a server without a desktop, where `xdg-open` is missing, ostra prints the path instead.

Files open with `click.launch`: `open` on macOS, the Windows shell on Windows and `xdg-open` on Linux. The output of `--json` and `--full`, and any output to a pipe or file, stays the plain chunk list. `--interactive` and `--no-interactive` choose the mode yourself.

Makefile shortcuts:

```bash
make run DIRECTORY=~/Documents/notes
make reindex
make search QUERY="transformer attention" LIMIT=3
make help             # list every target
```

## Upgrading from a version with ostra-daemon

Earlier versions had an `ostra-daemon` command that indexed in the background as a systemd user service. It has been removed: run `ostra index` after you change files or settings. If you installed the service, remove it, because systemd keeps restarting a command that no longer exists. Before you upgrade, run `ostra-daemon uninstall`. After an upgrade, run:

```bash
systemctl --user disable --now ostra-daemon.service
rm ~/.config/systemd/user/ostra-daemon.service
systemctl --user daemon-reload
```

## Architecture

```
src/ostra/
├── cli/                   command line: parser, command handlers, search output and file picker, progress bar
│   └── init/              ostra init: the setup steps, and the Prompter with its questionary implementation
├── services/
│   ├── ingestion/         pipeline, runner, text chunker, source factory, run_indexing
│   ├── embedding/         EmbeddingModel bundle, provider registry and model factory
│   ├── setup/             Ollama server check for ostra init
│   └── ...                search service, index session
├── providers/             adapters to outside systems
│   ├── sources/           DocumentSource (abstract) and the file system source
│   ├── embeddings/        EmbeddingProvider (abstract, applies query and document prefixes), the FastEmbed and Ollama providers
│   └── tokenizers/        Tokenizer (abstract) and the tokenizer-file implementation
├── repositories/          storage: SqliteChunkStore with one SqliteModelIndex for each embedding model
│   └── sqlite/            SQLite + sqlite-vec, versioned migrations, one index for each model
├── models/                data types and the model registry (one spec for each model)
└── enums/, errors/, utils/, constants.py
```

A source reads documents in its own thread and puts them in a bounded queue. The pipeline takes them from the queue, and it does not know where they come from. A new source, such as notes or email, is one new `DocumentSource` class. The document ID is the key of the stored chunks. The file system source uses the file path.

The services depend on abstract classes (`DocumentSource`, `EmbeddingProvider`, `ChunkSearchIndex`, `ChunkWriteIndex`), not on SQLite or Ollama. A new embedding provider is one name in `EmbeddingProviderName` and one builder in `PROVIDER_BUILDERS` of `services/embedding/embedding_model_factory.py`. A new embedding model is one `EmbeddingModelSpec` in `models/embedding_model_registry.py`, which names its provider, plus its `tokenizer.json` file under `providers/tokenizers/data/`.

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
| `make evals v2-potion-multilingual` | Run the retrieval evals of a run folder, here `evals/runs/v2-potion-multilingual`. `v1-baseline` needs Ollama |

The suite runs 306 tests. The tests call neither Ollama nor Hugging Face: they use fake embedding models and local tokenizer files. The `ostra init` tests answer the questions with a scripted prompter.

CI runs the same checks on Linux, macOS and Windows for each push to `master` and for each pull request.

## Evals

`make evals <run>` indexes the dataset that `evals/runs/<run>/config.json` names into a new database in the run folder, asks every question of the dataset and writes the rank of the expected file to `output.json`. Datasets (`evals/datasets/`) hold the documents and questions and stay fixed; runs (`evals/runs/`) hold the settings and results, so runs on the same dataset compare side by side. See [evals/README.md](evals/README.md).

## Performance

Measured on the development machine with `mxbai-embed-large` on a local Ollama server:

| Measure | Result |
|---|---|
| Embedding one full-size chunk (510 tokens) | median 115 ms |
| Embedding and storing 1 MB of text (408 chunks) | 48 s |
| Reindex of a 440 KB novel (243 chunks) | 28 s |
| Search | 1.4 ms (query embedding adds about 25 ms) |

More detail is in [reports/embedding-timing.md](reports/embedding-timing.md).

## Known limitations

- Only plain text formats are read. PDF and Office files are skipped.
- A moved file is embedded again, and the rows of deleted files, and of folders removed from the settings, are not removed.
- The chunk digests are SHA-256 of the chunk text. Identical text in two files is stored once per file path.

## Roadmap

- Move detection and cleanup for moved and deleted files
- Answer generation that cites the source file of each chunk
- A content-addressed chunk store that shares vectors between identical text

## License

ostra is released under the [MIT License](LICENSE).

The `potion-multilingual-128M` tokenizer file is MIT licensed by Minish Lab; see its [SOURCE.md](src/ostra/providers/tokenizers/data/potion-multilingual-128M/SOURCE.md). The `mxbai-embed-large-v1` tokenizer file has its own license. See [src/ostra/providers/tokenizers/data/mxbai-embed-large-v1/LICENSE.txt](src/ostra/providers/tokenizers/data/mxbai-embed-large-v1/LICENSE.txt).
