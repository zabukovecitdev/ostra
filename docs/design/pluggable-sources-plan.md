# Plan: let any source feed the indexing core

## Goal

Other programs, such as a web scraper pipeline, must be able to send documents to the same core that
chunks, embeds, stores and searches. Today the core is already independent of the origin of a document,
but the entry point `run_indexing` builds the file system source itself, and there is no stable module
for outside code to import.

After this plan:

- `run_indexing` receives its sources from the caller (pull: ostra reads the source).
- `index_documents` accepts documents that the caller already has (push: a scraper hands them over).
- `ostra.api` is the one module that outside code imports.

## Rules for the implementer

- Do the tasks in order. Run the checks of a task before you start the next one.
- Do not change files that the task does not name. In particular, do not touch `src/ostra/gui/app.py`
  (it has uncommitted work), the migrations, `IngestionPipeline`, `SqliteModelIndex` or `Document`.
- Do not commit. Do not add dependencies.
- Follow the existing style: one class per file, Google-style docstrings with `Args:` / `Returns:`,
  short plain sentences, type hints everywhere, test names `test_<unit>__<case>__<result>`.
- Follow `CLAUDE.md`: no `# noqa`, no `# complexipy: ignore`.
- Checks after every task, from the repository root:

  ```
  uv run ruff check .
  uv run ruff format --check .
  uv run mypy
  uv run complexipy
  uv run pytest
  ```

  All five must pass. If `ruff format --check` fails, run `uv run ruff format .` and check again.

## Task 1: `run_indexing` receives the sources

### 1a. `src/ostra/services/indexing_service.py`

Change only `run_indexing` and `_start_progress`.

- Remove the imports `from pathlib import Path` and
  `from ostra.providers.sources.file_system_source import FileSystemSource`.
- Change `from collections.abc import Callable` to `from collections.abc import Callable, Sequence`.
- Replace the parameters `directories` and `recursive` with one required parameter `sources`.
  New signature:

  ```python
  def run_indexing(
      config: AppConfig,
      model_index: SqliteModelIndex,
      embedding_model: EmbeddingModel,
      sources: Sequence[DocumentSource],
      reindex: bool = False,
      on_document_processed: Callable[[str], None] | None = None,
      show_progress: bool = False,
  ) -> None:
  ```

- In the docstring, remove the `directories` and `recursive` lines and add:
  `sources: The sources to read, in order. Each source runs after the one before it has finished.`
- Delete the statement that builds `sources: list[DocumentSource] = [FileSystemSource(...)]`.
  The rest of the body stays the same; it already uses the name `sources`.
- Change the parameter type of `_start_progress` from `list[DocumentSource]` to `Sequence[DocumentSource]`.

### 1b. `src/ostra/cli/main.py`, function `_index_files`

- Add the import `from ostra.providers.sources.file_system_source import FileSystemSource`.
- Keep the first four statements (the `given`, `directories`, and the `NO_DIRECTORIES_MESSAGE` check).
- Replace the `run_indexing(...)` call with:

  ```python
  run_indexing(
      config=config,
      model_index=model_index,
      embedding_model=embedding_model,
      sources=[FileSystemSource(directories=directories or config.directories, recursive=arguments.recursive)],
      reindex=arguments.reindex,
      show_progress=True,
  )
  ```

### 1c. `src/ostra/services/background_indexing_service.py`, function `_index_directories`

- Add the import `from ostra.providers.sources.file_system_source import FileSystemSource`.
- Add one argument to the `run_indexing(...)` call:
  `sources=[FileSystemSource(directories=config.directories)],`
  The daemon did not scan subfolders before, so do not pass `recursive`.

### 1d. `tests/test_main.py`, test `test_main__directory_flag__processes_given_folder_instead_of_config`

The test reads `kwargs["directories"]`, which no longer exists. Record the folders that the file system
source receives instead. Replace the line that patches `run_indexing` and the final assert:

```python
    created_with = []
    monkeypatch.setattr(main, "FileSystemSource", lambda directories, recursive: created_with.append(directories))
    monkeypatch.setattr(main, "run_indexing", lambda **kwargs: None)

    main.main(["index", "--directory", str(given)])

    assert created_with == [[given]]
```

Remove the now unused `indexed_directories = []` line.

### 1e. New file `tests/test_indexing_service.py`

This test proves that a source that is not the file system reaches the store and the search.

```python
from collections.abc import Iterator
from pathlib import Path

from ostra.models.app_config import AppConfig
from ostra.models.document import Document
from ostra.models.embedding_model import EmbeddingModel
from ostra.models.embedding_model_registry import MXBAI_EMBED_LARGE
from ostra.providers.embeddings.embedding_provider import EmbeddingProvider
from ostra.providers.sources.document_source import DocumentSource
from ostra.repositories.sqlite.sqlite_chunk_store import SqliteChunkStore
from ostra.services.chunk_search_service import ChunkSearchService
from ostra.services.indexing_service import run_indexing


class FakeProvider(EmbeddingProvider):
    """Return the same unit vector for every text, so that the cosine distance is defined."""

    def _embed(self, text: str) -> list[float]:
        return [1.0] + [0.0] * 1023


class FakeTokenizer:
    def count_tokens(self, text: str) -> int:
        return len(text.split())


class ListSource(DocumentSource):
    """Offer the documents of a list."""

    def __init__(self, documents: list[Document]):
        self._documents = documents

    def stream_documents(self) -> Iterator[Document]:
        yield from self._documents


def fake_model() -> EmbeddingModel:
    return EmbeddingModel(spec=MXBAI_EMBED_LARGE, provider=FakeProvider(MXBAI_EMBED_LARGE), tokenizer=FakeTokenizer())


def web_page(url: str) -> Document:
    return Document(source="web", document_id=url, version="1", title="Page", text="rivers carry sediment")


def test_run_indexing__two_sources_that_are_not_folders__stores_documents_of_both(
    database_path: Path, store: SqliteChunkStore
):
    """Store the documents of every given source under the source name of each document."""
    model = fake_model()
    index = store.model_index(MXBAI_EMBED_LARGE, create=True)
    processed: list[str] = []

    run_indexing(
        config=AppConfig(database_path=database_path),
        model_index=index,
        embedding_model=model,
        sources=[ListSource([web_page("https://example.com/a")]), ListSource([web_page("https://example.com/b")])],
        on_document_processed=processed.append,
    )

    matches = ChunkSearchService(embedding_provider=model.provider, model_index=index).search("rivers", limit=5)
    assert processed == ["https://example.com/a", "https://example.com/b"]
    assert {(match.source, match.document_id) for match in matches} == {
        ("web", "https://example.com/a"),
        ("web", "https://example.com/b"),
    }
```

`database_path` and `store` are fixtures in `tests/conftest.py`. Do not use the `index` fixture of
`conftest.py`: it has 4 dimensions and the fake model has 1024.

Run the checks.

## Task 2: `index_documents` for callers that push documents

### 2a. `src/ostra/services/indexing_service.py`

- Change the `collections.abc` import to `from collections.abc import Callable, Iterable, Sequence`.
- Add the imports `from ostra.models.document import Document` and
  `from ostra.services.index_session import open_index`.
- Add this function directly after `run_indexing`:

  ```python
  def index_documents(
      config: AppConfig,
      documents: Iterable[Document],
      model_id: str | None = None,
      reindex: bool = False,
  ) -> None:
      """Embed and store the chunks of documents that the caller already has.

      Open the database for this one call and create the index of the model when it does not exist.
      The documents are processed in the calling thread, one at a time, so the iterable may be a generator
      that produces them slowly. Call this from one thread at a time.

      Args:
          config: The application settings.
          documents: The documents to process.
          model_id: The model to use. None means the model of the config.
          reindex: Embed every chunk again, including chunks that are already stored.
      """
      with open_index(config, model_id=model_id, create=True) as session:
          pipeline = build_pipeline(
              config=config,
              model_index=session.model_index,
              embedding_model=session.embedding_model,
              reindex=reindex,
          )
          pipeline.process_documents(documents)
  ```

### 2b. `tests/test_indexing_service.py`

Add the imports `import pytest`, `from ostra.services import index_session`,
`from ostra.services.chunk_search_service import ChunkSearchService, search_chunks` (replace the existing
`ChunkSearchService` import) and `from ostra.services.indexing_service import index_documents, run_indexing`
(replace the existing `run_indexing` import). Add:

```python
def test_index_documents__documents_from_a_generator__can_be_searched(
    database_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Store pushed documents so that a later search in a new database session finds them."""
    monkeypatch.setattr(index_session, "build_embedding_model", lambda config, model_id=None: fake_model())
    config = AppConfig(database_path=database_path)

    index_documents(config, (web_page(url) for url in ["https://example.com/a"]))

    matches = search_chunks(config, query="rivers", limit=5, max_distance=None)
    assert [(match.source, match.document_id) for match in matches] == [("web", "https://example.com/a")]


def test_index_documents__same_document_twice__embeds_it_once(database_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Skip the chunks of a document that an earlier call already stored."""
    model = fake_model()
    calls: list[str] = []
    original = model.provider.embed_document
    monkeypatch.setattr(model.provider, "embed_document", lambda text: calls.append(text) or original(text))
    monkeypatch.setattr(index_session, "build_embedding_model", lambda config, model_id=None: model)
    config = AppConfig(database_path=database_path)

    index_documents(config, [web_page("https://example.com/a")])
    index_documents(config, [web_page("https://example.com/a")])

    assert len(calls) == 1
```

If mypy rejects the `calls.append(text) or original(text)` lambda, replace it with a small named function
that appends and then returns `original(text)`.

Run the checks.

## Task 3: `ostra.api`, the module for outside code

### 3a. New file `src/ostra/api.py`

```python
"""The names that other programs import to send documents to ostra and to search them.

Example:
    from ostra.api import Document, index_documents, load_config, search_chunks

    config = load_config()
    index_documents(config, [Document(source="web", document_id=url, version=etag, title=title, text=text)])
    matches = search_chunks(config, query="solar system", limit=5, max_distance=None)
"""

from ostra.models.app_config import AppConfig
from ostra.models.chunk_match import ChunkMatch
from ostra.models.document import Document
from ostra.providers.sources.document_source import DocumentSource
from ostra.repositories.config_repository import load_config
from ostra.services.chunk_search_service import search_chunks
from ostra.services.indexing_service import index_documents, run_indexing

__all__ = [
    "AppConfig",
    "ChunkMatch",
    "Document",
    "DocumentSource",
    "index_documents",
    "load_config",
    "run_indexing",
    "search_chunks",
]
```

Before you write the file, check that `load_config` exists in `src/ostra/repositories/config_repository.py`
and takes no required argument. It is imported that way in `background_indexing_service.py`.

### 3b. New file `tests/test_api.py`

```python
from ostra import api


def test_api__all_names__are_importable():
    """Offer every name that the module lists."""
    assert all(hasattr(api, name) for name in api.__all__)
```

### 3c. `README.md`, section `## Architecture`

- In the directory tree, add this line directly above the `├── cli/` line, aligned like the others:
  `├── api.py                 the names that other programs import: Document, index_documents, search_chunks`
- Replace the paragraph that starts with `A source reads documents in its own thread` with:

  ```
  A source reads documents in its own thread and puts them in a bounded queue. The pipeline takes them from the queue, and it does not know where they come from. The document ID is the key of the stored chunks. The file system source uses the file path, and a web source would use the URL.

  There are two ways to add documents from another origin. Both are in `ostra.api`.

  - Pull: write a `DocumentSource` class and give it to `run_indexing(..., sources=[...])`. ostra reads the source.
  - Push: call `index_documents(config, documents)` with documents that your program already has, for example at the end of a scraper pipeline. Call it from one thread at a time.

  The `source` field of a document is a free name such as `web`. Search results return it. Use a different `database_path` when you want to keep two collections apart, because search does not filter by source yet.
  ```

Run the checks.

## Done when

- The five checks pass.
- `grep -rn "FileSystemSource" src/ostra/services/indexing_service.py` prints nothing.
- `uv run python -c "from ostra.api import index_documents, search_chunks, Document, DocumentSource"` prints nothing
  and exits with 0.
- `git status --short` shows changes only in: `indexing_service.py`, `background_indexing_service.py`,
  `cli/main.py`, `api.py`, `README.md`, `tests/test_main.py`, `tests/test_indexing_service.py`,
  `tests/test_api.py` (plus the files that were already changed before the work started).

## Not in this plan

These need a design decision first. Do not start them.

- **Skip an unchanged document by `Document.version`.** The `documents` table is shared by all model
  indexes and its row is committed before the chunks are embedded. A version stored there would wrongly
  skip a document for a second model, after an interrupted run, and after a change of the chunk size.
  The version must be stored per model index and written only after the last chunk.
- **Filter search by source.** The vector search takes the nearest `k` rows before any join, so a filter
  on `documents.source` would return fewer than `limit` results. It needs a sqlite-vec metadata or
  partition column, which means rebuilding the vector tables.
- **Remove documents that the source no longer offers.** See `docs/issues/moved-and-deleted-files.md`.
- **Sources in the config file.** `AppConfig.directories` stays the only configured source. A scraper
  passes its own source or documents in code.
