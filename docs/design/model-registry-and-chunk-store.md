# Model registry and chunk store

Status: steps 1 and 2 are built (store, model indexes, migration 006, registry, vector identity, prefixes). Step 3 is partly built: bge-m3 and its tokenizer are in the registry. nomic-embed-text is not built.

## Problem

Vectors carry no model identity. The `chunk_vectors` table has a fixed size of 1024 (migration 005). A chunk is skipped when its digest is stored. After a model change, ostra would search old vectors with query vectors of the new model. It would give wrong results and no error.

The storage code has three more problems:
- `VectorDatabaseConnector` has 11 methods. `DatabaseFacade` and three repositories forward each call one to one.
- The storage key is `file_path`, but the pipeline works with documents.
- Search and index wiring is repeated in `main.py`, `daemon.py` and `gui/search.py`.

Constraints:
- An existing index must stay valid. Document IDs stay equal to the old file paths.
- The daemon and the GUI are separate processes. SQLite uses WAL for this. Chroma is single-process.
- mxbai-embed-large and bge-m3 both have 1024 dimensions, so a dimension check cannot tell them apart.

## Usage (caller's view)

```python
# Index (ostra index, daemon)
model = build_embedding_model(config, model_id=arguments.model)  # None means config.embedding_model
with open_chunk_store(config.database_backend, config.database_path) as store:
    index = store.model_index(model.spec, create=True)
    run_indexing(config, index, model, reindex=arguments.reindex)

# Search (ostra search, GUI)
with open_chunk_store(config.database_backend, config.database_path) as store:
    index = store.model_index(model.spec, create=False)  # ModelNotIndexedError: run ostra index
    matches = ChunkSearchService(model.provider, index).search(query, limit, max_distance)

# Eval: two models on the same documents, one connection
with open_chunk_store(backend, path) as store:
    for model in models:
        service = ChunkSearchService(model.provider, store.model_index(model.spec, create=False))
        report = evaluate(service, cases, limit)
```

## Shape

Each embedding model gets its own index inside one database: its own chunk table, vector table and digests. An index is created with the dimension of its model.

```python
@dataclass(frozen=True)
class EmbeddingModelSpec:
    model_id: str; ollama_model: str; dimensions: int; max_tokens: int; special_tokens: int
    tokenizer_file: str; default_chunk_size_tokens: int
    query_prefix: str = ""; document_prefix: str = ""
    @property
    def fingerprint(self) -> str: ...          # sha256(ollama_model, dimensions, document_prefix)

EMBEDDING_MODELS: Mapping[str, EmbeddingModelSpec]   # one list: config, CLI and GUI read it

@dataclass(frozen=True)
class Vector: model_id: str; values: tuple[float, ...]    # a vector knows its model

class EmbeddingProvider(ABC):
    def embed_query(self, text) -> Vector       # applies query_prefix, stamps model_id, checks length
    def embed_document(self, text) -> Vector    # applies document_prefix

@dataclass(frozen=True)
class EmbeddingModel: spec; provider; tokenizer       # concrete bundle, replaces the ABC

class ChunkStore(ABC):                                # one database
    def model_index(self, spec, create: bool) -> ModelIndex
    def close(self) -> None

class ModelIndex(ABC):                                # the chunks of one model
    def stored_digests(self, document_id) -> set[str]
    def save_document(self, document, new_chunks, current_digests) -> None   # one transaction
    def search(self, query: Vector, limit) -> list[ChunkMatch]

ChunkMatch(document_id, source, title, digest, text, distance)
```

Decisions that carry weight:
- **Identity, not size, guards the index.** Every `Vector` carries its `model_id`. The index stores the fingerprint. A mismatch raises `ModelIndexMismatchError`, and nothing repairs it silently.
- **Only the code that creates an index runs DDL.** Search uses `create=False`, so the GUI never changes the schema.
- **Each chunk is saved when it is embedded.** `save_document` takes a lazy iterable and commits after each chunk, then deletes the stale chunks at the end. This keeps a run resumable after an interruption. A reader can see a half-saved document, as before. (Changed during step 1: the first sketch used one transaction per document, which lost the progress of a long document.)
- **The store knows only id, dimensions and fingerprint.** Prefixes, tokenizer and Ollama stay in `models/`.
- **The query prefix is not in the fingerprint.** Query vectors are made fresh, so a prefix change needs no rebuild.
- **Migration 006** moves `files` and `file_chunks` to `documents` and `chunks_1` with an explicit `chunk_id`. The existing vector table stays where it is and becomes the mxbai index. No vector is copied or re-embedded. The migration makes a backup copy of the database file first.

Interface depth: the public surface is one factory, 2 store methods and 3 index methods. It replaces the 11 connector methods, the 11 facade forwards, 3 repositories and `DatabaseService`. The backends hide the permissions, WAL, migrations, DDL per dimension, rowid joins and the stale-chunk deletion.

## Synthesis decision

Base: the per-model index in one database (candidate A).

Taken from candidate B:
- `Vector` with `model_id`, `embed_query` and `embed_document`, and the frozen `EmbeddingModel` bundle.
- The identity check on every read and write. A fingerprint closes the gap that equal dimensions leave.
- `chunk_size_tokens` is optional, and the model spec gives the default.
- Removal of `get_file_digest`, `get_chunk_embedding` and the `OLLAMA_MODEL` override.
- The store re-reads the index registry inside `BEGIN IMMEDIATE` before DDL, so two processes cannot create the same index.

Rejected from B: one model per database with a rebuild in place. Search returns partial results for hours during a rebuild. A model switch destroys the old index. Eval needs a separate database file for each model.

Rejected from A: a one-time metadata rewrite for Chroma, and a lock file around migrations. Both add risk for little gain.

## Tradeoffs accepted

- We accept disk use of one index for each model that you try, in exchange for a switch back without a re-embed. There is no drop-model command yet.
- We accept that an inactive index goes stale, in exchange for no rebuild downtime.
- We accept a rewrite of the SQLite schema (migration 006), in exchange for an explicit `chunk_id` and `document_id` names. A backup copy protects the data.
- We accept a pass-through `ChunkSearchService`, because search, the GUI and eval share it.

## Alternatives considered

- **Guarded single index, rebuild in place (B).** Smaller migration, one vector table. It loses on switching cost and on model comparison.
- **One vector table for all models with a `model_id` column.** A vec0 table has one fixed dimension, so it fails for 768 and 1024 together.
- **Detect the model by dimension.** It cannot separate mxbai and bge-m3.

## Open questions and risks

- Should Chroma keep the old data? The proposal is that Chroma starts fresh with new collections per model, and its users reindex. Chroma is optional and single-process.
- Should the store keep `Document.version`? Nothing reads it today. The proposal drops the column and keeps the field on `Document`.
- A moving `:latest` tag passes the fingerprint check. Should the registry pin tags or store the `ollama show` digest?
- A `max_distance` tuned for mxbai does not carry over to other models. Should each spec have its own default?
- The daemon runs old code during an upgrade. Should `service_manager` stop it around the upgrade?
- The eval command lives on `feature/retrieval-eval`. It changes `main.py` as well. Which branch merges first?

## Next implementation step

Build `ChunkStore`, `ModelIndex` and migration 006 with mxbai only, and move the callers to them. Add the registry and `Vector` identity in the second step. Add nomic-embed-text and bge-m3 with their tokenizers in the third step. Each step ends with a green `make check`.
