"""Run the retrieval evals of one version folder.

A version folder holds the setup of one eval run and its results, so that runs can be compared:

    evals/<version>/config.json   the settings of the run (model, chunk size, search limit)
    evals/<version>/tasks.json    the questions and the file that answers each one
    evals/<version>/data/         the documents to index
    evals/<version>/ostra.db      the database of the run, written by the run
    evals/<version>/output.json   the results of the run, written by the run

Run it with `make evals <version>`, for example `make evals v1`.
"""

import argparse
import json
import statistics
import sys
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

from ostra.models.app_config import AppConfig
from ostra.models.chunk_match import ChunkMatch
from ostra.repositories.sqlite.sqlite_chunk_store import SqliteChunkStore
from ostra.services.chunk_search_service import ChunkSearchService
from ostra.services.embedding.embedding_model import EmbeddingModel
from ostra.services.embedding.embedding_model_factory import build_embedding_model
from ostra.services.ingestion.indexing_service import run_indexing
from ostra.services.ingestion.source_factory import build_sources

EVALS_DIRECTORY = Path(__file__).parent
CONFIG_FILE = "config.json"
TASKS_FILE = "tasks.json"
DATA_DIRECTORY = "data"
DATABASE_FILE = "ostra.db"
OUTPUT_FILE = "output.json"
# The number of chunks each search returns. The expected document must be among them to get a rank.
DEFAULT_SEARCH_LIMIT = 50
# The ranks that the summary counts hits for, such as hit@1 and hit@3.
HIT_RANKS = (1, 3, 5)

ModelBuilder = Callable[[AppConfig], EmbeddingModel]


@dataclass(frozen=True)
class EvalTask:
    """Hold one question and the name of the document file that answers it."""

    id: int | str
    question: str
    expected: str


@dataclass(frozen=True)
class EvalSettings:
    """Hold the settings of one version folder."""

    app_config: AppConfig
    search_limit: int


@dataclass(frozen=True)
class TaskResult:
    """Hold the outcome of one task."""

    id: int | str
    question: str
    expected: str
    seconds: float
    first_result: str | None
    first_distance: float | None
    expected_rank: int | None
    top_documents: list[str]


def resolve_version_directory(name: str) -> Path:
    """Find the version folder from the name on the command line.

    Args:
        name: A folder name inside the evals folder, such as v1, or a path to a version folder.

    Returns:
        The version folder.

    Raises:
        FileNotFoundError: When neither the path nor the folder inside the evals folder exists.
    """
    for candidate in (Path(name), EVALS_DIRECTORY / name):
        if (candidate / TASKS_FILE).is_file():
            return candidate.resolve()

    raise FileNotFoundError(f"Version folder with a {TASKS_FILE} not found. Name={name}")


def load_settings(version_directory: Path) -> EvalSettings:
    """Read the config file of a version folder.

    The database and the data folder always come from the version folder, so a run never touches the database of
    the config file in the home folder, even when OSTRA_ environment variables are set.

    Args:
        version_directory: The version folder.

    Returns:
        The settings of the run.
    """
    config_path = version_directory / CONFIG_FILE
    values = json.loads(config_path.read_text()) if config_path.is_file() else {}
    search_limit = int(values.pop("search_limit", DEFAULT_SEARCH_LIMIT))
    app_config = AppConfig(**values).model_copy(
        update={
            "database_path": version_directory / DATABASE_FILE,
            "directories": [version_directory / DATA_DIRECTORY],
        }
    )
    return EvalSettings(app_config=app_config, search_limit=search_limit)


def load_tasks(version_directory: Path) -> list[EvalTask]:
    """Read the tasks file of a version folder.

    Args:
        version_directory: The version folder.

    Returns:
        The tasks in file order.
    """
    entries = json.loads((version_directory / TASKS_FILE).read_text())
    return [EvalTask(id=entry["id"], question=entry["question"], expected=entry["expected"]) for entry in entries]


def setup(version_directory: Path) -> Path:
    """Create an empty database in the version folder. Delete the database of an earlier run first.

    Args:
        version_directory: The version folder.

    Returns:
        The location of the new database.
    """
    database_path = version_directory / DATABASE_FILE
    for suffix in ("", "-wal", "-shm", "-journal"):
        Path(f"{database_path}{suffix}").unlink(missing_ok=True)

    SqliteChunkStore.open(database_path).close()
    return database_path


def ranked_documents(matches: list[ChunkMatch]) -> list[str]:
    """List the documents of the matched chunks, each once, in the order of their best chunk.

    Args:
        matches: The chunks of a search, nearest first.

    Returns:
        The file names of the documents, nearest first.
    """
    return list(dict.fromkeys(match.title for match in matches))


def run_task(service: ChunkSearchService, task: EvalTask, search_limit: int) -> TaskResult:
    """Search for the question of a task and find the rank of the expected document.

    Args:
        service: The search over the indexed chunks.
        task: The task to run.
        search_limit: The number of chunks the search returns.

    Returns:
        The time of the search, the first document and the rank of the expected document.
    """
    started = time.perf_counter()
    matches = service.search(query=task.question, limit=search_limit)
    seconds = time.perf_counter() - started
    documents = ranked_documents(matches)
    return TaskResult(
        id=task.id,
        question=task.question,
        expected=task.expected,
        seconds=round(seconds, 4),
        first_result=documents[0] if documents else None,
        first_distance=round(matches[0].distance, 4) if matches else None,
        expected_rank=documents.index(task.expected) + 1 if task.expected in documents else None,
        top_documents=documents[:5],
    )


def summarize(results: list[TaskResult]) -> dict:
    """Compute the totals of a run.

    Args:
        results: The results of every task.

    Returns:
        The hit rate at each rank of HIT_RANKS, the mean reciprocal rank and the search times.
    """
    if not results:
        return {"tasks": 0}

    ranks = [result.expected_rank for result in results]
    seconds = [result.seconds for result in results]
    hits = {f"hit_at_{cutoff}": _hit_rate(ranks, cutoff) for cutoff in HIT_RANKS}
    return {
        "tasks": len(results),
        **hits,
        "mrr": round(statistics.fmean(_reciprocal_rank(rank) for rank in ranks), 4),
        "not_found": ranks.count(None),
        "mean_seconds": round(statistics.fmean(seconds), 4),
        "median_seconds": round(statistics.median(seconds), 4),
        "max_seconds": round(max(seconds), 4),
    }


def _reciprocal_rank(rank: int | None) -> float:
    """Return 1 / rank, or 0 when the expected document was not found."""
    if rank is None:
        return 0.0

    return 1 / rank


def _hit_rate(ranks: list[int | None], cutoff: int) -> float:
    """Return the share of the tasks whose expected document ranks at the cutoff or better, rounded to 4 decimals."""
    return round(sum(1 for rank in ranks if rank is not None and rank <= cutoff) / len(ranks), 4)


def run_evals(version_directory: Path, build_model: ModelBuilder = build_embedding_model) -> dict:
    """Index the documents of a version folder into a new database, run every task and write output.json.

    Args:
        version_directory: The version folder.
        build_model: Builds the embedding model from the settings. Tests pass a fake model.

    Returns:
        The content of output.json.
    """
    settings = load_settings(version_directory)
    tasks = load_tasks(version_directory)
    config = settings.app_config
    embedding_model = build_model(config)
    started_at = datetime.now(UTC).isoformat(timespec="seconds")
    database_path = setup(version_directory)

    with SqliteChunkStore.open(database_path) as store:
        model_index = store.model_index(embedding_model.spec, create=True)
        index_started = time.perf_counter()
        documents = run_indexing(
            chunk_size_tokens=config.chunk_size_tokens,
            chunk_overlap_tokens=config.chunk_overlap_tokens,
            model_index=model_index,
            embedding_model=embedding_model,
            sources=build_sources(config),
        )
        index_seconds = time.perf_counter() - index_started
        service = ChunkSearchService(embedding_provider=embedding_model.provider, model_index=model_index)
        results = [run_task(service, task, settings.search_limit) for task in tasks]

    output = {
        "version": version_directory.name,
        "started_at": started_at,
        "config": {
            "embedding_model": embedding_model.spec.model_id,
            "chunk_size_tokens": embedding_model.chunk_size_tokens(config.chunk_size_tokens),
            "chunk_overlap_tokens": config.chunk_overlap_tokens,
            "search_limit": settings.search_limit,
        },
        "indexing": {"documents": documents, "seconds": round(index_seconds, 3)},
        "summary": summarize(results),
        "tasks": [asdict(result) for result in results],
    }
    output_path = version_directory / OUTPUT_FILE
    output_path.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return output


def print_report(output: dict) -> None:
    """Print one line for each task and the summary of a run.

    Args:
        output: The content of output.json.
    """
    for task in output["tasks"]:
        rank = task["expected_rank"] or "-"
        mark = "ok  " if task["expected_rank"] == 1 else "MISS"
        print(f"{mark} rank={rank!s:>2} {task['seconds']:.3f}s  first={task['first_result']}  {task['question']}")

    print(json.dumps(output["summary"], indent=2))


def main(argv: list[str] | None = None) -> None:
    """Run the evals of the version folder named on the command line.

    Args:
        argv: Command line arguments without the program name. Read sys.argv when None.
    """
    parser = argparse.ArgumentParser(prog="evals", description="Run the retrieval evals of one version folder.")
    parser.add_argument("version", help="Version folder inside the evals folder, such as v1, or a path to one.")
    arguments = parser.parse_args(argv)
    try:
        version_directory = resolve_version_directory(arguments.version)
    except FileNotFoundError as error:
        sys.exit(str(error))

    output = run_evals(version_directory)
    print_report(output)
    print(f"Wrote {version_directory / OUTPUT_FILE}")


if __name__ == "__main__":
    main()
