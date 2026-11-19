"""Run the retrieval evals of one run folder against its dataset.

A dataset holds the documents and the questions. It stays the same once runs use it, so that their results compare:

    evals/datasets/<dataset>/tasks.json   the questions and the file that answers each one
    evals/datasets/<dataset>/data/        the documents to index

A run holds the settings of one experiment and its results:

    evals/runs/<run>/config.json          the dataset name and the settings (model, chunk size, search limit)
    evals/runs/<run>/ostra.db             the database of the run, written by the run
    evals/runs/<run>/output.json          the results of the run, written by the run

Run it with `make evals <run>`, for example `make evals v1-baseline`.
"""

import argparse
import hashlib
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
DATASETS_DIRECTORY = EVALS_DIRECTORY / "datasets"
RUNS_DIRECTORY = EVALS_DIRECTORY / "runs"
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
    """Hold the settings of one run."""

    app_config: AppConfig
    search_limit: int
    dataset_name: str
    dataset_directory: Path


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


def resolve_run_directory(name: str) -> Path:
    """Find the run folder from the name on the command line.

    Args:
        name: A folder name inside evals/runs, such as v1-baseline, or a path to a run folder.

    Returns:
        The run folder.

    Raises:
        FileNotFoundError: When neither the path nor the folder inside evals/runs has a config.json.
    """
    for candidate in (Path(name), RUNS_DIRECTORY / name):
        if (candidate / CONFIG_FILE).is_file():
            return candidate.resolve()

    raise FileNotFoundError(f"Run folder with a {CONFIG_FILE} not found. Name={name}")


def load_settings(run_directory: Path, datasets_directory: Path = DATASETS_DIRECTORY) -> EvalSettings:
    """Read the config file of a run folder and find its dataset.

    The database always lives in the run folder and the documents always come from the dataset, so a run never
    touches the database of the config file in the home folder, even when OSTRA_ environment variables are set.

    Args:
        run_directory: The run folder.
        datasets_directory: The folder that holds the datasets.

    Returns:
        The settings of the run.

    Raises:
        ValueError: When the config file names no dataset, or the dataset has no tasks.json.
    """
    values = json.loads((run_directory / CONFIG_FILE).read_text())
    dataset_name = values.pop("dataset", None)
    if not dataset_name:
        raise ValueError(f"{CONFIG_FILE} must name a dataset of {datasets_directory}. Run={run_directory}")

    dataset_directory = datasets_directory / dataset_name
    if not (dataset_directory / TASKS_FILE).is_file():
        raise ValueError(f"Dataset with a {TASKS_FILE} not found. Dataset={dataset_directory}")

    search_limit = int(values.pop("search_limit", DEFAULT_SEARCH_LIMIT))
    app_config = AppConfig(**values).model_copy(
        update={
            "database_path": run_directory / DATABASE_FILE,
            "directories": [dataset_directory / DATA_DIRECTORY],
        }
    )
    return EvalSettings(
        app_config=app_config,
        search_limit=search_limit,
        dataset_name=dataset_name,
        dataset_directory=dataset_directory,
    )


def load_tasks(dataset_directory: Path) -> list[EvalTask]:
    """Read the tasks file of a dataset.

    Args:
        dataset_directory: The dataset folder.

    Returns:
        The tasks in file order.
    """
    entries = json.loads((dataset_directory / TASKS_FILE).read_text())
    return [EvalTask(id=entry["id"], question=entry["question"], expected=entry["expected"]) for entry in entries]


def dataset_fingerprint(dataset_directory: Path) -> str:
    """Hash the tasks and the documents of a dataset, so that two runs show whether they used the same data.

    Args:
        dataset_directory: The dataset folder.

    Returns:
        The SHA-256 hex digest over tasks.json and the name and bytes of each document, in sorted order.
    """
    digest = hashlib.sha256()
    files = [dataset_directory / TASKS_FILE, *sorted((dataset_directory / DATA_DIRECTORY).iterdir())]
    for path in files:
        if path.is_file():
            digest.update(path.relative_to(dataset_directory).as_posix().encode("utf-8") + b"\0")
            digest.update(path.read_bytes() + b"\0")

    return digest.hexdigest()


def setup(run_directory: Path) -> Path:
    """Create an empty database in the run folder. Delete the database of an earlier run first.

    Args:
        run_directory: The run folder.

    Returns:
        The location of the new database.
    """
    database_path = run_directory / DATABASE_FILE
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


def run_evals(
    run_directory: Path,
    build_model: ModelBuilder = build_embedding_model,
    datasets_directory: Path = DATASETS_DIRECTORY,
) -> dict:
    """Index the dataset of a run into a new database in the run folder, run every task and write output.json.

    Args:
        run_directory: The run folder.
        build_model: Builds the embedding model from the settings. Tests pass a fake model.
        datasets_directory: The folder that holds the datasets.

    Returns:
        The content of output.json.
    """
    settings = load_settings(run_directory, datasets_directory)
    tasks = load_tasks(settings.dataset_directory)
    config = settings.app_config
    embedding_model = build_model(config)
    started_at = datetime.now(UTC).isoformat(timespec="seconds")
    database_path = setup(run_directory)

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
        "run": run_directory.name,
        "started_at": started_at,
        "dataset": {
            "name": settings.dataset_name,
            "fingerprint": dataset_fingerprint(settings.dataset_directory),
            "documents": documents,
            "tasks": len(tasks),
        },
        "config": {
            "embedding_model": embedding_model.spec.model_id,
            "chunk_size_tokens": embedding_model.chunk_size_tokens(config.chunk_size_tokens),
            "chunk_overlap_tokens": config.chunk_overlap_tokens,
            "search_limit": settings.search_limit,
        },
        "indexing": {"seconds": round(index_seconds, 3)},
        "summary": summarize(results),
        "tasks": [asdict(result) for result in results],
    }
    output_path = run_directory / OUTPUT_FILE
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
    """Run the evals of the run folder named on the command line.

    Args:
        argv: Command line arguments without the program name. Read sys.argv when None.
    """
    parser = argparse.ArgumentParser(prog="evals", description="Run the retrieval evals of one run folder.")
    parser.add_argument("run", help="Run folder inside evals/runs, such as v1-baseline, or a path to one.")
    arguments = parser.parse_args(argv)
    try:
        run_directory = resolve_run_directory(arguments.run)
        output = run_evals(run_directory)
    except (FileNotFoundError, ValueError) as error:
        sys.exit(str(error))

    print_report(output)
    print(f"Wrote {run_directory / OUTPUT_FILE}")


if __name__ == "__main__":
    main()
