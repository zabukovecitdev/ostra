import json
from pathlib import Path

import pytest
from conftest import WordTokenizer, make_spec

from evals.runner import (
    EvalTask,
    TaskResult,
    dataset_fingerprint,
    load_settings,
    resolve_run_directory,
    run_evals,
    setup,
    summarize,
)
from ostra.models.app_config import AppConfig
from ostra.providers.embeddings.embedding_provider import EmbeddingProvider
from ostra.services.embedding.embedding_model import EmbeddingModel

KEYWORDS = ("bees", "rivers", "stars", "ships")
KEYWORD_SPEC = make_spec("keyword-model", len(KEYWORDS))


class KeywordEmbeddingProvider(EmbeddingProvider):
    """Embed a text as the counts of a few keywords, so that a question finds the file that names its keyword."""

    def _embed(self, text: str) -> list[float]:
        """Return the count of each keyword, plus a small constant so that no vector is zero."""
        words = text.lower().split()
        return [words.count(keyword) + 0.01 for keyword in KEYWORDS]


def build_keyword_model(config: AppConfig) -> EmbeddingModel:
    """Stand in for the Ollama model."""
    return EmbeddingModel(spec=KEYWORD_SPEC, provider=KeywordEmbeddingProvider(KEYWORD_SPEC), tokenizer=WordTokenizer())


@pytest.fixture
def datasets_directory(tmp_path: Path) -> Path:
    """Provide a datasets folder with one dataset of three documents and three tasks."""
    datasets = tmp_path / "datasets"
    data = datasets / "tiny" / "data"
    data.mkdir(parents=True)
    (data / "bees.txt").write_text("bees make honey and bees dance")
    (data / "rivers.txt").write_text("rivers flow to the sea and rivers carve valleys")
    (data / "stars.txt").write_text("stars shine at night")
    tasks = [
        {"id": 1, "question": "what do bees make", "expected": "bees.txt"},
        {"id": 2, "question": "where do rivers flow", "expected": "rivers.txt"},
        {"id": 3, "question": "what do bees dance about", "expected": "stars.txt"},
    ]
    (datasets / "tiny" / "tasks.json").write_text(json.dumps(tasks))
    return datasets


@pytest.fixture
def run_directory(tmp_path: Path) -> Path:
    """Provide a run folder whose config names the tiny dataset."""
    run = tmp_path / "runs" / "r-test"
    run.mkdir(parents=True)
    config = {"dataset": "tiny", "chunk_size_tokens": 20, "chunk_overlap_tokens": 0, "search_limit": 1}
    (run / "config.json").write_text(json.dumps(config))
    return run


def test_run_evals_writes_the_rank_of_each_expected_document(run_directory: Path, datasets_directory: Path):
    output = run_evals(run_directory, build_model=build_keyword_model, datasets_directory=datasets_directory)

    written = json.loads((run_directory / "output.json").read_text())
    assert written == output
    assert [task["first_result"] for task in output["tasks"]] == ["bees.txt", "rivers.txt", "bees.txt"]
    assert [task["expected_rank"] for task in output["tasks"]] == [1, 1, None]
    assert all(task["seconds"] >= 0 for task in output["tasks"])
    assert output["run"] == "r-test"
    assert output["dataset"]["name"] == "tiny"
    assert output["dataset"]["documents"] == 3
    assert output["dataset"]["fingerprint"] == dataset_fingerprint(datasets_directory / "tiny")
    assert output["config"]["search_limit"] == 1
    assert output["summary"]["hit_at_1"] == pytest.approx(2 / 3, abs=1e-4)
    assert output["summary"]["not_found"] == 1
    assert (run_directory / "ostra.db").is_file()
    assert not (datasets_directory / "tiny" / "ostra.db").exists()


def test_run_evals_starts_from_an_empty_database_each_time(run_directory: Path, datasets_directory: Path):
    run_evals(run_directory, build_model=build_keyword_model, datasets_directory=datasets_directory)
    (datasets_directory / "tiny" / "data" / "stars.txt").unlink()

    output = run_evals(run_directory, build_model=build_keyword_model, datasets_directory=datasets_directory)

    assert "stars.txt" not in {name for task in output["tasks"] for name in task["top_documents"]}


def test_dataset_fingerprint_changes_when_a_document_changes(datasets_directory: Path):
    dataset = datasets_directory / "tiny"
    before = dataset_fingerprint(dataset)

    (dataset / "data" / "stars.txt").write_text("stars shine at night and at dawn")

    assert dataset_fingerprint(dataset) != before


def test_dataset_fingerprint_changes_when_a_task_changes(datasets_directory: Path):
    dataset = datasets_directory / "tiny"
    before = dataset_fingerprint(dataset)

    (dataset / "tasks.json").write_text("[]")

    assert dataset_fingerprint(dataset) != before


def test_setup_replaces_an_old_database(tmp_path: Path):
    (tmp_path / "ostra.db").write_text("not a database")

    database_path = setup(tmp_path)

    assert database_path == tmp_path / "ostra.db"
    assert database_path.read_bytes().startswith(b"SQLite format 3")


def test_load_settings_keeps_the_database_in_the_run_folder(
    run_directory: Path, datasets_directory: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv("OSTRA_DATABASE_PATH", str(run_directory / "elsewhere.db"))

    settings = load_settings(run_directory, datasets_directory)

    assert settings.app_config.database_path == run_directory / "ostra.db"
    assert settings.app_config.directories == [datasets_directory / "tiny" / "data"]
    assert settings.dataset_directory == datasets_directory / "tiny"


def test_load_settings_uses_the_default_search_limit(tmp_path: Path, datasets_directory: Path):
    (tmp_path / "config.json").write_text(json.dumps({"dataset": "tiny"}))

    assert load_settings(tmp_path, datasets_directory).search_limit == 50


def test_load_settings_requires_a_dataset(tmp_path: Path, datasets_directory: Path):
    (tmp_path / "config.json").write_text("{}")

    with pytest.raises(ValueError, match="must name a dataset"):
        load_settings(tmp_path, datasets_directory)


def test_load_settings_rejects_an_unknown_dataset(tmp_path: Path, datasets_directory: Path):
    (tmp_path / "config.json").write_text(json.dumps({"dataset": "missing"}))

    with pytest.raises(ValueError, match="Dataset with a tasks.json not found"):
        load_settings(tmp_path, datasets_directory)


def test_resolve_run_directory_rejects_a_folder_without_config(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        resolve_run_directory(str(tmp_path))


def test_resolve_run_directory_finds_the_baseline_run():
    assert resolve_run_directory("v1-baseline").name == "v1-baseline"


def test_summarize_counts_hits_and_reciprocal_ranks():
    def result(rank: int | None) -> TaskResult:
        task = EvalTask(id=1, question="q", expected="a.txt")
        return TaskResult(task.id, task.question, task.expected, 0.1, "a.txt", 0.2, rank, [])

    summary = summarize([result(1), result(2), result(None), result(4)])

    assert summary["hit_at_1"] == 0.25
    assert summary["hit_at_3"] == 0.5
    assert summary["hit_at_5"] == 0.75
    assert summary["mrr"] == pytest.approx((1 + 0.5 + 0 + 0.25) / 4, abs=1e-4)
    assert summary["not_found"] == 1
