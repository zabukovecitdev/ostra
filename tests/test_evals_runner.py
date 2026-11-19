import json
from pathlib import Path

import pytest
from conftest import WordTokenizer, make_spec

from evals.runner import EvalTask, TaskResult, load_settings, resolve_version_directory, run_evals, setup, summarize
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
def version_directory(tmp_path: Path) -> Path:
    """Provide a version folder with three documents and three tasks."""
    version = tmp_path / "v-test"
    data = version / "data"
    data.mkdir(parents=True)
    (data / "bees.txt").write_text("bees make honey and bees dance")
    (data / "rivers.txt").write_text("rivers flow to the sea and rivers carve valleys")
    (data / "stars.txt").write_text("stars shine at night")
    tasks = [
        {"id": 1, "question": "what do bees make", "expected": "bees.txt"},
        {"id": 2, "question": "where do rivers flow", "expected": "rivers.txt"},
        {"id": 3, "question": "what do bees dance about", "expected": "stars.txt"},
    ]
    (version / "tasks.json").write_text(json.dumps(tasks))
    (version / "config.json").write_text(
        json.dumps({"chunk_size_tokens": 20, "chunk_overlap_tokens": 0, "search_limit": 1})
    )
    return version


def test_run_evals_writes_the_rank_of_each_expected_document(version_directory: Path):
    output = run_evals(version_directory, build_model=build_keyword_model)

    written = json.loads((version_directory / "output.json").read_text())
    assert written == output
    assert [task["first_result"] for task in output["tasks"]] == ["bees.txt", "rivers.txt", "bees.txt"]
    assert [task["expected_rank"] for task in output["tasks"]] == [1, 1, None]
    assert all(task["seconds"] >= 0 for task in output["tasks"])
    assert output["indexing"]["documents"] == 3
    assert output["config"]["search_limit"] == 1
    assert output["summary"]["hit_at_1"] == pytest.approx(2 / 3, abs=1e-4)
    assert output["summary"]["not_found"] == 1
    assert (version_directory / "ostra.db").is_file()


def test_run_evals_starts_from_an_empty_database_each_time(version_directory: Path):
    run_evals(version_directory, build_model=build_keyword_model)
    (version_directory / "data" / "stars.txt").unlink()

    output = run_evals(version_directory, build_model=build_keyword_model)

    assert "stars.txt" not in {name for task in output["tasks"] for name in task["top_documents"]}


def test_setup_replaces_an_old_database(tmp_path: Path):
    (tmp_path / "ostra.db").write_text("not a database")

    database_path = setup(tmp_path)

    assert database_path == tmp_path / "ostra.db"
    assert database_path.read_bytes().startswith(b"SQLite format 3")


def test_load_settings_keeps_the_database_in_the_version_folder(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("OSTRA_DATABASE_PATH", str(tmp_path / "elsewhere.db"))

    settings = load_settings(tmp_path)

    assert settings.app_config.database_path == tmp_path / "ostra.db"
    assert settings.app_config.directories == [tmp_path / "data"]
    assert settings.search_limit == 50


def test_resolve_version_directory_rejects_a_folder_without_tasks(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        resolve_version_directory(str(tmp_path))


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
