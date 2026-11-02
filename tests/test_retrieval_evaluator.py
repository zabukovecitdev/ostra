import json
from pathlib import Path

import pytest

from ostra.evaluation.evaluation_cases import EvaluationCase, RelevantChunk, load_evaluation_cases
from ostra.evaluation.retrieval_evaluator import RetrievalEvaluator
from ostra.store.chunk_match import ChunkMatch


def _match(document_id: str, text: str = "", distance: float = 0.3) -> ChunkMatch:
    return ChunkMatch(
        document_id=document_id, source="file_system", title="t", digest="d", text=text, distance=distance
    )


def _evaluator(results: dict[str, list[ChunkMatch]], queries: list[str] | None = None) -> RetrievalEvaluator:
    def search(query: str, limit: int) -> list[ChunkMatch]:
        if queries is not None:
            queries.append(query)
        return results[query][:limit]

    return RetrievalEvaluator(search)


def test_evaluate__relevant_chunk_second__returns_rank_two_and_half_reciprocal_rank():
    """Rank a case by the position of its first relevant result."""
    evaluator = _evaluator({"rivers": [_match("/a/press.txt"), _match("/a/rivers.txt")]})

    report = evaluator.evaluate([EvaluationCase("rivers", [RelevantChunk("rivers.txt")])], limit=5)

    assert report.cases[0].rank == 2
    assert report.hit_rate == 1.0
    assert report.mean_reciprocal_rank == 0.5


def test_evaluate__relevant_chunk_below_limit__counts_as_miss():
    """Count a case as a miss when the relevant chunk is not in the first `limit` results."""
    evaluator = _evaluator({"rivers": [_match("/a/press.txt"), _match("/a/rivers.txt")]})

    report = evaluator.evaluate([EvaluationCase("rivers", [RelevantChunk("rivers.txt")])], limit=1)

    assert report.cases[0].rank is None
    assert report.hit_rate == 0.0
    assert report.mean_reciprocal_rank == 0.0


def test_evaluate__one_of_two_relevant_found__returns_half_recall():
    """Return the share of the relevant chunks that the search found."""
    evaluator = _evaluator({"q": [_match("/a/rivers.txt", "gravel beds and stones")]})
    case = EvaluationCase("q", [RelevantChunk("rivers.txt", "gravel  beds"), RelevantChunk("rivers.txt", "otters")])

    report = evaluator.evaluate([case], limit=5)

    assert report.recall == 0.5


def test_evaluate__text_not_in_chunk__does_not_match():
    """Require the relevant text inside the chunk when the case sets it."""
    evaluator = _evaluator({"q": [_match("/a/rivers.txt", "floodplains")]})

    report = evaluator.evaluate([EvaluationCase("q", [RelevantChunk("rivers.txt", "otters")])], limit=5)

    assert report.cases[0].rank is None


def test_evaluate__file_name_only_partly_equal__does_not_match():
    """Compare whole path parts, so that `rivers.txt` does not match `old_rivers.txt`."""
    evaluator = _evaluator({"q": [_match("/a/old_rivers.txt")]})

    report = evaluator.evaluate([EvaluationCase("q", [RelevantChunk("rivers.txt")])], limit=5)

    assert report.cases[0].rank is None


def test_evaluate__query_prefix_set__searches_prefixed_query():
    """Put the prefix in front of each query before the search."""
    queries: list[str] = []
    evaluator = _evaluator({"find: q": [_match("/a/rivers.txt")]}, queries)

    evaluator.evaluate([EvaluationCase("q", [RelevantChunk("rivers.txt")])], limit=5, query_prefix="find: ")

    assert queries == ["find: q"]


def test_evaluate__mixed_results__averages_relevant_and_other_distances():
    """Average the distances of relevant and other results apart, to help choose a distance threshold."""
    evaluator = _evaluator({"q": [_match("/a/rivers.txt", distance=0.2), _match("/a/press.txt", distance=0.6)]})

    report = evaluator.evaluate([EvaluationCase("q", [RelevantChunk("rivers.txt")])], limit=5)

    assert report.mean_relevant_distance == 0.2
    assert report.mean_other_distance == 0.6


def test_load_evaluation_cases__valid_file__returns_cases(tmp_path: Path):
    """Read the queries and relevant chunks from the JSON file."""
    path = tmp_path / "cases.json"
    path.write_text(json.dumps([{"query": "q", "relevant": [{"file": "a.txt", "text": "x"}, {"file": "b.txt"}]}]))

    assert load_evaluation_cases(path) == [EvaluationCase("q", [RelevantChunk("a.txt", "x"), RelevantChunk("b.txt")])]


@pytest.mark.parametrize(
    ("content", "message"),
    [
        ("{not json", "not valid JSON"),
        ('{"query": "q"}', "must hold a list"),
        ('[{"relevant": [{"file": "a.txt"}]}]', "no query"),
        ('[{"query": "q", "relevant": []}]', "no relevant chunk"),
    ],
)
def test_load_evaluation_cases__invalid_file__raises_value_error(tmp_path: Path, content: str, message: str):
    """Reject a file that is not a list of complete cases."""
    path = tmp_path / "cases.json"
    path.write_text(content)

    with pytest.raises(ValueError, match=message):
        load_evaluation_cases(path)
