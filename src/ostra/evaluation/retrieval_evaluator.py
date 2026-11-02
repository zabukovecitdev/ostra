from collections.abc import Callable
from dataclasses import dataclass
from pathlib import PurePath
from statistics import fmean

from ostra.evaluation.evaluation_cases import EvaluationCase, RelevantChunk
from ostra.store.chunk_match import ChunkMatch

SearchFunction = Callable[[str, int], list[ChunkMatch]]


@dataclass(frozen=True)
class CaseResult:
    """Hold the search result of one evaluation case."""

    query: str
    rank: int | None
    recall: float
    relevant_distances: list[float]
    other_distances: list[float]


@dataclass(frozen=True)
class EvaluationReport:
    """Hold the retrieval metrics over all evaluation cases."""

    limit: int
    cases: list[CaseResult]
    hit_rate: float
    recall: float
    mean_reciprocal_rank: float
    mean_relevant_distance: float | None
    mean_other_distance: float | None


class RetrievalEvaluator:
    """Run each evaluation query through the search and measure how well the relevant chunks rank."""

    def __init__(self, search: SearchFunction):
        self._search = search

    def evaluate(self, cases: list[EvaluationCase], limit: int, query_prefix: str = "") -> EvaluationReport:
        """Search every case and compute the hit rate, recall and mean reciprocal rank at the limit.

        Args:
            cases: The queries and their relevant chunks.
            limit: The number of results to read for each query (the k in recall@k).
            query_prefix: Text to put in front of each query before the search.

        Returns:
            The metrics, averaged over the cases, and the result of each case.
        """
        results = [self._evaluate_case(case, limit, query_prefix) for case in cases]
        relevant_distances = [distance for result in results for distance in result.relevant_distances]
        other_distances = [distance for result in results for distance in result.other_distances]

        return EvaluationReport(
            limit=limit,
            cases=results,
            hit_rate=_mean([1.0 if result.rank is not None else 0.0 for result in results]),
            recall=_mean([result.recall for result in results]),
            mean_reciprocal_rank=_mean([1 / result.rank if result.rank else 0.0 for result in results]),
            mean_relevant_distance=fmean(relevant_distances) if relevant_distances else None,
            mean_other_distance=fmean(other_distances) if other_distances else None,
        )

    def _evaluate_case(self, case: EvaluationCase, limit: int, query_prefix: str) -> CaseResult:
        """Search one case and find the positions of its relevant chunks.

        Args:
            case: The query and its relevant chunks.
            limit: The number of results to read.
            query_prefix: Text to put in front of the query.

        Returns:
            The rank of the first relevant result, the share of relevant chunks found, and the distances.
        """
        matches = self._search(query_prefix + case.query, limit)
        found: set[RelevantChunk] = set()
        rank = None
        relevant_distances = []
        other_distances = []
        for position, match in enumerate(matches, start=1):
            matched = [relevant for relevant in case.relevant if _is_match(match, relevant)]
            if not matched:
                other_distances.append(match.distance)
                continue

            relevant_distances.append(match.distance)
            found.update(matched)
            if rank is None:
                rank = position

        return CaseResult(
            query=case.query,
            rank=rank,
            recall=len(found) / len(case.relevant),
            relevant_distances=relevant_distances,
            other_distances=other_distances,
        )


def _is_match(match: ChunkMatch, relevant: RelevantChunk) -> bool:
    """Check whether a search result is the relevant chunk.

    Args:
        match: The search result.
        relevant: The chunk that the query must find.

    Returns:
        True when the document ID ends with the relevant file and the chunk holds the relevant text.
    """
    relevant_parts = PurePath(relevant.file).parts
    if PurePath(match.document_id).parts[-len(relevant_parts) :] != relevant_parts:
        return False

    if relevant.text is None:
        return True

    return _normalize(relevant.text) in _normalize(match.text)


def _normalize(text: str) -> str:
    """Collapse every run of whitespace to one space, so that line breaks do not prevent a match.

    Args:
        text: The text to normalize.

    Returns:
        The text with single spaces.
    """
    return " ".join(text.split())


def _mean(values: list[float]) -> float:
    """Return the mean of the values, or 0 when there are none.

    Args:
        values: The values to average.

    Returns:
        The arithmetic mean.
    """
    if not values:
        return 0.0

    return fmean(values)
