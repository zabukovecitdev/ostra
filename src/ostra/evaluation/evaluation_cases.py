import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class RelevantChunk:
    """Describe a chunk that a query must find.

    A stored chunk matches when its file path ends with `file`. When `text` is set, the chunk text must also
    contain it. Whitespace differences are ignored.
    """

    file: str
    text: str | None = None


@dataclass(frozen=True)
class EvaluationCase:
    """Hold one query and the chunks that the search must return for it."""

    query: str
    relevant: list[RelevantChunk] = field(default_factory=list)


def load_evaluation_cases(path: Path) -> list[EvaluationCase]:
    """Read the evaluation cases from a JSON file.

    The file holds a list of objects such as
    {"query": "...", "relevant": [{"file": "notes/rivers.txt", "text": "optional part of the chunk"}]}.

    Args:
        path: Location of the JSON file.

    Returns:
        The cases in the order of the file.

    Raises:
        ValueError: When the file is not valid JSON or a case has no query or no relevant chunk.
    """
    try:
        raw_cases = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"Evaluation file is not valid JSON. Path={path}") from error

    if not isinstance(raw_cases, list):
        raise ValueError(f"Evaluation file must hold a list of cases. Path={path}")

    return [_to_case(raw_case, position) for position, raw_case in enumerate(raw_cases, start=1)]


def _to_case(raw_case: dict, position: int) -> EvaluationCase:
    """Convert one JSON object to an evaluation case.

    Args:
        raw_case: The object from the JSON file.
        position: The 1-based position of the case in the file, for the error message.

    Returns:
        The evaluation case.

    Raises:
        ValueError: When the query or the relevant chunks are missing.
    """
    query = str(raw_case.get("query", "")).strip()
    raw_relevant = raw_case.get("relevant") or []
    if not query:
        raise ValueError(f"Evaluation case has no query. Case={position}")

    if not raw_relevant:
        raise ValueError(f"Evaluation case has no relevant chunk. Case={position}")

    relevant = [RelevantChunk(file=item["file"], text=item.get("text")) for item in raw_relevant]

    return EvaluationCase(query=query, relevant=relevant)
