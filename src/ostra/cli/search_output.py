import json
from dataclasses import dataclass

from ostra.models.chunk_match import ChunkMatch

MAX_SHORT_TEXT_CHARACTERS = 300


@dataclass(frozen=True)
class SearchOutputOptions:
    """Hold how the search results are printed."""

    full_text: bool = False
    as_json: bool = False


def print_search_results(matches: list[ChunkMatch], options: SearchOutputOptions) -> None:
    """Print the chunks of a search, nearest first.

    Args:
        matches: The chunks to print.
        options: Print the whole text of each chunk instead of a shortened text, and print JSON instead of text.
    """
    if options.as_json:
        print(json.dumps([_match_to_dict(match) for match in matches], ensure_ascii=False, indent=2))
        return

    if not matches:
        print("No matching chunks.")
        return

    for match in matches:
        print(f"match {round(_similarity(match.distance) * 100)}%  file={match.document_id}")
        print(_shorten_text(match.text, options.full_text))
        print()


def _similarity(distance: float) -> float:
    """Convert a cosine distance to a similarity.

    Args:
        distance: The cosine distance of a chunk. 0 means the same direction and 1 means unrelated.

    Returns:
        The similarity from -1 to 1, where 1 means the same direction.
    """
    return 1 - distance


def _shorten_text(text: str, full_text: bool) -> str:
    """Cut a chunk text to a short preview, unless the full text is requested.

    Args:
        text: The text of the chunk.
        full_text: Return the text unchanged when True.

    Returns:
        The text, or its first characters followed by an ellipsis when it is long.
    """
    if full_text or len(text) <= MAX_SHORT_TEXT_CHARACTERS:
        return text

    return text[:MAX_SHORT_TEXT_CHARACTERS].rstrip() + "..."


def _match_to_dict(match: ChunkMatch) -> dict:
    """Convert a search match to a dictionary that holds the full text.

    Args:
        match: The chunk match from the search.

    Returns:
        The document ID, source, title, distance, similarity and full text of the chunk.
    """
    return {
        "document_id": match.document_id,
        "source": match.source,
        "title": match.title,
        "file_path": match.document_id,
        "distance": match.distance,
        "similarity": round(_similarity(match.distance), 4),
        "text": match.text,
    }
