from collections.abc import Callable, Iterable, Iterator


class TextChunker:
    """Split a stream of words into chunks of a fixed token budget with an overlap between chunks."""

    def __init__(self, chunk_size_tokens: int, chunk_overlap_tokens: int, count_tokens: Callable[[str], int]):
        if chunk_size_tokens < 1:
            raise ValueError(f"Chunk size must be at least 1 token. ChunkSizeTokens={chunk_size_tokens}")

        if chunk_overlap_tokens < 0 or chunk_overlap_tokens >= chunk_size_tokens:
            raise ValueError(
                "Chunk overlap must be at least 0 and smaller than the chunk size. "
                f"ChunkOverlapTokens={chunk_overlap_tokens} ChunkSizeTokens={chunk_size_tokens}"
            )

        self._chunk_size_tokens = chunk_size_tokens
        self._chunk_overlap_tokens = chunk_overlap_tokens
        self._count_tokens = count_tokens

    def stream_chunks(self, words: Iterable[str]) -> Iterator[str]:
        """Yield each chunk of words, joined with single spaces.

        A chunk ends before a word that would push it over the token budget. The next chunk starts with the
        last words of the previous chunk, up to the overlap budget. No chunk repeats words that were already
        yielded, except the overlap words.

        Args:
            words: The words to split, in reading order.

        Yields:
            The text of one chunk.
        """
        buffer: list[tuple[str, int]] = []
        buffer_tokens = 0
        unyielded_words = 0
        for word in words:
            word_tokens = self._count_tokens(word)
            if buffer and buffer_tokens + word_tokens > self._chunk_size_tokens:
                yield " ".join(text for text, _ in buffer)
                while buffer and buffer_tokens > self._chunk_overlap_tokens:
                    _, removed_tokens = buffer.pop(0)
                    buffer_tokens -= removed_tokens
                unyielded_words = 0

            buffer.append((word, word_tokens))
            buffer_tokens += word_tokens
            unyielded_words += 1

        if unyielded_words > 0:
            yield " ".join(text for text, _ in buffer)
