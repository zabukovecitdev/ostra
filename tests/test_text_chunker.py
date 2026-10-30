import pytest

from ostra.services.text_chunker import TextChunker


def one_token_per_word(word: str) -> int:
    """Count each word as one token so that the budgets are easy to follow in the tests."""
    return 1


def chunk(words: list[str], chunk_size_tokens: int, chunk_overlap_tokens: int) -> list[str]:
    """Split the words with a chunker and return the chunk texts."""
    chunker = TextChunker(chunk_size_tokens, chunk_overlap_tokens, count_tokens=one_token_per_word)
    return list(chunker.stream_chunks(iter(words)))


def test_stream_chunks__no_words__yields_nothing():
    """Yield no chunk when there are no words."""
    assert chunk([], 3, 1) == []


def test_stream_chunks__fewer_words_than_size__yields_one_chunk():
    """Yield one chunk with all words when the text is shorter than the chunk size."""
    assert chunk(["a", "b"], 3, 1) == ["a b"]


def test_stream_chunks__exactly_one_chunk__yields_no_repeated_tail():
    """Yield one chunk and no extra chunk when the words fill exactly one chunk."""
    assert chunk(["a", "b", "c"], 3, 1) == ["a b c"]


def test_stream_chunks__overlap_set__repeats_last_words_in_next_chunk():
    """Start each chunk with the last words of the previous chunk."""
    assert chunk(["a", "b", "c", "d", "e"], 3, 1) == ["a b c", "c d e"]


def test_stream_chunks__no_overlap__yields_consecutive_chunks():
    """Yield chunks without repeated words when the overlap is 0."""
    assert chunk(["a", "b", "c", "d"], 2, 0) == ["a b", "c d"]


def test_stream_chunks__partial_last_chunk__yields_only_new_words_at_end():
    """Yield the final chunk with the overlap words and the words that follow them."""
    assert chunk(["a", "b", "c", "d", "e", "f"], 4, 1) == ["a b c d", "d e f"]


def test_stream_chunks__overlap_only_left__yields_no_extra_chunk():
    """Yield no extra chunk when the words after the last full chunk are all overlap words."""
    assert chunk(["a", "b", "c", "d"], 3, 2) == ["a b c", "b c d"]


def test_stream_chunks__generator_input__yields_chunks_in_order():
    """Accept a generator of words and yield the chunks in reading order."""
    words = (word for word in ["a", "b", "c", "d"])

    chunks = list(TextChunker(3, 1, count_tokens=one_token_per_word).stream_chunks(words))

    assert chunks == ["a b c", "c d"]


@pytest.mark.parametrize("chunk_size_tokens", [0, -1])
def test_text_chunker__chunk_size_below_one__raises_value_error(chunk_size_tokens: int):
    """Raise ValueError when the chunk size is below one token."""
    with pytest.raises(ValueError, match="at least 1 token"):
        TextChunker(chunk_size_tokens, 0, count_tokens=one_token_per_word)


def test_text_chunker__overlap_equal_to_size__raises_value_error():
    """Raise ValueError when the overlap is not smaller than the chunk size."""
    with pytest.raises(ValueError, match="smaller than the chunk size"):
        TextChunker(3, 3, count_tokens=one_token_per_word)


def test_text_chunker__negative_overlap__raises_value_error():
    """Raise ValueError when the overlap is negative."""
    with pytest.raises(ValueError, match="at least 0"):
        TextChunker(3, -1, count_tokens=one_token_per_word)
