from pathlib import Path

from ostra.tokenization.mxbai_tokenizer import MxbaiTokenizer

DATA_DIRECTORY = Path(__file__).parent / "data"


def test_count_tokens__plain_words__counts_word_pieces_without_special_tokens():
    """Count each word and each punctuation mark as a token, without the [CLS] and [SEP] tokens."""
    counter = MxbaiTokenizer()

    assert counter.count_tokens("hello, world!") == 4


def test_count_tokens__split_words__counts_each_word_piece():
    """Count the word pieces of a word that the vocabulary splits."""
    counter = MxbaiTokenizer()

    assert counter.count_tokens("don't") == 3


def test_count_tokens__word_by_word__matches_whole_text_count():
    """Return the same total when each word is counted separately and when the whole text is counted."""
    counter = MxbaiTokenizer()
    text = (DATA_DIRECTORY / "river_ecosystems_field_guide.txt").read_text()

    per_word_total = sum(counter.count_tokens(word) for word in text.split())

    assert per_word_total == counter.count_tokens(text)
