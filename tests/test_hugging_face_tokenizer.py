from pathlib import Path

from ostra.tokenization.hugging_face_tokenizer import HuggingFaceTokenizer

DATA_DIRECTORY = Path(__file__).parent / "data"


def test_count_tokens__plain_words__counts_word_pieces_without_special_tokens():
    """Count each word and each punctuation mark as a token, without the [CLS] and [SEP] tokens."""
    counter = HuggingFaceTokenizer("mxbai-embed-large-v1/tokenizer.json")

    assert counter.count_tokens("hello, world!") == 4


def test_count_tokens__split_words__counts_each_word_piece():
    """Count the word pieces of a word that the vocabulary splits."""
    counter = HuggingFaceTokenizer("mxbai-embed-large-v1/tokenizer.json")

    assert counter.count_tokens("don't") == 3


def test_count_tokens__word_by_word__matches_whole_text_count():
    """Return the same total when each word is counted separately and when the whole text is counted."""
    counter = HuggingFaceTokenizer("mxbai-embed-large-v1/tokenizer.json")
    text = (DATA_DIRECTORY / "river_ecosystems_field_guide.txt").read_text()

    per_word_total = sum(counter.count_tokens(word) for word in text.split())

    assert per_word_total == counter.count_tokens(text)
