from importlib import resources
from pathlib import Path

from tokenizers import Tokenizer as TokenizersTokenizer

from ostra.tokenization.base import Tokenizer


class HuggingFaceTokenizer(Tokenizer):
    """Count tokens with the tokenizer file of an embedding model.

    The tokenizer files ship with the package, so no download is needed.
    """

    def __init__(self, tokenizer_file: str):
        path = Path(str(resources.files("ostra.tokenization") / "data" / tokenizer_file))
        self._tokenizer = TokenizersTokenizer.from_file(str(path))

    def count_tokens(self, text: str) -> int:
        """Return the number of model tokens in a text, without the special tokens.

        The model tokenizer splits words at whitespace, so counting one word at a time gives the same total as
        counting the whole text.

        Args:
            text: The text to count.

        Returns:
            The number of tokens that the model creates for the text.
        """
        return len(self._tokenizer.encode(text, add_special_tokens=False).ids)
