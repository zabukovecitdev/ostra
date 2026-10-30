from importlib import resources
from pathlib import Path

from tokenizers import Tokenizer as HuggingFaceTokenizer

from ostra.tokenization.base import Tokenizer

BUNDLED_MXBAI_TOKENIZER_PATH = Path(
    str(resources.files("ostra.tokenization") / "data" / "mxbai-embed-large-v1" / "tokenizer.json")
)


class MxbaiTokenizer(Tokenizer):
    """Count tokens with the tokenizer of the mxbai-embed-large model.

    The tokenizer file ships with the package, so no download is needed.
    """

    def __init__(self, tokenizer_path: Path = BUNDLED_MXBAI_TOKENIZER_PATH):
        self._tokenizer = HuggingFaceTokenizer.from_file(str(tokenizer_path))

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
