from tokenizers import Tokenizer as HuggingFaceTokenizer

from personal_rag.tokenization.base import Tokenizer

MXBAI_TOKENIZER_NAME = "mixedbread-ai/mxbai-embed-large-v1"


class MxbaiTokenizer(Tokenizer):
    """Count tokens with the tokenizer of the mxbai-embed-large model."""

    def __init__(self, tokenizer_name: str = MXBAI_TOKENIZER_NAME):
        self._tokenizer = HuggingFaceTokenizer.from_pretrained(tokenizer_name)

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
