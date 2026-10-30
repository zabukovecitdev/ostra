from abc import ABC, abstractmethod


class Tokenizer(ABC):
    """Define the interface that every tokenizer of an embedding model implements."""

    @abstractmethod
    def count_tokens(self, text: str) -> int:
        """Return the number of model tokens in a text, without the special tokens.

        Args:
            text: The text to count.

        Returns:
            The number of tokens that the model creates for the text.
        """
