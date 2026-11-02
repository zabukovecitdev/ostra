import hashlib
from dataclasses import dataclass


@dataclass(frozen=True)
class EmbeddingModelSpec:
    """Describe one embedding model.

    The model ID is the identity that the stored index keeps. A change that alters the vectors needs a new
    model ID.
    """

    model_id: str
    ollama_model: str
    dimensions: int
    max_tokens: int
    special_tokens: int
    default_chunk_size_tokens: int
    tokenizer_file: str
    query_prefix: str = ""
    document_prefix: str = ""

    @property
    def fingerprint(self) -> str:
        """Return a digest of everything that decides what the stored vectors mean.

        The query prefix is not part of it, because the application creates a query vector fresh for each search.

        Returns:
            The SHA-256 hex digest of the Ollama model, the dimensions and the document prefix.
        """
        text = f"{self.ollama_model}|{self.dimensions}|{self.document_prefix}"
        return hashlib.sha256(text.encode("utf-8")).hexdigest()
