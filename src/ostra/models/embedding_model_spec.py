import hashlib
from dataclasses import dataclass

from ostra.enums.embedding_provider_name import EmbeddingProviderName


@dataclass(frozen=True)
class EmbeddingModelSpec:
    """Describe one embedding model.

    The model ID is the identity that the stored index keeps. A change that alters the vectors needs a new
    model ID.
    """

    model_id: str
    provider: EmbeddingProviderName
    # The name of the model at its provider, such as the Ollama tag or the Hugging Face repository.
    provider_model: str
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
        The provider is part of it, because two providers can create different vectors from the same weights.
        Ollama models leave out the provider name, so that the indexes that earlier versions wrote stay valid.

        Returns:
            The SHA-256 hex digest of the provider model, the dimensions and the document prefix.
        """
        model = self.provider_model
        if self.provider is not EmbeddingProviderName.OLLAMA:
            model = f"{self.provider.value}:{model}"

        text = f"{model}|{self.dimensions}|{self.document_prefix}"
        return hashlib.sha256(text.encode("utf-8")).hexdigest()
