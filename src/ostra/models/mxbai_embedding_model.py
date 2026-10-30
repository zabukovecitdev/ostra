from ostra.embeddings.base import EmbeddingProvider
from ostra.embeddings.ollama_embedding_provider import OllamaEmbeddingProvider, OllamaSettings
from ostra.models.base import EmbeddingModel
from ostra.tokenization.base import Tokenizer
from ostra.tokenization.mxbai_tokenizer import MxbaiTokenizer


class MxbaiEmbeddingModel(EmbeddingModel):
    """Embed text with mxbai-embed-large on Ollama and count tokens with its tokenizer."""

    def __init__(self, settings: OllamaSettings):
        self._settings = settings

    def get_embedding_provider(self) -> EmbeddingProvider:
        """Return an Ollama provider that serves the mxbai-embed-large model.

        Returns:
            The Ollama embedding provider configured with the settings of this model.
        """
        return OllamaEmbeddingProvider(self._settings)

    def get_tokenizer(self) -> Tokenizer:
        """Return the tokenizer of the mxbai-embed-large model.

        Returns:
            The mxbai tokenizer.
        """
        return MxbaiTokenizer()
