from enum import Enum


class EmbeddingProviderName(Enum):
    """The services that can create the embeddings."""

    # Runs the model in the ostra process on the CPU. The model downloads once on first use.
    FASTEMBED = "fastembed"
    # Sends the text to an Ollama server.
    OLLAMA = "ollama"
