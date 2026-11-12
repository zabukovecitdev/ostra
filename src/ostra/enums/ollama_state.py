from enum import Enum


class OllamaState(Enum):
    """The state of the Ollama server and its embedding model."""

    READY = "ready"
    MODEL_MISSING = "model_missing"
    SERVER_UNAVAILABLE = "server_unavailable"
