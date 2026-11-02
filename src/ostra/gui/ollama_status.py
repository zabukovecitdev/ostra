from dataclasses import dataclass
from enum import Enum
from urllib.parse import urlparse

from ollama import Client

from ostra.embeddings.ollama_embedding_provider import OllamaSettings

CHECK_TIMEOUT_SECONDS = 3
INSTALL_URL = "https://ollama.com/download"
MODEL_LIBRARY_URL = "https://ollama.com/library/"


class OllamaState(Enum):
    """The state of the Ollama server and its embedding model."""

    READY = "ready"
    MODEL_MISSING = "model_missing"
    SERVER_UNAVAILABLE = "server_unavailable"


@dataclass(frozen=True)
class OllamaStatus:
    """Hold the result of an Ollama check."""

    state: OllamaState
    model: str


def check_ollama(settings: OllamaSettings, ollama_model: str) -> OllamaStatus:
    """Check that the Ollama server runs and has the embedding model.

    Args:
        settings: The Ollama server address.
        ollama_model: The name of the model that must be installed on the server.

    Returns:
        The state: ready, model missing, or server unavailable. The check stops after a few seconds.
    """
    model = ollama_model if ":" in ollama_model else f"{ollama_model}:latest"
    try:
        installed = Client(host=settings.base_url, timeout=CHECK_TIMEOUT_SECONDS).list().models
    except Exception:
        return OllamaStatus(state=OllamaState.SERVER_UNAVAILABLE, model=model)

    if any(entry.model == model for entry in installed):
        return OllamaStatus(state=OllamaState.READY, model=model)

    return OllamaStatus(state=OllamaState.MODEL_MISSING, model=model)


def is_valid_server_url(url: str) -> bool:
    """Tell whether a text is an http or https address with a host.

    Args:
        url: The text to test.

    Returns:
        True when the text has the form http://host or https://host with an optional port and path.
    """
    parsed = urlparse(url.strip())
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def model_url(ollama_model: str) -> str:
    """Return the page of a model in the Ollama library.

    Args:
        ollama_model: The name of the model, with or without a tag.

    Returns:
        The address of the model page.
    """
    return MODEL_LIBRARY_URL + ollama_model.split(":")[0]
