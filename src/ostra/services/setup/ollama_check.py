from dataclasses import dataclass

import httpx
from ollama import Client, ResponseError

CONNECT_TIMEOUT_SECONDS = 5.0
DEFAULT_TAG = ":latest"


@dataclass(frozen=True)
class OllamaStatus:
    """What an Ollama server reported about itself and one model."""

    reachable: bool
    model_present: bool = False
    error: str = ""


def check_ollama(base_url: str, provider_model: str) -> OllamaStatus:
    """Ask an Ollama server whether it runs and whether it has the model.

    Args:
        base_url: The address of the server, for example http://localhost:11434.
        provider_model: The Ollama name of the model, for example bge-m3:latest.

    Returns:
        Whether the server answered, whether it has the model, and the error when it did not answer.
    """
    try:
        response = Client(host=base_url, timeout=CONNECT_TIMEOUT_SECONDS).list()
    except (ConnectionError, httpx.HTTPError, ResponseError) as error:
        return OllamaStatus(reachable=False, error=str(error))

    names = {_with_tag(model.model) for model in response.models if model.model}
    return OllamaStatus(reachable=True, model_present=_with_tag(provider_model) in names)


def pull_command(provider_model: str) -> str:
    """Return the command that downloads the model to the Ollama server.

    Args:
        provider_model: The Ollama name of the model.

    Returns:
        The ollama pull command, without the default tag.
    """
    return f"ollama pull {provider_model.removesuffix(DEFAULT_TAG)}"


def _with_tag(name: str) -> str:
    """Add the default tag to a model name without a tag, so that bge-m3 and bge-m3:latest compare equal."""
    return name if ":" in name else f"{name}{DEFAULT_TAG}"
