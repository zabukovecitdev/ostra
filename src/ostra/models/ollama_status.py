from dataclasses import dataclass

from ostra.enums.ollama_state import OllamaState


@dataclass(frozen=True)
class OllamaStatus:
    """Hold the result of an Ollama check."""

    state: OllamaState
    model: str
