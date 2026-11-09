from types import SimpleNamespace

import pytest

from ostra.embeddings.ollama_embedding_provider import OllamaSettings
from ostra.gui import ollama_status
from ostra.gui.ollama_status import OllamaState, check_ollama, is_valid_server_url, model_url


class _FakeClient:
    def __init__(self, models: list[str] | None):
        self._models = models

    def list(self):
        if self._models is None:
            raise ConnectionError("no server")

        return SimpleNamespace(models=[SimpleNamespace(model=name) for name in self._models])


def _patch_client(monkeypatch: pytest.MonkeyPatch, models: list[str] | None) -> None:
    monkeypatch.setattr(ollama_status, "Client", lambda host, timeout: _FakeClient(models))


def test_check_ollama__model_installed__returns_ready(monkeypatch: pytest.MonkeyPatch):
    """Return ready when the server lists the model."""
    _patch_client(monkeypatch, ["mxbai-embed-large:latest"])

    assert check_ollama(OllamaSettings(), "mxbai-embed-large:latest").state == OllamaState.READY


def test_check_ollama__model_name_without_tag__matches_latest_tag(monkeypatch: pytest.MonkeyPatch):
    """Treat a model name without a tag as the latest tag."""
    _patch_client(monkeypatch, ["mxbai-embed-large:latest"])

    assert check_ollama(OllamaSettings(), "mxbai-embed-large").state == OllamaState.READY


def test_check_ollama__other_model_installed__returns_model_missing(monkeypatch: pytest.MonkeyPatch):
    """Return model missing when the server does not list the model."""
    _patch_client(monkeypatch, ["llama3:latest"])

    assert check_ollama(OllamaSettings(), "mxbai-embed-large:latest").state == OllamaState.MODEL_MISSING


def test_check_ollama__server_down__returns_server_unavailable(monkeypatch: pytest.MonkeyPatch):
    """Return server unavailable when the request fails."""
    _patch_client(monkeypatch, None)

    assert check_ollama(OllamaSettings(), "mxbai-embed-large:latest").state == OllamaState.SERVER_UNAVAILABLE


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("http://localhost:11434", True),
        ("https://gpu-box.local", True),
        ("gpu-box:11434", False),
        ("ftp://gpu-box", False),
        ("http://", False),
        ("", False),
    ],
)
def test_is_valid_server_url__text_given__tells_whether_it_is_an_http_address(url: str, expected: bool):
    """Accept http and https addresses with a host and reject other text."""
    assert is_valid_server_url(url) is expected


def test_model_url__model_with_tag__returns_library_page_without_tag():
    """Link to the library page of the model, whatever the tag is."""
    assert model_url("nomic-embed-text:latest") == "https://ollama.com/library/nomic-embed-text"
