import json
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from ostra.embeddings.base import EmbeddingProvider
from ostra.embeddings.ollama_embedding_provider import OllamaEmbeddingProvider, OllamaSettings


class FakeOllamaHandler(BaseHTTPRequestHandler):
    """Answer embed requests with a fixed response and remember the last request body."""

    response_body: dict = {}
    last_request: dict = {}

    def do_POST(self):
        """Read the JSON request and reply with the configured response body."""
        length = int(self.headers["Content-Length"])
        type(self).last_request = json.loads(self.rfile.read(length))
        body = json.dumps(type(self).response_body).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        """Keep the test output free of request logs."""


@pytest.fixture
def fake_server() -> Iterator[HTTPServer]:
    """Run a local HTTP server that imitates the Ollama embed endpoint."""
    server = HTTPServer(("127.0.0.1", 0), FakeOllamaHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server
    server.shutdown()
    server.server_close()


def base_url(server: HTTPServer) -> str:
    """Return the base URL of the fake server."""
    host, port = server.server_address
    return f"http://{host}:{port}"


def test_ollama_embedding_provider__is_an_embedding_provider():
    """Implement the embedding provider interface."""
    assert issubclass(OllamaEmbeddingProvider, EmbeddingProvider)


def test_get_embedding__server_returns_vector__returns_first_embedding(fake_server: HTTPServer):
    """Return the embedding vector from the server response."""
    FakeOllamaHandler.response_body = {"embeddings": [[0.1, 0.2, 0.3]]}
    provider = OllamaEmbeddingProvider(OllamaSettings(base_url=base_url(fake_server), model="test-model"))

    embedding = provider.get_embedding("hello")

    assert embedding == [0.1, 0.2, 0.3]


def test_get_embedding__text_sent__posts_model_and_input(fake_server: HTTPServer):
    """Send the configured model and the text to the embed endpoint."""
    FakeOllamaHandler.response_body = {"embeddings": [[0.1]]}
    provider = OllamaEmbeddingProvider(OllamaSettings(base_url=base_url(fake_server), model="test-model"))

    provider.get_embedding("hello")

    assert FakeOllamaHandler.last_request == {"model": "test-model", "input": "hello"}


def test_get_embedding__no_embedding_in_response__raises_value_error(fake_server: HTTPServer):
    """Raise ValueError when the server response contains no embedding."""
    FakeOllamaHandler.response_body = {"embeddings": []}
    provider = OllamaEmbeddingProvider(OllamaSettings(base_url=base_url(fake_server), model="test-model"))

    with pytest.raises(ValueError, match="contains no embedding"):
        provider.get_embedding("hello")


def test_ollama_settings__defaults__point_to_local_server_and_model(monkeypatch: pytest.MonkeyPatch):
    """Use the local Ollama server and the mxbai-embed-large model by default."""
    monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)
    monkeypatch.delenv("OLLAMA_MODEL", raising=False)

    settings = OllamaSettings()

    assert settings.base_url == "http://localhost:11434"
    assert settings.model == "mxbai-embed-large:latest"


def test_ollama_settings__environment_set__overrides_defaults(monkeypatch: pytest.MonkeyPatch):
    """Read the server address and the model from OLLAMA_ environment variables."""
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://10.0.0.5:11434")
    monkeypatch.setenv("OLLAMA_MODEL", "nomic-embed-text")

    settings = OllamaSettings()

    assert settings.base_url == "http://10.0.0.5:11434"
    assert settings.model == "nomic-embed-text"
