import json
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest
from conftest import make_spec

from ostra.embeddings.base import EmbeddingProvider
from ostra.embeddings.ollama_embedding_provider import OllamaEmbeddingProvider, OllamaSettings
from ostra.embeddings.vector import Vector

SPEC = make_spec("test-model", 3, query_prefix="query: ", document_prefix="doc: ")


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


def make_provider(server: HTTPServer) -> OllamaEmbeddingProvider:
    """Build a provider for the fake server."""
    host, port = server.server_address
    return OllamaEmbeddingProvider(SPEC, OllamaSettings(base_url=f"http://{host}:{port}"))


def test_ollama_embedding_provider__is_an_embedding_provider():
    """Implement the embedding provider interface."""
    assert issubclass(OllamaEmbeddingProvider, EmbeddingProvider)


def test_embed_document__server_returns_vector__returns_vector_with_model_id(fake_server: HTTPServer):
    """Return the embedding from the server response, stamped with the model ID."""
    FakeOllamaHandler.response_body = {"embeddings": [[0.1, 0.2, 0.3]]}

    vector = make_provider(fake_server).embed_document("hello")

    assert vector == Vector(model_id="test-model", values=(0.1, 0.2, 0.3))


def test_embed_document__text_sent__posts_model_prefixed_input_and_no_truncation(fake_server: HTTPServer):
    """Send the Ollama model name, the text with the document prefix, and the order not to cut the text."""
    FakeOllamaHandler.response_body = {"embeddings": [[0.1, 0.2, 0.3]]}

    make_provider(fake_server).embed_document("hello")

    assert FakeOllamaHandler.last_request == {"model": "test-model:latest", "input": "doc: hello", "truncate": False}


def test_embed_query__text_sent__posts_input_with_query_prefix(fake_server: HTTPServer):
    """Put the query prefix, and not the document prefix, in front of a query."""
    FakeOllamaHandler.response_body = {"embeddings": [[0.1, 0.2, 0.3]]}

    make_provider(fake_server).embed_query("hello")

    assert FakeOllamaHandler.last_request["input"] == "query: hello"


def test_embed_document__no_embedding_in_response__raises_value_error(fake_server: HTTPServer):
    """Raise ValueError when the server response contains no embedding."""
    FakeOllamaHandler.response_body = {"embeddings": []}

    with pytest.raises(ValueError, match="contains no embedding"):
        make_provider(fake_server).embed_document("hello")


def test_embed_document__vector_has_wrong_size__raises_value_error(fake_server: HTTPServer):
    """Raise ValueError when the model returns another number of values than its spec says."""
    FakeOllamaHandler.response_body = {"embeddings": [[0.1, 0.2]]}

    with pytest.raises(ValueError, match="wrong size.*Expected=3 Actual=2"):
        make_provider(fake_server).embed_document("hello")


def test_ollama_settings__defaults__point_to_local_server(monkeypatch: pytest.MonkeyPatch):
    """Use the local Ollama server by default."""
    monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)

    assert OllamaSettings().base_url == "http://localhost:11434"


def test_ollama_settings__environment_set__overrides_server_address(monkeypatch: pytest.MonkeyPatch):
    """Read the server address from the OLLAMA_BASE_URL environment variable."""
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://10.0.0.5:11434")

    assert OllamaSettings().base_url == "http://10.0.0.5:11434"
