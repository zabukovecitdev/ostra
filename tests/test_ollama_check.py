import httpx
import pytest
from ollama import ListResponse

from ostra.services.setup import ollama_check
from ostra.services.setup.ollama_check import check_ollama, pull_command


class FakeClient:
    """Stand in for the Ollama client and return a fixed model list, or raise an error."""

    models: list[str] = []
    error: Exception | None = None
    hosts: list[str] = []

    def __init__(self, host: str, timeout: float):
        FakeClient.hosts.append(host)

    def list(self) -> ListResponse:
        if FakeClient.error is not None:
            raise FakeClient.error
        return ListResponse(models=[ListResponse.Model(model=name) for name in FakeClient.models])


@pytest.fixture(autouse=True)
def fake_client(monkeypatch: pytest.MonkeyPatch):
    """Replace the Ollama client, so that no test calls a server."""
    FakeClient.models, FakeClient.error, FakeClient.hosts = [], None, []
    monkeypatch.setattr(ollama_check, "Client", FakeClient)


def test_check_ollama__model_pulled__reports_present():
    """Report the model as present when the server lists it."""
    FakeClient.models = ["bge-m3:latest", "llama3:8b"]

    assert check_ollama("http://server:11434", "bge-m3:latest") == ollama_check.OllamaStatus(True, True)
    assert FakeClient.hosts == ["http://server:11434"]


def test_check_ollama__listed_without_tag__matches_latest():
    """Treat a name without a tag as the latest tag."""
    FakeClient.models = ["bge-m3"]

    assert check_ollama("http://server:11434", "bge-m3:latest").model_present


def test_check_ollama__other_tag_only__reports_missing():
    """Report the model as missing when the server has another tag only."""
    FakeClient.models = ["bge-m3:567m"]

    status = check_ollama("http://server:11434", "bge-m3:latest")

    assert (status.reachable, status.model_present) == (True, False)


@pytest.mark.parametrize("error", [ConnectionError("refused"), httpx.ConnectTimeout("timed out")])
def test_check_ollama__server_down__reports_unreachable(error: Exception):
    """Report the server as unreachable with the error text."""
    FakeClient.error = error

    status = check_ollama("http://server:11434", "bge-m3:latest")

    assert (status.reachable, status.model_present) == (False, False)
    assert status.error


def test_pull_command__latest_tag__leaves_tag_out():
    """Print the short model name for the latest tag."""
    assert pull_command("mxbai-embed-large:latest") == "ollama pull mxbai-embed-large"
