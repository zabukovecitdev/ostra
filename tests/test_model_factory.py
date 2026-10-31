from pathlib import Path

import pytest

from ostra.config import AppConfig
from ostra.models.factory import build_embedding_model, ollama_settings
from ostra.models.mxbai_embedding_model import MxbaiEmbeddingModel


def test_ollama_settings__base_url_in_config__overrides_environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Use the config file address when both the config file and the environment set one."""
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://from-env:11434")
    config = AppConfig(database_path=tmp_path / "db", ollama_base_url="http://from-config:11434")

    assert ollama_settings(config).base_url == "http://from-config:11434"


def test_ollama_settings__no_base_url_in_config__uses_environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Use the environment address when the config file sets none."""
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://from-env:11434")

    assert ollama_settings(AppConfig(database_path=tmp_path / "db")).base_url == "http://from-env:11434"


def test_build_embedding_model__ollama_provider__returns_mxbai_model(tmp_path: Path):
    """Return the mxbai model for the Ollama provider."""
    assert isinstance(build_embedding_model(AppConfig(database_path=tmp_path / "db")), MxbaiEmbeddingModel)
