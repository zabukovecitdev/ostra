from pathlib import Path

import pytest

from ostra.config import AppConfig
from ostra.models.factory import build_embedding_model, ollama_settings
from ostra.models.registry import MXBAI_EMBED_LARGE
from ostra.tokenization.hugging_face_tokenizer import HuggingFaceTokenizer


def test_ollama_settings__base_url_in_config__overrides_environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Use the config file address when both the config file and the environment set one."""
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://from-env:11434")
    config = AppConfig(database_path=tmp_path / "db", ollama_base_url="http://from-config:11434")

    assert ollama_settings(config).base_url == "http://from-config:11434"


def test_ollama_settings__no_base_url_in_config__uses_environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Use the environment address when the config file sets none."""
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://from-env:11434")

    assert ollama_settings(AppConfig(database_path=tmp_path / "db")).base_url == "http://from-env:11434"


def test_ollama_settings__model_environment_variable_set__raises_value_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Refuse the old OLLAMA_MODEL variable, so that it does not silently pick another model."""
    monkeypatch.setenv("OLLAMA_MODEL", "nomic-embed-text")

    with pytest.raises(ValueError, match="OLLAMA_MODEL"):
        ollama_settings(AppConfig(database_path=tmp_path / "db"))


def test_build_embedding_model__default_config__returns_mxbai_model_with_its_tokenizer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Return the mxbai spec, a provider and the mxbai tokenizer for the default config."""
    monkeypatch.delenv("OLLAMA_MODEL", raising=False)

    model = build_embedding_model(AppConfig(database_path=tmp_path / "db"))

    assert model.spec == MXBAI_EMBED_LARGE
    assert isinstance(model.tokenizer, HuggingFaceTokenizer)


def test_build_embedding_model__model_id_given__overrides_the_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Use the model that is named instead of the model of the config file."""
    monkeypatch.delenv("OLLAMA_MODEL", raising=False)
    config = AppConfig(database_path=tmp_path / "db", embedding_model="mxbai-embed-large")

    assert build_embedding_model(config, model_id="mxbai-embed-large").spec == MXBAI_EMBED_LARGE


def test_build_embedding_model__unknown_model_id__raises_value_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Refuse a model that is not in the registry."""
    monkeypatch.delenv("OLLAMA_MODEL", raising=False)

    with pytest.raises(ValueError, match="not supported"):
        build_embedding_model(AppConfig(database_path=tmp_path / "db"), model_id="magic")
