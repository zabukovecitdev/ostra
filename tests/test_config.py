import json
import sys
from pathlib import Path

import pytest
from platformdirs import user_data_path

from ostra.constants import DEFAULT_CHUNK_OVERLAP_TOKENS, DEFAULT_DATABASE_PATH
from ostra.errors.config_not_found_error import ConfigNotFoundError
from ostra.models.app_config import AppConfig
from ostra.repositories.config_repository import (
    DEFAULT_CONFIG_PATH,
    default_config_path,
    load_config,
    read_config_file,
    save_config,
    with_environment,
)


def test_load_config__config_file_missing__raises_config_not_found_error_with_path(tmp_path: Path):
    """Raise ConfigNotFoundError that names the path when the config file does not exist."""
    config_path = tmp_path / "ostra" / "config.json"

    with pytest.raises(ConfigNotFoundError) as error:
        load_config(config_path)

    assert error.value.config_path == config_path


def test_load_config__config_file_missing__creates_no_file_or_folder(tmp_path: Path):
    """Leave the disk unchanged when the config file does not exist."""
    config_path = tmp_path / "ostra" / "config.json"

    with pytest.raises(ConfigNotFoundError):
        load_config(config_path)

    assert not config_path.exists()
    assert not config_path.parent.exists()


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX permission modes do not apply on Windows")
def test_save_config__config_file_missing__creates_owner_only_directory_and_file(tmp_path: Path):
    """Create the config directory with mode 700 and the config file with mode 600."""
    config_path = tmp_path / "ostra" / "config.json"

    save_config(AppConfig(database_path=tmp_path / "ostra.db"), config_path)

    assert config_path.parent.stat().st_mode & 0o777 == 0o700
    assert config_path.stat().st_mode & 0o777 == 0o600


def test_load_config__config_file_exists__keeps_file_unchanged(tmp_path: Path):
    """Keep an existing config file unchanged."""
    config_path = tmp_path / "config.json"
    config_path.write_text('{"database_path": "/data/rag/store.db"}')

    load_config(config_path)

    assert config_path.read_text() == '{"database_path": "/data/rag/store.db"}'


def test_load_config__database_path_set__returns_configured_path(tmp_path: Path):
    """Return the database path that the config file sets."""
    config_path = tmp_path / "config.json"
    config_path.write_text('{"database_path": "/data/rag/store.db"}')

    config = load_config(config_path)

    assert config.database_path == Path("/data/rag/store.db")


def test_load_config__database_path_starts_with_tilde__expands_home_directory(tmp_path: Path):
    """Expand a leading tilde in the database path to the home directory."""
    config_path = tmp_path / "config.json"
    config_path.write_text('{"database_path": "~/rag/store.db"}')

    config = load_config(config_path)

    assert config.database_path == Path.home() / "rag" / "store.db"


def test_load_config__database_path_key_missing__returns_default_database_path(tmp_path: Path):
    """Return the default database path when the config file has no database_path key."""
    config_path = tmp_path / "config.json"
    config_path.write_text("{}")

    config = load_config(config_path)

    assert config.database_path == DEFAULT_DATABASE_PATH


def test_load_config__directories_set__returns_configured_directories(tmp_path: Path):
    """Return the directories that the config file lists, with a leading tilde expanded."""
    config_path = tmp_path / "config.json"
    config_path.write_text('{"directories": ["/data/notes", "~/Projects"]}')

    config = load_config(config_path)

    assert config.directories == [Path("/data/notes"), Path.home() / "Projects"]


def test_load_config__directories_key_missing__returns_no_directories(tmp_path: Path):
    """Return no folders when the config file has no directories key."""
    config_path = tmp_path / "config.json"
    config_path.write_text("{}")

    config = load_config(config_path)

    assert config.directories == []


def test_load_config__directories_not_a_list__raises_value_error(tmp_path: Path):
    """Raise ValueError when the directories key is not a list."""
    config_path = tmp_path / "config.json"
    config_path.write_text('{"directories": "/data/notes"}')

    with pytest.raises(ValueError, match=r"directories\n.*valid list"):
        load_config(config_path)


def test_load_config__file_is_invalid_json__raises_value_error(tmp_path: Path):
    """Raise ValueError when the config file does not contain valid JSON."""
    config_path = tmp_path / "config.json"
    config_path.write_text("{not json")

    with pytest.raises(ValueError, match="not valid JSON"):
        load_config(config_path)


def test_default_database_path__module_constant__points_to_user_data_folder():
    """Point the default database path to the ostra.db file in the user data folder."""
    assert DEFAULT_DATABASE_PATH == user_data_path("ostra") / "ostra.db"


def test_default_config_path__legacy_file_exists__returns_legacy_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Return the legacy config path when a config file exists there, so an earlier install keeps its settings."""
    legacy_path = tmp_path / "ostra" / "config.json"
    legacy_path.parent.mkdir()
    legacy_path.write_text("{}")
    monkeypatch.setattr("ostra.repositories.config_repository.LEGACY_CONFIG_PATH", legacy_path)

    assert default_config_path() == legacy_path


def test_default_config_path__legacy_file_missing__returns_user_config_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Return the config path in the user config folder when no legacy config file exists."""
    monkeypatch.setattr("ostra.repositories.config_repository.LEGACY_CONFIG_PATH", tmp_path / "missing" / "config.json")

    assert default_config_path() == DEFAULT_CONFIG_PATH


def test_load_config__chunk_settings_missing__returns_no_chunk_size_and_default_overlap(tmp_path: Path):
    """Return no chunk size, so that the model default applies, and an overlap of 50 tokens."""
    config_path = tmp_path / "config.json"
    config_path.write_text("{}")

    config = load_config(config_path)

    assert config.chunk_size_tokens is None
    assert config.chunk_overlap_tokens == DEFAULT_CHUNK_OVERLAP_TOKENS == 50


def test_load_config__chunk_settings_set__returns_configured_values(tmp_path: Path):
    """Return the chunk size and overlap that the config file sets."""
    config_path = tmp_path / "config.json"
    config_path.write_text('{"chunk_size_tokens": 800, "chunk_overlap_tokens": 100}')

    config = load_config(config_path)

    assert config.chunk_size_tokens == 800
    assert config.chunk_overlap_tokens == 100


@pytest.mark.parametrize("value", ['"many"', "1.5", "true"])
def test_load_config__chunk_size_not_an_integer__raises_value_error(tmp_path: Path, value: str):
    """Raise ValueError when the chunk size is not an integer."""
    config_path = tmp_path / "config.json"
    config_path.write_text(f'{{"chunk_size_tokens": {value}}}')

    with pytest.raises(ValueError, match=r"chunk_size_tokens\n"):
        load_config(config_path)


def test_save_config__settings_saved__load_config_returns_same_settings(tmp_path: Path):
    """Read back the settings that were written."""
    config_path = tmp_path / "config.json"
    config = AppConfig(
        database_path=tmp_path / "db.sqlite",
        directories=[tmp_path / "notes"],
        chunk_size_tokens=200,
        chunk_overlap_tokens=20,
    )

    save_config(config, config_path)

    assert load_config(config_path) == config


def test_load_config__ollama_base_url_set__returns_url(tmp_path: Path):
    """Return the Ollama server address from the config file."""
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"ollama_base_url": "http://gpu-box:11434"}))

    assert load_config(config_path).ollama_base_url == "http://gpu-box:11434"


def test_load_config__ollama_base_url_missing__returns_none(tmp_path: Path):
    """Return no address when the config file has no ollama_base_url key."""
    config_path = tmp_path / "config.json"
    config_path.write_text("{}")

    assert load_config(config_path).ollama_base_url is None


def test_load_config__old_file_with_embedding_provider__keeps_its_model(tmp_path: Path):
    """Load a config file that an earlier version wrote. The model decides the provider, so the old key is ignored."""
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"embedding_provider": "ollama", "embedding_model": "mxbai-embed-large"}))

    assert load_config(config_path).embedding_model == "mxbai-embed-large"


def test_load_config__embedding_model_missing__returns_default_model(tmp_path: Path):
    """Use the model that runs in the ostra process when the config file names no model."""
    config_path = tmp_path / "config.json"
    config_path.write_text("{}")

    assert load_config(config_path).embedding_model == "potion-multilingual-128m"


def test_load_config__embedding_model_unknown__raises_value_error_that_lists_supported_models(tmp_path: Path):
    """Reject a model that is not in the registry."""
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"embedding_model": "magic"}))

    with pytest.raises(
        ValueError, match=r"embedding_model\n.*must be one of potion-multilingual-128m, mxbai-embed-large, bge-m3"
    ):
        load_config(config_path)


def test_save_config__chunk_size_not_set__leaves_the_key_out(tmp_path: Path):
    """Write no chunk size when the model default applies, so that a model change takes its own default."""
    config_path = tmp_path / "config.json"

    save_config(AppConfig(database_path=tmp_path / "db"), config_path)

    assert "chunk_size_tokens" not in json.loads(config_path.read_text())


def test_load_config__old_file_with_chunk_size_510__keeps_the_value(tmp_path: Path):
    """Keep a chunk size that an earlier version wrote into the config file."""
    config_path = tmp_path / "config.json"
    config_path.write_text('{"chunk_size_tokens": 510}')

    assert load_config(config_path).chunk_size_tokens == 510


def test_load_config__environment_variable_set__overrides_config_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Let an OSTRA_ environment variable override the key of the same name in the config file."""
    config_path = tmp_path / "config.json"
    config_path.write_text('{"chunk_size_tokens": 300, "chunk_overlap_tokens": 20}')
    monkeypatch.setenv("OSTRA_CHUNK_SIZE_TOKENS", "800")

    config = load_config(config_path)

    assert config.chunk_size_tokens == 800
    assert config.chunk_overlap_tokens == 20


def test_load_config__directories_environment_variable__reads_json_list(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Read OSTRA_DIRECTORIES as a JSON list and expand a leading tilde."""
    config_path = tmp_path / "config.json"
    config_path.write_text("{}")
    monkeypatch.setenv("OSTRA_DIRECTORIES", '["~/notes", "/data"]')

    assert load_config(config_path).directories == [Path.home() / "notes", Path("/data")]


def test_load_config__unknown_key__is_ignored(tmp_path: Path):
    """Ignore keys that the settings do not know, so that an older or newer config file still loads."""
    config_path = tmp_path / "config.json"
    config_path.write_text('{"removed_setting": 1}')

    assert load_config(config_path).embedding_model == "potion-multilingual-128m"


def test_load_config__file_is_not_an_object__raises_value_error(tmp_path: Path):
    """Raise ValueError when the config file holds JSON that is not an object."""
    config_path = tmp_path / "config.json"
    config_path.write_text("[]")

    with pytest.raises(ValueError, match="must contain a JSON object"):
        load_config(config_path)


def test_read_config_file__file_missing__returns_none_and_creates_nothing(tmp_path: Path):
    """Return None and leave the file missing."""
    config_path = tmp_path / "config.json"

    assert read_config_file(config_path) is None
    assert not config_path.exists()


def test_read_config_file__environment_variable_set__ignores_it(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Read the value of the file, not the OSTRA_ environment variable."""
    monkeypatch.setenv("OSTRA_EMBEDDING_MODEL", "bge-m3")
    config_path = tmp_path / "config.json"
    config_path.write_text('{"embedding_model": "mxbai-embed-large", "recursive": true}')

    config = read_config_file(config_path)

    assert config is not None
    assert (config.embedding_model, config.recursive) == ("mxbai-embed-large", True)


def test_read_config_file__invalid_value__raises_value_error(tmp_path: Path):
    """Raise a ValueError that names the file when a setting is not valid."""
    config_path = tmp_path / "config.json"
    config_path.write_text('{"embedding_model": "unknown"}')

    with pytest.raises(ValueError, match="invalid settings"):
        read_config_file(config_path)


def test_with_environment__environment_variable_set__overrides_given_settings(monkeypatch: pytest.MonkeyPatch):
    """Apply an OSTRA_ variable over the given settings, as load_config does for the file values."""
    monkeypatch.setenv("OSTRA_DATABASE_PATH", "/data/env.db")
    config = AppConfig.from_file_values({"database_path": "/data/file.db", "chunk_size_tokens": 300})

    applied = with_environment(config)

    assert applied.database_path == Path("/data/env.db")
    assert applied.chunk_size_tokens == 300


def test_with_environment__no_environment_variable__keeps_given_settings(monkeypatch: pytest.MonkeyPatch):
    """Return the given settings unchanged when no OSTRA_ variable is set."""
    monkeypatch.delenv("OSTRA_DATABASE_PATH", raising=False)
    config = AppConfig.from_file_values({"database_path": "/data/file.db"})

    assert with_environment(config).model_dump() == config.model_dump()
