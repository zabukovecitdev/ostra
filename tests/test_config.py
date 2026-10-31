import json
import sys
from pathlib import Path

import pytest
from platformdirs import user_data_path

from ostra.config import (
    DEFAULT_CHUNK_OVERLAP_TOKENS,
    DEFAULT_CHUNK_SIZE_TOKENS,
    DEFAULT_CONFIG_PATH,
    DEFAULT_DATABASE_PATH,
    AppConfig,
    default_config_path,
    load_config,
    save_config,
)
from ostra.connectors.database_facade import DatabaseBackend


def test_load_config__config_file_missing__returns_default_settings(tmp_path: Path):
    """Return the default database path and no folders when the config file does not exist."""
    config = load_config(tmp_path / "ostra" / "config.json")

    assert config.database_path == DEFAULT_DATABASE_PATH
    assert config.directories == []


def test_load_config__config_file_missing__creates_config_file_with_defaults(tmp_path: Path):
    """Create the config file with the default database path and an empty folder list."""
    config_path = tmp_path / "ostra" / "config.json"

    load_config(config_path)

    assert json.loads(config_path.read_text()) == {
        "database_path": str(DEFAULT_DATABASE_PATH),
        "directories": [],
        "database_backend": "sqlite",
        "chunk_size_tokens": 510,
        "chunk_overlap_tokens": 50,
        "embedding_provider": "ollama",
    }


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX permission modes do not apply on Windows")
def test_load_config__config_file_missing__creates_owner_only_directory_and_file(tmp_path: Path):
    """Create the config directory with mode 700 and the config file with mode 600."""
    config_path = tmp_path / "ostra" / "config.json"

    load_config(config_path)

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

    with pytest.raises(ValueError, match="must be a list"):
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
    monkeypatch.setattr("ostra.config.LEGACY_CONFIG_PATH", legacy_path)

    assert default_config_path() == legacy_path


def test_default_config_path__legacy_file_missing__returns_user_config_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Return the config path in the user config folder when no legacy config file exists."""
    monkeypatch.setattr("ostra.config.LEGACY_CONFIG_PATH", tmp_path / "missing" / "config.json")

    assert default_config_path() == DEFAULT_CONFIG_PATH


def test_load_config__chunk_settings_missing__returns_default_chunk_size_and_overlap(tmp_path: Path):
    """Return a chunk size of 510 tokens and an overlap of 50 tokens when the config file has no chunk keys."""
    config_path = tmp_path / "config.json"
    config_path.write_text("{}")

    config = load_config(config_path)

    assert config.chunk_size_tokens == DEFAULT_CHUNK_SIZE_TOKENS == 510
    assert config.chunk_overlap_tokens == DEFAULT_CHUNK_OVERLAP_TOKENS == 50


def test_load_config__chunk_settings_set__returns_configured_values(tmp_path: Path):
    """Return the chunk size and overlap that the config file sets."""
    config_path = tmp_path / "config.json"
    config_path.write_text('{"chunk_size_tokens": 800, "chunk_overlap_tokens": 100}')

    config = load_config(config_path)

    assert config.chunk_size_tokens == 800
    assert config.chunk_overlap_tokens == 100


@pytest.mark.parametrize("value", ['"800"', "1.5", "true"])
def test_load_config__chunk_size_not_an_integer__raises_value_error(tmp_path: Path, value: str):
    """Raise ValueError when the chunk size is not an integer."""
    config_path = tmp_path / "config.json"
    config_path.write_text(f'{{"chunk_size_tokens": {value}}}')

    with pytest.raises(ValueError, match="chunk_size_tokens must be an integer"):
        load_config(config_path)


def test_load_config__backend_key_missing__returns_sqlite_backend(tmp_path: Path):
    """Return the SQLite backend when the config file has no database_backend key."""
    config_path = tmp_path / "config.json"
    config_path.write_text("{}")

    assert load_config(config_path).database_backend is DatabaseBackend.SQLITE


def test_load_config__backend_set_to_chroma__returns_chroma_backend(tmp_path: Path):
    """Return the Chroma backend that the config file sets."""
    config_path = tmp_path / "config.json"
    config_path.write_text('{"database_backend": "chroma"}')

    assert load_config(config_path).database_backend is DatabaseBackend.CHROMA


def test_load_config__backend_not_supported__raises_value_error(tmp_path: Path):
    """Raise ValueError when the config file names a backend that is not supported."""
    config_path = tmp_path / "config.json"
    config_path.write_text('{"database_backend": "postgres"}')

    with pytest.raises(ValueError, match="database_backend must be one of sqlite, chroma"):
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


def test_load_config__unknown_provider__raises_value_error(tmp_path: Path):
    """Reject a provider name that is not supported."""
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"embedding_provider": "magic"}))

    with pytest.raises(ValueError, match="embedding_provider"):
        load_config(config_path)
