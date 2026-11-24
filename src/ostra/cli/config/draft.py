from collections.abc import Callable
from dataclasses import dataclass, field, replace
from pathlib import Path

from ostra.cli.config.prompter import Prompter
from ostra.models.app_config import AppConfig
from ostra.services.setup.ollama_check import OllamaStatus, check_ollama

# The settings that decide what the index holds, with the name the user sees. A change to one of them needs a run
# of ostra index. The Ollama address is not one of them: another server with the same model makes the same vectors.
MODEL_CHANGE = "model"
INDEX_SETTING_NAMES = {
    "embedding_model": MODEL_CHANGE,
    "directories": "folders",
    "recursive": "subfolders",
    "chunk_size_tokens": "chunk size",
    "chunk_overlap_tokens": "chunk overlap",
    "database_path": "database",
}


@dataclass
class SetupDraft:
    """The answers of the setup so far. Each step reads and changes it."""

    embedding_model: str
    ollama_base_url: str | None
    directories: list[Path]
    recursive: bool
    database_path: Path
    chunk_size_tokens: int | None
    chunk_overlap_tokens: int

    @classmethod
    def from_config(cls, config: AppConfig) -> SetupDraft:
        """Start the draft from the settings, so that each question offers the current value.

        The folders become absolute, so that a relative folder of the file and the same folder typed as an
        absolute path count as one folder.

        Args:
            config: The current settings, or the defaults.

        Returns:
            The draft.
        """
        return cls(
            embedding_model=config.embedding_model,
            ollama_base_url=config.ollama_base_url,
            directories=[folder.absolute() for folder in config.directories],
            recursive=config.recursive,
            database_path=config.database_path,
            chunk_size_tokens=config.chunk_size_tokens,
            chunk_overlap_tokens=config.chunk_overlap_tokens,
        )

    def copy(self) -> SetupDraft:
        """Return a copy that later answers do not change."""
        return replace(self, directories=list(self.directories))

    def to_config(self) -> AppConfig:
        """Validate the answers as settings, without the OSTRA_ environment variables.

        Returns:
            The settings to save.
        """
        return AppConfig.from_file_values(
            {
                "embedding_model": self.embedding_model,
                "ollama_base_url": self.ollama_base_url,
                "directories": self.directories,
                "recursive": self.recursive,
                "database_path": self.database_path,
                "chunk_size_tokens": self.chunk_size_tokens,
                "chunk_overlap_tokens": self.chunk_overlap_tokens,
            }
        )


def index_changes(before: SetupDraft, after: SetupDraft) -> tuple[str, ...]:
    """Name the settings that changed and need a run of ostra index to apply.

    Args:
        before: The settings when the setup started.
        after: The answers.

    Returns:
        The names of the changed settings, in a fixed order. The order of the folders does not count.
    """
    return tuple(
        name for key, name in INDEX_SETTING_NAMES.items() if _comparable(before, key) != _comparable(after, key)
    )


def _comparable(draft: SetupDraft, key: str) -> object:
    """Return a setting in a form that compares equal when only the order of the folders differs."""
    value = getattr(draft, key)
    return frozenset(value) if isinstance(value, list) else value


@dataclass(frozen=True)
class SetupServices:
    """The outside systems that the setup talks to. Tests replace them."""

    check_ollama: Callable[[str, str], OllamaStatus] = check_ollama


@dataclass
class SetupContext:
    """What each setup step needs: the prompter, the answers so far and the outside systems."""

    prompter: Prompter
    draft: SetupDraft
    config_path: Path
    services: SetupServices = field(default_factory=SetupServices)
    # Folders from the --directory option. They are added after the user decides about the configured folders.
    given_directories: list[Path] = field(default_factory=list)
    # The settings when the setup started, to tell which answers change the index.
    starting: SetupDraft | None = None

    def index_changes(self) -> tuple[str, ...]:
        """Name the answers so far that need a run of ostra index."""
        return index_changes(self.starting or self.draft, self.draft)
