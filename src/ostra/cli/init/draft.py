from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from ostra.cli.init.prompter import Prompter
from ostra.models.app_config import AppConfig
from ostra.services.daemon.systemd_service import install_service, is_supported
from ostra.services.setup.ollama_check import OllamaStatus, check_ollama


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
    # Minutes between the runs of the background service, or None to install no service.
    daemon_interval_minutes: int | None = None

    @classmethod
    def from_config(cls, config: AppConfig) -> SetupDraft:
        """Start the draft from the settings, so that each question offers the current value.

        Args:
            config: The current settings, or the defaults.

        Returns:
            The draft.
        """
        return cls(
            embedding_model=config.embedding_model,
            ollama_base_url=config.ollama_base_url,
            directories=list(config.directories),
            recursive=config.recursive,
            database_path=config.database_path,
            chunk_size_tokens=config.chunk_size_tokens,
            chunk_overlap_tokens=config.chunk_overlap_tokens,
        )

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


@dataclass(frozen=True)
class SetupServices:
    """The outside systems that the setup talks to. Tests replace them."""

    check_ollama: Callable[[str, str], OllamaStatus] = check_ollama
    daemon_supported: Callable[[], bool] = is_supported
    install_daemon: Callable[[int], str] = install_service


@dataclass
class SetupContext:
    """What each setup step needs: the prompter, the answers so far and the outside systems."""

    prompter: Prompter
    draft: SetupDraft
    config_path: Path
    services: SetupServices = field(default_factory=SetupServices)
