import os
from collections.abc import Sequence
from pathlib import Path

from ostra.cli.init.draft import SetupContext, SetupDraft, SetupServices
from ostra.cli.init.prompter import Prompter, SetupCancelled
from ostra.cli.init.steps import SETUP_STEPS, SetupStep
from ostra.models.app_config import AppConfig
from ostra.repositories.config_repository import read_config_file, save_config

ENVIRONMENT_PREFIX = "OSTRA_"


def run_setup(
    prompter: Prompter,
    config_path: Path,
    services: SetupServices | None = None,
    directories: Sequence[Path] = (),
    steps: Sequence[SetupStep] = SETUP_STEPS,
) -> AppConfig | None:
    """Ask the setup questions, save the settings and install the background service when chosen.

    Args:
        prompter: Asks the questions.
        config_path: The config file to read the current values from and to write.
        services: The outside systems. None means the real Ollama server and systemd.
        directories: Folders to add to the configured ones, from the --directory option.
        steps: The setup steps, in order.

    Returns:
        The saved settings, or None when the user cancelled. Nothing is saved then.
    """
    _print_welcome(config_path)
    try:
        draft = SetupDraft.from_config(_current_config(prompter, config_path))
        context = SetupContext(prompter, draft, config_path, services or SetupServices(), list(directories))
        for step in steps:
            step(context)
    except SetupCancelled:
        print("Setup cancelled. Nothing was saved.")
        return None

    config = draft.to_config()
    save_config(config, config_path)
    print(f"Settings saved to {config_path}.")
    _install_daemon(context)
    return config


def _print_welcome(config_path: Path) -> None:
    """Greet the user and warn about environment variables that override the saved settings."""
    print("Welcome to ostra. This setup writes your settings to the config file.")
    print("Press Ctrl-C at any time to stop without saving.\n")
    overrides = sorted(name for name in os.environ if name.startswith(ENVIRONMENT_PREFIX))
    if overrides:
        print(f"These environment variables override the saved settings: {', '.join(overrides)}\n")


def _current_config(prompter: Prompter, config_path: Path) -> AppConfig:
    """Return the settings of the config file, or the defaults when there is no file.

    Raises:
        SetupCancelled: When the file is not valid and the user does not want to replace it.
    """
    try:
        config = read_config_file(config_path)
    except ValueError as error:
        return _replace_invalid_config(prompter, error)

    if config is None:
        return AppConfig.from_file_values({})

    print(f"Found settings at {config_path}. Press Enter to keep a current value.\n")
    return config


def _replace_invalid_config(prompter: Prompter, error: ValueError) -> AppConfig:
    """Ask before a config file that cannot be read is replaced, so that its settings are not lost by accident.

    Returns:
        The default settings, when the user agrees to replace the file.

    Raises:
        SetupCancelled: Without a person to ask, or when the user wants to keep the file.
    """
    print(f"The config file could not be read.\n{error}\n")
    if not prompter.interactive:
        print("Fix or remove the file, then run ostra init again.")
        raise SetupCancelled

    if not prompter.confirm("Start from the defaults and replace the file when you save?", default=False):
        raise SetupCancelled

    return AppConfig.from_file_values({})


def _install_daemon(context: SetupContext) -> None:
    """Install the background service when the user chose it, and report a failure without raising."""
    interval = context.draft.daemon_interval_minutes
    if interval is None:
        return

    try:
        print(context.services.install_daemon(interval))
    except (RuntimeError, OSError) as error:
        print(f"The background service was not installed: {error}")
