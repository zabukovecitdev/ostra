import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from ostra.cli.config.draft import MODEL_CHANGE, SetupContext, SetupDraft, SetupServices
from ostra.cli.config.menu import MENU_ENTRIES, run_menu
from ostra.cli.config.prompter import Prompter, SetupCancelled
from ostra.cli.config.steps import WALKTHROUGH_STEPS, add_given_folders, run_steps, save_step
from ostra.models.app_config import AppConfig
from ostra.repositories.config_repository import read_config_file, save_config

ENVIRONMENT_PREFIX = "OSTRA_"


@dataclass(frozen=True)
class SetupResult:
    """The saved settings and what the user must do so that the index follows them."""

    config: AppConfig
    # The names of the changed settings that need a run of ostra index, such as model or chunk size.
    index_changes: tuple[str, ...]
    # The embedding model before the setup, which searches keep using until the settings are saved.
    previous_model: str
    # True when the user chose to index right after the save step. A model change is then saved by the caller
    # only after indexing succeeds. Other changes are saved first.
    index_now: bool
    # False on the first setup, or when the config file could not be read and the user replaced it. Nothing
    # was saved before, so no settings are in use yet.
    had_config_file: bool

    @property
    def model_changed(self) -> bool:
        """Tell whether the model changed, so that searches find nothing until the new index is filled."""
        return MODEL_CHANGE in self.index_changes


def run_setup(
    prompter: Prompter,
    config_path: Path,
    services: SetupServices | None = None,
    directories: Sequence[Path] = (),
    shortcut: str | None = None,
) -> SetupResult | None:
    """Ask the setup questions and save the settings.

    The first setup, or a config file without folders, walks through every setting. Otherwise the user gets a menu
    of the settings. Without a person to answer, the setup walks through the settings with their defaults.

    Args:
        prompter: Asks the questions.
        config_path: The config file to read the current values from and to write.
        services: The outside systems. None means the real Ollama server.
        directories: Folders to add to the configured ones, from the --directory option.
        shortcut: The one setting to change, such as model, when the menu is shown. Ignored in the walkthrough.

    Returns:
        The saved settings and the changes that need indexing, or None when the user cancelled. Nothing is
        saved then. A model change that the user chose to index first is saved by the caller, after indexing.
    """
    try:
        context = _ask(prompter, config_path, services, directories, shortcut)
    except SetupCancelled:
        print("Setup cancelled. Nothing was saved.")
        return None

    return _save(context)


def _ask(
    prompter: Prompter,
    config_path: Path,
    services: SetupServices | None,
    directories: Sequence[Path],
    shortcut: str | None,
) -> SetupContext:
    """Read the current settings, ask the questions and return the answers in the context."""
    existing = _read_existing(prompter, config_path)
    walkthrough = existing is None or not existing.directories or not prompter.interactive
    _print_intro(config_path, walkthrough, found=existing is not None)
    draft = SetupDraft.from_config(existing if existing is not None else AppConfig.from_file_values({}))
    context = SetupContext(
        prompter,
        draft,
        config_path,
        services or SetupServices(),
        list(directories),
        starting=draft.copy(),
        had_config_file=existing is not None,
    )
    _run_flow(context, walkthrough, shortcut)
    return context


def _run_flow(context: SetupContext, walkthrough: bool, shortcut: str | None) -> None:
    """Run the walkthrough, the one shortcut setting or the menu. Each one ends with the save step."""
    if walkthrough:
        run_steps(WALKTHROUGH_STEPS, context)
        _finish(context)
        return

    add_given_folders(context)
    if shortcut is None:
        run_menu(context)
        return

    run_steps(MENU_ENTRIES[shortcut], context)
    _finish(context)


def _finish(context: SetupContext) -> None:
    """Run the save step, and show the menu again when the user chooses to go back to it."""
    if not save_step(context):
        run_menu(context)


def _save(context: SetupContext) -> SetupResult:
    """Save the answers, unless the model changes and the user chose to index first.

    Returns:
        The result, with the settings that were answered.
    """
    config = context.draft.to_config()
    previous_model = (context.starting or context.draft).embedding_model
    result = SetupResult(
        config=config,
        index_changes=context.index_changes(),
        previous_model=previous_model,
        index_now=context.index_now,
        had_config_file=context.had_config_file,
    )
    if result.index_now and result.model_changed:
        return result

    save_config(config, context.config_path)
    print(f"Settings saved to {context.config_path}.")
    return result


def _print_intro(config_path: Path, walkthrough: bool, found: bool) -> None:
    """Greet the user for the walkthrough, or name the file for the menu, and warn about environment variables."""
    if walkthrough:
        print("Welcome to ostra. This setup writes your settings to the config file.")
        print("Press Ctrl-C at any time to stop without saving.\n")
        if found:
            print(f"Found settings at {config_path}. Press Enter to keep a current value.\n")
    else:
        print(f"Settings in {config_path}\n")

    overrides = sorted(name for name in os.environ if name.upper().startswith(ENVIRONMENT_PREFIX))
    if overrides:
        print(f"These environment variables override the saved settings: {', '.join(overrides)}\n")


def _read_existing(prompter: Prompter, config_path: Path) -> AppConfig | None:
    """Return the settings of the config file, or None when there is no file or the user replaces an invalid one.

    Raises:
        SetupCancelled: When the file is not valid and the user does not want to replace it.
    """
    try:
        return read_config_file(config_path)
    except ValueError as error:
        _confirm_replacement(prompter, error)
        return None


def _confirm_replacement(prompter: Prompter, error: ValueError) -> None:
    """Ask before a config file that cannot be read is replaced, so that its settings are not lost by accident.

    Raises:
        SetupCancelled: Without a person to ask, or when the user wants to keep the file.
    """
    print(f"The config file could not be read.\n{error}\n")
    if not prompter.interactive:
        print("Fix or remove the file, then run ostra config again.")
        raise SetupCancelled

    if not prompter.confirm("Start from the defaults and replace the file when you save?", default=False):
        raise SetupCancelled
