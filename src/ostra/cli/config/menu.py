from ostra.cli.config.draft import SetupContext, SetupDraft
from ostra.cli.config.prompter import Choice, SetupCancelled
from ostra.cli.config.steps import (
    WALKTHROUGH_STEPS,
    SetupStep,
    choose_database,
    choose_folders,
    choose_indexing_settings,
    choose_model,
    configure_connection,
    describe_chunks,
    describe_folders,
    fit_chunk_size,
    run_steps,
    save_step,
)

# The settings of the menu. Each one is also the shortcut of ostra config, such as ostra config model.
MODEL = "model"
FOLDERS = "folders"
CHUNKS = "chunks"
DATABASE = "database"
SETTING_SHORTCUTS = (MODEL, FOLDERS, CHUNKS, DATABASE)

WALK_THROUGH = "all"
SAVE_AND_EXIT = "save"
EXIT_WITHOUT_SAVING = "exit"
MENU_QUESTION = "What do you want to change?"


def leave_without_saving(context: SetupContext) -> None:
    """Stop the setup without saving anything.

    Raises:
        SetupCancelled: Always.
    """
    raise SetupCancelled


# The steps that each menu value runs. Saving is not a step: run_menu asks the save step.
MENU_ENTRIES: dict[str, tuple[SetupStep, ...]] = {
    MODEL: (choose_model, configure_connection, fit_chunk_size),
    FOLDERS: (choose_folders,),
    CHUNKS: (choose_indexing_settings,),
    DATABASE: (choose_database,),
    WALK_THROUGH: WALKTHROUGH_STEPS,
    SAVE_AND_EXIT: (),
    EXIT_WITHOUT_SAVING: (leave_without_saving,),
}


def run_menu(context: SetupContext) -> None:
    """Ask what to change, run the steps of the answer, and repeat until the user chooses to save.

    Raises:
        SetupCancelled: When the user exits without saving or presses Ctrl-C.
    """
    while True:
        value = context.prompter.select(MENU_QUESTION, menu_choices(context.draft))
        if value == SAVE_AND_EXIT and save_step(context):
            return

        run_steps(MENU_ENTRIES[value], context)


def menu_choices(draft: SetupDraft) -> list[Choice]:
    """Offer each setting with its current value, so that the menu shows the answers so far."""
    folders = describe_folders(draft) + (" (with subfolders)" if draft.recursive else "")
    return [
        Choice(_row("Embedding model", draft.embedding_model), MODEL),
        Choice(_row("Folders", folders), FOLDERS),
        Choice(_row("Chunk settings", describe_chunks(draft)), CHUNKS),
        Choice(_row("Database", str(draft.database_path)), DATABASE),
        Choice("Walk through all settings", WALK_THROUGH),
        Choice("Save and exit", SAVE_AND_EXIT),
        Choice("Exit without saving", EXIT_WITHOUT_SAVING),
    ]


def _row(label: str, value: str) -> str:
    """Show a label and its value in one column, so that the values line up."""
    return f"{label:<20}{value}"
