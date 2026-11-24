from collections.abc import Callable, Sequence
from pathlib import Path
from urllib.parse import urlparse

from ostra.cli.config.draft import SetupContext, SetupDraft
from ostra.cli.config.prompter import Choice
from ostra.constants import DEFAULT_CHUNK_OVERLAP_TOKENS
from ostra.enums.embedding_provider_name import EmbeddingProviderName
from ostra.models.embedding_model_registry import EMBEDDING_MODELS, get_model_spec
from ostra.models.embedding_model_spec import EmbeddingModelSpec
from ostra.models.ollama_settings import DEFAULT_OLLAMA_BASE_URL, OllamaSettings
from ostra.services.setup.ollama_check import pull_command

SetupStep = Callable[[SetupContext], None]

# What the registry does not say about each model: its languages and whether it needs setup.
MODEL_NOTES = {
    "potion-multilingual-128m": "100+ languages, no setup needed",
    "mxbai-embed-large": "English",
    "bge-m3": "100+ languages",
}
LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1"}

RETRY = "retry"
CHANGE_ADDRESS = "change"
CONTINUE = "continue"
CONNECTION_CHOICES = (
    Choice("Check again", RETRY),
    Choice("Change the address", CHANGE_ADDRESS),
    Choice("Save the settings anyway", CONTINUE),
)

# The answers of the save step.
SAVE = "save"
BACK_TO_MENU = "menu"
INDEX_NOW = "index_now"
INDEX_LATER = "index_later"
SAVE_CHOICES = (Choice("Save", SAVE), Choice("Back to the menu", BACK_TO_MENU))
INDEX_CHOICES = (
    Choice("Save and index now", INDEX_NOW),
    Choice("Save, index later", INDEX_LATER),
    Choice("Back to the menu", BACK_TO_MENU),
)


def choose_model(context: SetupContext) -> None:
    """Ask which embedding model to use."""
    choices = [Choice(spec.model_id, spec.model_id, _model_description(spec)) for spec in EMBEDDING_MODELS.values()]
    context.draft.embedding_model = context.prompter.select(
        "Which embedding model should ostra use?", choices, default=context.draft.embedding_model
    )


def configure_connection(context: SetupContext) -> None:
    """Ask for the Ollama server address and check that the server has the model. Skip models without Ollama."""
    if _model(context.draft).provider is not EmbeddingProviderName.OLLAMA:
        return

    action = CHANGE_ADDRESS
    while action != CONTINUE:
        if action == CHANGE_ADDRESS:
            _ask_ollama_address(context)
        action = _check_connection(context)


def choose_folders(context: SetupContext) -> None:
    """Ask which folders to index and whether to include their subfolders."""
    draft, prompter = context.draft, context.prompter
    if draft.directories:
        print("Configured folders:\n" + "\n".join(f"  {folder}" for folder in draft.directories))
        if not prompter.confirm("Keep these folders?", default=True):
            draft.directories = []

    add_given_folders(context)
    _add_folders(context)
    if not draft.directories:
        print("No folders yet. Run ostra config again or pass folders to ostra index.")

    draft.recursive = prompter.confirm("Index the files in subfolders too?", default=draft.recursive)


def choose_indexing_settings(context: SetupContext) -> None:
    """Ask whether to keep the chunk settings of the model or to set them."""
    draft, prompter = context.draft, context.prompter
    spec = _model(draft)
    uses_defaults = draft.chunk_size_tokens is None and draft.chunk_overlap_tokens == DEFAULT_CHUNK_OVERLAP_TOKENS
    question = (
        f"Use the default chunk settings ({spec.default_chunk_size_tokens} tokens per chunk, "
        f"{DEFAULT_CHUNK_OVERLAP_TOKENS} tokens of overlap)?"
    )
    if prompter.confirm(question, default=uses_defaults):
        draft.chunk_size_tokens = None
        draft.chunk_overlap_tokens = DEFAULT_CHUNK_OVERLAP_TOKENS
        return

    largest = _largest_chunk(spec)
    size = prompter.integer(
        f"Tokens per chunk (1 to {largest}):",
        default=min(draft.chunk_size_tokens or spec.default_chunk_size_tokens, largest),
        validate=_range_check(1, largest),
    )
    draft.chunk_size_tokens = size
    draft.chunk_overlap_tokens = prompter.integer(
        f"Tokens shared by consecutive chunks (0 to {size - 1}):",
        default=min(draft.chunk_overlap_tokens, size - 1),
        validate=_range_check(0, size - 1),
    )


def choose_database(context: SetupContext) -> None:
    """Offer to change the location of the database."""
    draft, prompter = context.draft, context.prompter
    if not prompter.confirm(f"Keep the database at {draft.database_path}?", default=True):
        answer = prompter.path("Database file:", default=str(draft.database_path), validate=_check_database_path)
        draft.database_path = _absolute(answer)


WALKTHROUGH_STEPS: tuple[SetupStep, ...] = (
    choose_model,
    configure_connection,
    choose_folders,
    choose_indexing_settings,
    choose_database,
)


def fit_chunk_size(context: SetupContext) -> None:
    """After a model change, offer the model default when the chunk size does not fit the new model."""
    draft, prompter = context.draft, context.prompter
    spec = _model(draft)
    largest = _largest_chunk(spec)
    if draft.chunk_size_tokens is None or draft.chunk_size_tokens <= largest:
        return

    print(
        f"The chunk size of {draft.chunk_size_tokens} tokens does not fit {spec.model_id}, which takes up to {largest}."
    )
    if prompter.confirm(f"Use the model default of {spec.default_chunk_size_tokens} tokens per chunk?", default=True):
        draft.chunk_size_tokens = None
        return

    choose_indexing_settings(context)


def add_given_folders(context: SetupContext) -> None:
    """Add the folders of the --directory option that the configured folders do not have yet."""
    _add_unique(context.draft.directories, context.given_directories)


def save_step(context: SetupContext) -> bool:
    """Show the answers and ask what to do with them.

    Without a person to answer, the settings are saved and not indexed.

    Returns:
        True when the settings are to be saved, False when the user goes back to the menu. The context tells
        whether to index right after saving.
    """
    print(_summary(context.draft, context.index_changes()))
    if not context.prompter.interactive:
        context.index_now = False
        return True

    if context.index_changes() and context.draft.directories:
        choice = context.prompter.select("What now?", INDEX_CHOICES, default=INDEX_NOW)
    else:
        choice = context.prompter.select(f"Save these settings to {context.config_path}?", SAVE_CHOICES, default=SAVE)
    context.index_now = choice == INDEX_NOW
    return choice != BACK_TO_MENU


def run_steps(steps: Sequence[SetupStep], context: SetupContext) -> None:
    """Run the steps in order."""
    for step in steps:
        step(context)


def describe_folders(draft: SetupDraft) -> str:
    """Name the folders, comma separated, or none."""
    return ", ".join(str(folder) for folder in draft.directories) or "none"


def describe_chunks(draft: SetupDraft) -> str:
    """Show the chunk size and the overlap, with the model default for the size when none is set."""
    if draft.chunk_size_tokens is None:
        return f"model default ({_model(draft).default_chunk_size_tokens} / {draft.chunk_overlap_tokens})"

    return f"{draft.chunk_size_tokens} / {draft.chunk_overlap_tokens}"


def _model(draft: SetupDraft) -> EmbeddingModelSpec:
    """Return the spec of the chosen model."""
    return get_model_spec(draft.embedding_model)


def _largest_chunk(spec: EmbeddingModelSpec) -> int:
    """Return the most tokens that a chunk may have: the model input without the special tokens."""
    return spec.max_tokens - spec.special_tokens


def _model_description(spec: EmbeddingModelSpec) -> str:
    """Describe where a model runs and which languages it knows."""
    where = "needs an Ollama server" if spec.provider is EmbeddingProviderName.OLLAMA else "runs in ostra on the CPU"
    notes = MODEL_NOTES.get(spec.model_id)
    return f"{where}, {notes}" if notes else where


def _ask_ollama_address(context: SetupContext) -> None:
    """Ask for the address of the Ollama server and warn when the server is on another machine."""
    draft = context.draft
    answer = context.prompter.text(
        "Ollama server address:", default=draft.ollama_base_url or OllamaSettings().base_url, validate=_check_url
    )
    # Save the built-in local address as no address, so that OLLAMA_BASE_URL can still choose a server later. Save
    # any other address, also one from OLLAMA_BASE_URL, so that a shell without the variable uses it too.
    draft.ollama_base_url = None if answer == DEFAULT_OLLAMA_BASE_URL else answer
    if urlparse(answer).hostname not in LOCAL_HOSTS:
        print("The text of your files will be sent to this server to be embedded.")


def _check_connection(context: SetupContext) -> str:
    """Check the Ollama server, and ask what to do when it does not answer or lacks the model.

    Returns:
        CONTINUE when the server is ready or the user goes on anyway, else RETRY or CHANGE_ADDRESS.
    """
    spec = _model(context.draft)
    address = context.draft.ollama_base_url or OllamaSettings().base_url
    status = context.services.check_ollama(address, spec.provider_model)
    if status.model_present:
        print(f"Connected. The server has {spec.provider_model}.")
        return CONTINUE

    if status.reachable:
        print(f"The server does not have {spec.provider_model}. Download it on the server with:")
        print(f"  {pull_command(spec.provider_model)}")
    else:
        print(f"Could not reach the Ollama server at {address}. {status.error}".rstrip())

    if not context.prompter.interactive:
        return CONTINUE

    return context.prompter.select("What next?", CONNECTION_CHOICES, default=RETRY)


def _add_folders(context: SetupContext) -> None:
    """Ask for folders until the user leaves the answer empty."""
    while answer := context.prompter.path("Folder to index (leave empty to finish):", validate=_check_folder):
        _add_unique(context.draft.directories, [_absolute(answer)])


def _add_unique(folders: list[Path], new_folders: Sequence[Path]) -> None:
    """Add the new folders that the list does not have yet, as absolute paths."""
    for folder in (_absolute(str(new_folder)) for new_folder in new_folders):
        if folder not in folders:
            folders.append(folder)


def _summary(draft: SetupDraft, changes: tuple[str, ...]) -> str:
    """Describe the answers, one setting per line, and the changes that need indexing."""
    rows: list[tuple[str, object]] = [
        ("Model", draft.embedding_model),
        ("Folders", describe_folders(draft)),
        ("Subfolders", "yes" if draft.recursive else "no"),
        ("Chunks", describe_chunks(draft)),
        ("Database", draft.database_path),
    ]
    if _model(draft).provider is EmbeddingProviderName.OLLAMA:
        rows.insert(1, ("Ollama server", draft.ollama_base_url or OllamaSettings().base_url))
    if changes:
        rows.append(("Needs ostra index", ", ".join(changes)))

    return "\nSummary\n" + "\n".join(f"  {name:<20}{value}" for name, value in rows)


def _absolute(answer: str) -> Path:
    """Turn a path answer into an absolute path, with a leading ~ expanded."""
    return Path(answer).expanduser().absolute()


def _check_folder(answer: str) -> str | None:
    """Accept an empty answer or an existing folder."""
    if not answer:
        return None

    folder = Path(answer).expanduser()
    if not folder.exists():
        return "The folder does not exist."

    return None if folder.is_dir() else "This is a file, not a folder."


def _check_database_path(answer: str) -> str | None:
    """Accept a path that is not empty and not a folder."""
    if not answer:
        return "Enter a file path."

    return "This is a folder. Enter a file path." if Path(answer).expanduser().is_dir() else None


def _check_url(answer: str) -> str | None:
    """Accept an http or https address with a host."""
    parsed = urlparse(answer)
    if parsed.scheme in {"http", "https"} and parsed.hostname:
        return None

    return "Enter an address such as http://localhost:11434."


def _range_check(lowest: int, highest: int) -> Callable[[int], str | None]:
    """Build a validator that accepts the numbers from lowest to highest."""

    def check(value: int) -> str | None:
        return None if lowest <= value <= highest else f"Enter a number from {lowest} to {highest}."

    return check
