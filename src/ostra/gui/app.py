import json
from dataclasses import replace
from functools import partial
from pathlib import Path

from nicegui import run, ui

from ostra.config import default_config_path, load_config, save_config
from ostra.daemon import STATUS_PATH, index_once
from ostra.gui.folder_picker import FolderPicker
from ostra.gui.ollama_status import INSTALL_URL, OllamaState, check_ollama, is_valid_server_url, model_url
from ostra.gui.search import search_chunks
from ostra.gui.theme import BUTTON_PROPS, FIELD_PROPS, apply_theme
from ostra.models.factory import ollama_settings
from ostra.models.provider import EmbeddingProviderName
from ostra.models.registry import EMBEDDING_MODELS, get_model_spec
from ostra.store.database_backend import DatabaseBackend

DEFAULT_PORT = 8765
RESULT_LIMIT = 10

# One run at a time for all browser tabs of this process.
_indexing: dict = {"running": False, "current": None, "done": 0}


def main() -> None:
    """Start the web interface on the local machine and open it in the browser."""
    ui.run(title="ostra", port=DEFAULT_PORT, reload=False, host="127.0.0.1", favicon="🔎", dark=None)


def _header(active: str) -> None:
    """Add the title bar with links to both pages.

    Args:
        active: The path of the current page.
    """
    apply_theme()
    with ui.header().classes("items-center gap-2"):
        ui.label("ostra").classes("ostra-brand q-mr-md")
        for label, path in (("Search", "/"), ("Config", "/config")):
            ui.link(label, path).classes("ostra-nav" + (" ostra-nav--active" if active == path else ""))


@ui.page("/")
def search_page() -> None:
    """Show the search box and the matching chunks. Show the welcome page when no folder is configured."""
    if not load_config().directories:
        ui.navigate.to("/welcome")
        return

    _header("/")
    state = {"busy": False}

    async def run_search() -> None:
        query = query_input.value.strip()
        if not query or state["busy"]:
            return

        state["busy"] = True
        spinner.set_visibility(True)
        results.clear()
        try:
            matches = await run.io_bound(
                search_chunks,
                load_config(),
                query,
                RESULT_LIMIT,
                max_distance_input.value or None,
            )
        except Exception as error:
            ui.notify(f"Search failed: {error}", type="negative")
            return
        finally:
            state["busy"] = False
            spinner.set_visibility(False)

        if matches is None:
            return

        with results:
            if not matches:
                ui.label("No matching chunks.")
            for match in matches:
                with ui.card().classes("w-full").props(FIELD_PROPS).props("flat"):
                    with ui.row().classes("items-center gap-3"):
                        ui.label(f"{round((1 - match.distance) * 100)}%").classes("ostra-score")
                        ui.label(match.document_id).classes("ostra-muted")
                    ui.label(match.text).classes("ostra-text")

    with ui.column().classes("ostra-page"):
        with ui.row().classes("w-full items-center"):
            query_input = (
                ui.input(placeholder="Search your documents")
                .classes("grow")
                .props(FIELD_PROPS)
                .on("keydown.enter", run_search)
            )
            ui.button("Search", on_click=run_search).props(BUTTON_PROPS)
        max_distance_input = (
            ui.number("Max distance (optional)", min=0, max=2, step=0.05, format="%.2f")
            .classes("w-64")
            .props(FIELD_PROPS)
        )
        spinner = ui.spinner()
        spinner.set_visibility(False)
        results = ui.column().classes("w-full gap-2")


@ui.page("/welcome")
def welcome_page() -> None:
    """Show one button that adds folders to index, and the state of the Ollama model."""
    _header("")

    async def add_folders() -> None:
        """Let the user choose a folder, save it in the config file, and open the config page."""
        chosen = await FolderPicker(None)
        if chosen is None:
            return

        config = load_config()
        if chosen not in config.directories:
            save_config(replace(config, directories=[*config.directories, chosen]))
        ui.navigate.to("/config")

    async def save_url_and_check(address: str) -> None:
        """Save the server address that the user typed, then check the server again.

        Args:
            address: The text of the address box.
        """
        url = address.strip()
        if not is_valid_server_url(url):
            ui.notify("Enter an address such as http://192.168.1.20:11434", type="negative")
            return

        save_config(replace(load_config(), ollama_base_url=url))
        await show_model_status()

    async def show_model_status() -> None:
        """Check Ollama and show the result below the button."""
        config = load_config()
        settings = ollama_settings(config)
        status_box.clear()
        with status_box:
            ui.label("Checking Ollama...").classes("ostra-muted")

        ollama_model = get_model_spec(config.embedding_model).ollama_model
        status = await run.io_bound(check_ollama, settings, ollama_model)
        if status is None:
            return

        status_box.clear()
        with status_box:
            if status.state == OllamaState.READY:
                _status_line(f"Model ready: {status.model}", ok=True)
                ui.label(f"Server: {settings.base_url}").classes("ostra-muted")
                return

            if status.state == OllamaState.MODEL_MISSING:
                _status_line(f"Model not installed: {status.model}", ok=False)
                ui.label(f"Run on the server: ollama pull {status.model}").classes("ostra-muted")
                ui.link("Model instructions", model_url(ollama_model), new_tab=True).classes("ostra-link")
            else:
                _status_line(f"Ollama is not reachable at {settings.base_url}", ok=False)
                ui.link("Install or start Ollama", INSTALL_URL, new_tab=True).classes("ostra-link")
            ui.label("Ollama runs on another machine? Enter its address:").classes("ostra-muted q-mt-md")
            with ui.row().classes("items-center no-wrap"):
                url_input = ui.input(value=settings.base_url).props(FIELD_PROPS).classes("w-64")
                url_input.on("keydown.enter", lambda: save_url_and_check(url_input.value))
                ui.button("Save and check", on_click=lambda: save_url_and_check(url_input.value)).props(BUTTON_PROPS)

    with ui.column().classes("ostra-page ostra-welcome"):
        ui.button("Add folders to index", icon="create_new_folder", on_click=add_folders).props(
            "unelevated no-caps stack"
        ).classes("ostra-welcome-button")
        status_box = ui.column().classes("items-center gap-1")
    ui.timer(0.1, show_model_status, once=True)


def _status_line(text: str, ok: bool) -> None:
    """Add a line with a colored dot.

    Args:
        text: The text of the line.
        ok: True for a green dot and False for a red dot.
    """
    with ui.row().classes("items-center no-wrap"):
        ui.element("span").classes("ostra-dot " + ("ostra-dot--ok" if ok else "ostra-dot--bad"))
        ui.label(text)


def _read_status_error() -> str | None:
    """Return the error of the last background indexing run.

    Returns:
        The error text, or None when the run succeeded or has not run.
    """
    status_path = Path(STATUS_PATH)
    if not status_path.exists():
        return None

    return json.loads(status_path.read_text()).get("error")


def _read_status() -> str:
    """Describe the last background indexing run.

    Returns:
        A one-line text, or a hint when the daemon has not run yet.
    """
    status_path = Path(STATUS_PATH)
    if not status_path.exists():
        return "The background indexer has not run yet. Start it with: ostra-daemon install"

    status = json.loads(status_path.read_text())
    if status.get("message"):
        return status["message"]

    if status["error"]:
        return f"Last run {status['finished_at']} failed: {status['error']}"

    return f"Last run {status['finished_at']} processed {status['files_processed']} files."


@ui.page("/config")
def config_page() -> None:
    """Show the settings form and the state of the background indexer."""
    _header("/config")
    config = load_config()
    folders = list(config.directories)

    @ui.refreshable
    def folder_list() -> None:
        """Show each configured folder with a remove button."""
        if not folders:
            ui.label("No folders yet. Add a folder to start indexing.").classes("ostra-muted")
        for folder in folders:
            with ui.row().classes("w-full items-center no-wrap"):
                ui.label(str(folder)).classes("grow ostra-text")
                ui.button(icon="close", on_click=lambda folder=folder: remove_folder(folder)).props("flat round dense")

    def remove_folder(folder: Path) -> None:
        """Remove a folder from the list on the page. The change is saved with the Save button.

        Args:
            folder: The folder to remove.
        """
        folders.remove(folder)
        folder_list.refresh()

    async def add_folder() -> None:
        """Open the folder picker and add the chosen folder to the list on the page."""
        chosen = await FolderPicker(folders[-1] if folders else None)
        if chosen is None:
            return

        if chosen in folders:
            ui.notify("This folder is already in the list.", type="warning")
            return

        folders.append(chosen)
        folder_list.refresh()

    def save() -> None:
        directories = list(folders)
        missing = [str(directory) for directory in directories if not directory.is_dir()]
        if url_input.value.strip() and not is_valid_server_url(url_input.value):
            ui.notify("The Ollama address must start with http:// or https://", type="negative")
            return

        chunk_size = int(chunk_size_input.value)
        if overlap_input.value >= chunk_size:
            ui.notify("Overlap must be smaller than the chunk size.", type="negative")
            return

        save_config(
            replace(
                config,
                database_path=Path(database_path_input.value).expanduser(),
                directories=directories,
                database_backend=DatabaseBackend(backend_input.value),
                embedding_model=model_input.value,
                chunk_size_tokens=None
                if chunk_size == get_model_spec(model_input.value).default_chunk_size_tokens
                else chunk_size,
                chunk_overlap_tokens=int(overlap_input.value),
                embedding_provider=EmbeddingProviderName(provider_input.value),
                ollama_base_url=url_input.value.strip() or None,
            )
        )
        ui.notify("Saved. The background indexer uses the new settings on its next run.", type="positive")
        if missing:
            ui.notify(f"Folders not found: {', '.join(missing)}", type="warning")

    def show_progress() -> None:
        """Show the document that is in progress and the number of finished documents."""
        if _indexing["running"]:
            progress.set_visibility(True)
            progress_label.set_text(f"Indexing {_indexing['current'] or '...'}  ({_indexing['done']} done)")

    async def index_now() -> None:
        """Run one indexing pass with the saved settings and show the result."""
        if _indexing["running"]:
            ui.notify("Indexing is already running.", type="warning")
            return

        _indexing.update(running=True, current=None, done=0)
        progress.set_visibility(True)
        index_button.props("loading")
        index_button.disable()
        try:
            await run.io_bound(
                partial(
                    index_once,
                    on_document_started=lambda document_id: _indexing.update(current=document_id),
                    on_document_processed=lambda _: _indexing.update(done=_indexing["done"] + 1),
                )
            )
        finally:
            _indexing["running"] = False
            progress.set_visibility(False)
            progress_label.set_text("")
            index_button.props(remove="loading")
            index_button.enable()

        status_label.set_text(_read_status())
        failed = bool(_read_status_error())
        ui.notify(status_label.text, type="negative" if failed else "positive")

    with ui.column().classes("ostra-page"):
        ui.label(f"Config file: {default_config_path()}").classes("ostra-muted")
        status_label = ui.label(_read_status()).classes("ostra-muted")
        index_button = ui.button("Index now", icon="sync", on_click=index_now).props(BUTTON_PROPS)
        ui.label("Indexes the saved settings. Save your changes first.").classes("ostra-muted")
        progress = ui.linear_progress(show_value=False).props("indeterminate").classes("w-full")
        progress.set_visibility(False)
        progress_label = ui.label("").classes("ostra-muted")
        ui.timer(0.3, show_progress)
        ui.label("Folders to index").classes("font-semibold")
        with ui.card().classes("w-full").props("flat"):
            folder_list()
            ui.button("Add folder", icon="folder_open", on_click=add_folder).props(BUTTON_PROPS + " flat")
        ui.label("Embedding model").classes("font-semibold q-mt-md")
        provider_input = ui.select(
            [provider.value for provider in EmbeddingProviderName],
            value=config.embedding_provider.value,
            label="Provider",
        ).props(FIELD_PROPS)
        model_input = ui.select(list(EMBEDDING_MODELS), value=config.embedding_model, label="Model").props(FIELD_PROPS)
        url_input = (
            ui.input("Ollama server address", value=config.ollama_base_url or ollama_settings(config).base_url)
            .classes("w-full")
            .props(FIELD_PROPS)
        )
        ui.label("The text of your files is sent to this server. Use a machine that you trust.").classes("ostra-muted")
        database_path_input = (
            ui.input("Database path", value=str(config.database_path)).classes("w-full").props(FIELD_PROPS)
        )
        backend_input = ui.select(
            [backend.value for backend in DatabaseBackend], value=config.database_backend.value, label="Backend"
        ).props(FIELD_PROPS)
        chunk_size_input = ui.number(
            "Chunk size (tokens)",
            value=config.chunk_size_tokens or get_model_spec(config.embedding_model).default_chunk_size_tokens,
            min=1,
            precision=0,
        ).props(FIELD_PROPS)
        overlap_input = ui.number(
            "Chunk overlap (tokens)", value=config.chunk_overlap_tokens, min=0, precision=0
        ).props(FIELD_PROPS)
        ui.button("Save", on_click=save).props(BUTTON_PROPS)


if __name__ in {"__main__", "__mp_main__"}:
    main()
