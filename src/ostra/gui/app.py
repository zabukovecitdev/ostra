import json
from dataclasses import replace
from pathlib import Path

from nicegui import run, ui

from ostra.config import default_config_path, load_config, save_config
from ostra.connectors.database_facade import DatabaseBackend
from ostra.daemon import STATUS_PATH
from ostra.gui.search import search_chunks
from ostra.gui.theme import BUTTON_PROPS, FIELD_PROPS, apply_theme

DEFAULT_PORT = 8765
RESULT_LIMIT = 10


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
    """Show the search box and the matching chunks."""
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

        with results:
            if not matches:
                ui.label("No matching chunks.")
            for match in matches:
                with ui.card().classes("w-full").props(FIELD_PROPS).props("flat"):
                    with ui.row().classes("items-center gap-3"):
                        ui.label(f"{round((1 - match.distance) * 100)}%").classes("ostra-score")
                        ui.label(match.file_path).classes("ostra-muted")
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


def _read_status() -> str:
    """Describe the last background indexing run.

    Returns:
        A one-line text, or a hint when the daemon has not run yet.
    """
    status_path = Path(STATUS_PATH)
    if not status_path.exists():
        return "The background indexer has not run yet. Start it with: ostra-daemon install"

    status = json.loads(status_path.read_text())
    if status["error"]:
        return f"Last run {status['finished_at']} failed: {status['error']}"

    return f"Last run {status['finished_at']} processed {status['files_processed']} files."


@ui.page("/config")
def config_page() -> None:
    """Show the settings form and the state of the background indexer."""
    _header("/config")
    config = load_config()

    def save() -> None:
        directories = [Path(line.strip()).expanduser() for line in directories_input.value.splitlines() if line.strip()]
        missing = [str(directory) for directory in directories if not directory.is_dir()]
        if not directories:
            ui.notify("Add at least one folder.", type="negative")
            return

        if overlap_input.value >= chunk_size_input.value:
            ui.notify("Overlap must be smaller than the chunk size.", type="negative")
            return

        save_config(
            replace(
                config,
                database_path=Path(database_path_input.value).expanduser(),
                directories=directories,
                database_backend=DatabaseBackend(backend_input.value),
                chunk_size_tokens=int(chunk_size_input.value),
                chunk_overlap_tokens=int(overlap_input.value),
            )
        )
        ui.notify("Saved. The background indexer uses the new settings on its next run.", type="positive")
        if missing:
            ui.notify(f"Folders not found: {', '.join(missing)}", type="warning")

    with ui.column().classes("ostra-page"):
        ui.label(f"Config file: {default_config_path()}").classes("ostra-muted")
        ui.label(_read_status()).classes("ostra-muted")
        directories_input = (
            ui.textarea("Folders to index (one per line)", value="\n".join(str(path) for path in config.directories))
            .classes("w-full")
            .props(FIELD_PROPS)
        )
        database_path_input = (
            ui.input("Database path", value=str(config.database_path)).classes("w-full").props(FIELD_PROPS)
        )
        backend_input = ui.select(
            [backend.value for backend in DatabaseBackend], value=config.database_backend.value, label="Backend"
        ).props(FIELD_PROPS)
        chunk_size_input = ui.number("Chunk size (tokens)", value=config.chunk_size_tokens, min=1, precision=0).props(
            FIELD_PROPS
        )
        overlap_input = ui.number(
            "Chunk overlap (tokens)", value=config.chunk_overlap_tokens, min=0, precision=0
        ).props(FIELD_PROPS)
        ui.button("Save", on_click=save).props(BUTTON_PROPS)


if __name__ in {"__main__", "__mp_main__"}:
    main()
