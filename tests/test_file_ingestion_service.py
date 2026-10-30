from pathlib import Path

import pytest

from personal_rag.connectors.sqlite_connector import SqliteConnector, SqliteSettings
from personal_rag.hashing import hash_file, hash_line
from personal_rag.repositories.file_chunk_repository import FileChunkRepository
from personal_rag.repositories.file_repository import FileRepository
from personal_rag.services.database_service import DatabaseService
from personal_rag.services.file_ingestion_service import FileIngestionService
from personal_rag.services.file_service import FileService


@pytest.fixture
def connector(tmp_path: Path):
    """Provide an initialized SQLite connector and close it after the test."""
    sqlite_connector = SqliteConnector(SqliteSettings(path=tmp_path / "personal_rag.db"))
    DatabaseService(connector=sqlite_connector).initialize()
    yield sqlite_connector
    sqlite_connector.close()


def make_service(connector: SqliteConnector, directory: Path) -> FileIngestionService:
    """Build an ingestion service that reads one directory and stores results through the connector."""
    return FileIngestionService(
        file_service=FileService(directories=[directory]),
        file_repository=FileRepository(connector=connector),
        file_chunk_repository=FileChunkRepository(connector=connector),
    )


def stored_chunks(connector: SqliteConnector, file_path: Path) -> set[str]:
    """Return the stored line digests of one file."""
    rows = connector.connect().execute("SELECT digest FROM file_chunks WHERE file_path = ?", (str(file_path),))
    return {row[0] for row in rows.fetchall()}


def test_process_files__new_file__prints_each_line(connector, tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    """Print each line of a new file without its line ending."""
    (tmp_path / "a.txt").write_text("alpha\nbeta\n")

    make_service(connector, tmp_path).process_files()

    assert capsys.readouterr().out == "alpha\nbeta\n"


def test_process_files__new_file__saves_file_digest(connector, tmp_path: Path):
    """Save the digest of the whole file in the files table."""
    file_path = tmp_path / "a.txt"
    file_path.write_text("alpha\nbeta\n")

    make_service(connector, tmp_path).process_files()
    rows = connector.connect().execute("SELECT file_path, digest FROM files").fetchall()

    assert rows == [(str(file_path), hash_file(file_path))]


def test_process_files__new_file__saves_digest_of_each_line(connector, tmp_path: Path):
    """Save the digest of each line in the file_chunks table."""
    file_path = tmp_path / "a.txt"
    file_path.write_text("alpha\nbeta\n")

    make_service(connector, tmp_path).process_files()

    assert stored_chunks(connector, file_path) == {hash_line("alpha\n"), hash_line("beta\n")}


def test_process_files__identical_lines__prints_the_line_once(connector, tmp_path: Path, capsys):
    """Print a repeated line only once within one file."""
    (tmp_path / "a.txt").write_text("same\nsame\n")

    make_service(connector, tmp_path).process_files()

    assert capsys.readouterr().out == "same\n"


def test_process_files__second_run_file_unchanged__prints_nothing(connector, tmp_path: Path, capsys):
    """Print no lines on a second run when no line changed."""
    (tmp_path / "a.txt").write_text("alpha\nbeta\n")
    service = make_service(connector, tmp_path)
    service.process_files()
    capsys.readouterr()

    service.process_files()

    assert capsys.readouterr().out == ""


def test_process_files__run_stopped_after_first_line__resumes_at_next_line(connector, tmp_path: Path, capsys):
    """Skip the lines that a stopped run already saved and process the remaining lines."""
    file_path = tmp_path / "a.txt"
    file_path.write_text("alpha\nbeta\ngamma\n")
    FileChunkRepository(connector=connector).save_chunk(str(file_path), hash_line("alpha\n"))

    make_service(connector, tmp_path).process_files()

    assert capsys.readouterr().out == "beta\ngamma\n"


def test_process_files__line_changed__prints_only_the_changed_line(connector, tmp_path: Path, capsys):
    """Print only the line whose content changed since the last run."""
    file_path = tmp_path / "a.txt"
    file_path.write_text("alpha\nbeta\n")
    service = make_service(connector, tmp_path)
    service.process_files()
    capsys.readouterr()
    file_path.write_text("alpha\nbeta changed\n")

    service.process_files()

    assert capsys.readouterr().out == "beta changed\n"


def test_process_files__line_inserted_above__prints_only_the_new_line(connector, tmp_path: Path, capsys):
    """Print only the new line when a line is inserted above existing lines."""
    file_path = tmp_path / "a.txt"
    file_path.write_text("alpha\nbeta\n")
    service = make_service(connector, tmp_path)
    service.process_files()
    capsys.readouterr()
    file_path.write_text("new\nalpha\nbeta\n")

    service.process_files()

    assert capsys.readouterr().out == "new\n"


def test_process_files__file_got_shorter__deletes_digests_of_removed_lines(connector, tmp_path: Path):
    """Delete the stored digests of lines that the file no longer has."""
    file_path = tmp_path / "a.txt"
    file_path.write_text("alpha\nbeta\ngamma\n")
    service = make_service(connector, tmp_path)
    service.process_files()
    file_path.write_text("alpha\n")

    service.process_files()

    assert stored_chunks(connector, file_path) == {hash_line("alpha\n")}


def test_process_files__file_changed__updates_stored_file_digest(connector, tmp_path: Path):
    """Replace the stored file digest when the file content changed."""
    file_path = tmp_path / "a.txt"
    file_path.write_text("alpha\n")
    service = make_service(connector, tmp_path)
    service.process_files()
    file_path.write_text("alpha\nbeta\n")

    service.process_files()
    rows = connector.connect().execute("SELECT digest FROM files").fetchall()

    assert rows == [(hash_file(file_path),)]
