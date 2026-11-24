import shutil
import subprocess
from pathlib import Path

import pytest

from evals.code_version import code_version, describe, git_state

needs_git = pytest.mark.skipif(shutil.which("git") is None, reason="git is not installed")


def git(repository: Path, *arguments: str) -> str:
    """Run a git command in the repository with a fixed author and return its output."""
    command = ["git", "-c", "user.name=Test", "-c", "user.email=test@example.com", *arguments]
    return subprocess.run(command, cwd=repository, capture_output=True, text=True, check=True).stdout.strip()


@pytest.fixture
def repository(tmp_path: Path) -> Path:
    """Provide a git repository on branch main with one commit of a source file and a run config."""
    git(tmp_path, "init", "--quiet", "--initial-branch=main")
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "search.py").write_text("LIMIT = 5\n")
    (tmp_path / "evals" / "runs" / "r1").mkdir(parents=True)
    (tmp_path / "evals" / "runs" / "r1" / "config.json").write_text("{}\n")
    git(tmp_path, "add", ".")
    git(tmp_path, "commit", "--quiet", "-m", "first")
    return tmp_path


@needs_git
def test_git_state_of_a_clean_checkout(repository: Path):
    state = git_state(repository)

    assert state == {
        "commit": git(repository, "rev-parse", "HEAD"),
        "branch": "main",
        "uncommitted_changes": False,
        "uncommitted_changes_hash": None,
    }


@needs_git
def test_git_state_reports_a_changed_source_file(repository: Path):
    (repository / "src" / "search.py").write_text("LIMIT = 10\n")
    first = git_state(repository)
    (repository / "src" / "search.py").write_text("LIMIT = 20\n")
    second = git_state(repository)

    assert first["uncommitted_changes"] is True
    assert first["uncommitted_changes_hash"] is not None
    assert second["uncommitted_changes_hash"] != first["uncommitted_changes_hash"]


@needs_git
def test_git_state_gives_the_same_hash_for_the_same_changes(repository: Path):
    (repository / "src" / "search.py").write_text("LIMIT = 10\n")

    assert git_state(repository)["uncommitted_changes_hash"] == git_state(repository)["uncommitted_changes_hash"]


@needs_git
def test_git_state_reports_a_new_source_file(repository: Path):
    (repository / "src" / "rerank.py").write_text("WEIGHT = 1\n")

    assert git_state(repository)["uncommitted_changes"] is True


@needs_git
def test_git_state_ignores_changes_to_run_folders(repository: Path):
    (repository / "evals" / "runs" / "r1" / "config.json").write_text('{"search_limit": 10}\n')
    (repository / "evals" / "runs" / "r1" / "output.json").write_text("{}\n")

    state = git_state(repository)

    assert state["uncommitted_changes"] is False
    assert state["uncommitted_changes_hash"] is None


@needs_git
def test_git_state_names_a_detached_checkout_head(repository: Path):
    git(repository, "checkout", "--quiet", "--detach")

    assert git_state(repository)["branch"] == "HEAD"


@needs_git
def test_git_state_outside_a_repository(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path.parent))

    assert set(git_state(tmp_path).values()) == {None}


def test_git_state_without_git(repository: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("PATH", "")

    assert set(git_state(repository).values()) == {None}


def test_code_version_lists_the_versions_of_ostra_python_and_the_libraries(tmp_path: Path):
    system = code_version(tmp_path)

    assert system["ostra_version"]
    assert system["python"]
    assert system["packages"]["sqlite-vec"]


def test_describe_a_clean_checkout():
    system = {"commit": "f57e2af1c3d4e5f6", "branch": "main", "uncommitted_changes": False}

    assert describe(system) == "Code: f57e2af1c3d4 on main"


def test_describe_uncommitted_changes():
    system = {
        "commit": "f57e2af1c3d4e5f6",
        "branch": "main",
        "uncommitted_changes": True,
        "uncommitted_changes_hash": "3a9c0b1d2e3f4a5b6c",
    }

    assert describe(system) == "Code: f57e2af1c3d4 on main, uncommitted changes 3a9c0b1d2e3f"


def test_describe_without_git():
    assert describe({"commit": None}).startswith("Code: unknown commit")
