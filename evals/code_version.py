"""Describe the code and the machine that ran an eval, so that every output.json says which version produced it.

The git commit is the version of the code. A commit alone is not enough when the code had changes that were not
committed yet: two runs on the same commit can then run different code. So the description also says whether there
were uncommitted changes, and gives a hash of them, so that two such runs show whether their changes were the same.

Only the files that can change the results count as changes (RESULT_PATHS). Editing the config.json of a run or
rewriting its output.json does not count: the settings of a run are recorded in output.json on their own.
"""

import hashlib
import platform
import subprocess
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

REPOSITORY_DIRECTORY = Path(__file__).parent.parent
# The files whose changes can change the results of a run.
RESULT_PATHS = ("src", "evals/runner.py", "evals/code_version.py", "pyproject.toml", "uv.lock")
# The libraries that embed, tokenize and store, whose upgrades can change the results.
PACKAGES = ("fastembed", "ollama", "sqlite-vec", "tokenizers")
GIT_TIMEOUT_SECONDS = 10
# The git fields when the code is not in a git repository or git is not installed.
NO_GIT = {
    "commit": None,
    "branch": None,
    "uncommitted_changes": None,
    "uncommitted_changes_hash": None,
}


def code_version(repository: Path = REPOSITORY_DIRECTORY) -> dict:
    """Describe the code, the Python and the libraries that run the evals.

    Args:
        repository: The folder of the git repository.

    Returns:
        The ostra version, the git state of the code, the Python version, the platform and the library versions.
    """
    return {
        "ostra_version": _package_version("ostra"),
        **git_state(repository),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {name: _package_version(name) for name in PACKAGES},
    }


def git_state(repository: Path) -> dict:
    """Describe the commit, the branch and the uncommitted changes of the code.

    Args:
        repository: The folder of the git repository.

    Returns:
        The full commit hash, the branch name ("HEAD" when no branch is checked out), whether files of RESULT_PATHS
        differ from the commit, and a SHA-256 hash of those differences (None when there are none). Every field is
        None when the folder is not in a git repository or git is not installed.
    """
    commit = _git(repository, "rev-parse", "HEAD")
    if commit is None:
        return dict(NO_GIT)

    changes_hash = uncommitted_changes_hash(repository)
    return {
        "commit": commit,
        "branch": _git(repository, "rev-parse", "--abbrev-ref", "HEAD"),
        "uncommitted_changes": changes_hash is not None,
        "uncommitted_changes_hash": changes_hash,
    }


def uncommitted_changes_hash(repository: Path) -> str | None:
    """Hash the changes to the files of RESULT_PATHS that are not committed, including new files.

    Args:
        repository: The folder of the git repository.

    Returns:
        The SHA-256 hex digest of the diff against the commit and of the name and bytes of each new file, or None
        when there are no changes.
    """
    diff = _git(repository, "diff", "--no-ext-diff", "--no-color", "HEAD", "--", *RESULT_PATHS) or ""
    new_files = _git(repository, "ls-files", "--others", "--exclude-standard", "--", *RESULT_PATHS) or ""
    if not diff and not new_files:
        return None

    digest = hashlib.sha256(diff.encode("utf-8") + b"\0")
    for name in sorted(new_files.splitlines()):
        digest.update(name.encode("utf-8") + b"\0")
        digest.update((repository / name).read_bytes() + b"\0")

    return digest.hexdigest()


def describe(system: dict) -> str:
    """Write the code version of a run as one line for the report.

    Args:
        system: The system block of output.json.

    Returns:
        A line such as "Code: f57e2af1c3d4 on master" or "Code: f57e2af1c3d4 on master, uncommitted changes 3a9c...".
    """
    commit = system.get("commit")
    if commit is None:
        return "Code: unknown commit (not a git repository or git not installed)"

    line = f"Code: {commit[:12]} on {system['branch']}"
    if system["uncommitted_changes"]:
        line += f", uncommitted changes {system['uncommitted_changes_hash'][:12]}"

    return line


def _git(repository: Path, *arguments: str) -> str | None:
    """Run a git command in the repository and return its output, or None when git fails or is not installed."""
    try:
        completed = subprocess.run(
            ["git", *arguments],
            cwd=repository,
            capture_output=True,
            text=True,
            timeout=GIT_TIMEOUT_SECONDS,
            check=False,
        )
    except OSError, subprocess.TimeoutExpired:
        return None

    if completed.returncode != 0:
        return None

    return completed.stdout.strip()


def _package_version(name: str) -> str | None:
    """Return the installed version of a package, or None when it is not installed."""
    try:
        return version(name)
    except PackageNotFoundError:
        return None
