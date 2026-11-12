from pathlib import Path


def list_subfolders(folder: Path) -> list[Path]:
    """Return the visible subfolders of a folder, sorted by name.

    Args:
        folder: The folder to read.

    Returns:
        The subfolders whose names do not start with a dot. An empty list when the folder cannot be read.
    """
    try:
        children = list(folder.iterdir())
    except OSError:
        return []

    return sorted(
        (child for child in children if not child.name.startswith(".") and is_folder(child)),
        key=lambda child: child.name.lower(),
    )


def is_folder(path: Path) -> bool:
    """Tell whether a path is a folder. A path that cannot be read counts as no folder.

    Args:
        path: The path to test.

    Returns:
        True when the path is a folder.
    """
    try:
        return path.is_dir()
    except OSError:
        return False
