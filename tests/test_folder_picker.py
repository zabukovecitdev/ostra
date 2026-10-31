from pathlib import Path

from ostra.gui.folder_picker import list_subfolders


def test_list_subfolders__mixed_content__returns_visible_folders_sorted(tmp_path: Path):
    """Return the folders by name, without files and without hidden folders."""
    (tmp_path / "beta").mkdir()
    (tmp_path / "Alpha").mkdir()
    (tmp_path / ".hidden").mkdir()
    (tmp_path / "notes.txt").write_text("text")

    assert list_subfolders(tmp_path) == [tmp_path / "Alpha", tmp_path / "beta"]


def test_list_subfolders__folder_missing__returns_empty_list(tmp_path: Path):
    """Return an empty list when the folder cannot be read."""
    assert list_subfolders(tmp_path / "missing") == []
