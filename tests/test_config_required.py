from pathlib import Path

import pytest

from ostra.cli import main
from ostra.constants import NO_CONFIG_MESSAGE
from ostra.repositories import config_repository


@pytest.mark.parametrize("command", [["index"], ["search", "alpha"]], ids=["index", "search"])
def test_main__config_file_missing__exits_with_status_one_and_creates_nothing(
    command: list[str],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
):
    """Exit with status 1 and print the setup message, without creating the config file or opening the database."""
    config_path = tmp_path / "ostra" / "config.json"
    monkeypatch.setattr(config_repository, "default_config_path", lambda: config_path)

    def fail_if_opened(*args, **kwargs):
        raise AssertionError("the database must not be opened without a config file")

    monkeypatch.setattr(main, "open_index", fail_if_opened)

    with pytest.raises(SystemExit) as exit_info:
        main.main(command)

    assert exit_info.value.code == 1
    assert capsys.readouterr().err == f"{NO_CONFIG_MESSAGE}\n"
    assert not config_path.exists()
    assert list(tmp_path.iterdir()) == []
