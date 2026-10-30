import sys
from pathlib import Path

import pytest

from ostra.connectors.database_facade import DatabaseBackend, DatabaseFacade


def test_database_facade__chroma_not_installed__raises_import_error_with_install_hint(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    """Raise an ImportError that names the chroma extra when chromadb is not installed."""
    monkeypatch.delitem(sys.modules, "ostra.connectors.chroma_connector", raising=False)
    monkeypatch.setitem(sys.modules, "chromadb", None)

    with pytest.raises(ImportError, match=r"ostra\[chroma\]"):
        DatabaseFacade(backend=DatabaseBackend.CHROMA, database_path=tmp_path / "ostra.db")
