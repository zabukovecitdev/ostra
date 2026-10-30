import pytest

from ostra.connectors.base import VectorDatabaseConnector


def test_vector_database_connector__instantiated_directly__raises_type_error():
    """Raise TypeError when the abstract base class is instantiated."""
    with pytest.raises(TypeError):
        VectorDatabaseConnector()
