from abc import ABC, abstractmethod
from types import TracebackType
from typing import Self

from ostra.models.embedding_model_spec import EmbeddingModelSpec
from ostra.store.model_index import ModelIndex


class ChunkStore(ABC):
    """Hold one database with one index for each embedding model. Use it as a context manager."""

    @abstractmethod
    def model_index(self, spec: EmbeddingModelSpec, create: bool) -> ModelIndex:
        """Return the index of an embedding model.

        Args:
            spec: The embedding model.
            create: Create the index when it does not exist. Only the code that writes chunks sets this.

        Returns:
            The index of the model.

        Raises:
            ModelNotIndexedError: When the index does not exist and `create` is False.
            ModelIndexMismatchError: When the stored index was built with other settings than the spec has.
        """

    @abstractmethod
    def close(self) -> None:
        """Release the open database."""

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()
