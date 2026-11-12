from dataclasses import dataclass

from ostra.models.vector import Vector


@dataclass(frozen=True)
class EmbeddedChunk:
    """Hold a chunk with its digest and its embedding vector."""

    digest: str
    text: str
    embedding: Vector
