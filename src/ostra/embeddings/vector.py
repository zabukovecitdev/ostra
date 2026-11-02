from dataclasses import dataclass


@dataclass(frozen=True)
class Vector:
    """Hold an embedding vector together with the ID of the model that made it.

    A store refuses a vector of another model, so vectors of two models never mix.
    """

    model_id: str
    values: tuple[float, ...]
