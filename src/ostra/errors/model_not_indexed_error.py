class ModelNotIndexedError(RuntimeError):
    """Report that nothing was indexed with the requested embedding model."""

    def __init__(self, model_id: str, indexed_model_ids: list[str]):
        indexed = ", ".join(indexed_model_ids) or "none"
        super().__init__(
            f"No index exists for the model {model_id}. Indexed models: {indexed}. "
            f"Run: ostra index (with the model {model_id} selected)."
        )
        self.model_id = model_id
        self.indexed_model_ids = indexed_model_ids
