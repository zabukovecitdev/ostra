class ModelIndexMismatchError(RuntimeError):
    """Report that a stored index was built with other settings than the selected model has."""

    def __init__(self, model_id: str, stored_dimensions: int, expected_dimensions: int):
        super().__init__(
            f"The index of the model {model_id} does not match the model settings. "
            f"Stored dimensions: {stored_dimensions}. Expected dimensions: {expected_dimensions}. "
            "The model changed after the index was built. Use a new model ID or rebuild the index."
        )
        self.model_id = model_id
