from pathlib import Path


class ConfigNotFoundError(RuntimeError):
    """Report that the config file does not exist yet."""

    def __init__(self, config_path: Path):
        super().__init__(f"Config file not found. Path={config_path}")
        self.config_path = config_path
