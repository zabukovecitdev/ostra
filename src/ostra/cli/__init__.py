import logging


def configure_logging() -> None:
    """Log at INFO level, but keep the per-request lines of the HTTP client out of the output."""
    logging.basicConfig(level=logging.INFO)
    logging.getLogger("httpx").setLevel(logging.WARNING)
