from pydantic_settings import BaseSettings, SettingsConfigDict

# The address of an Ollama server on the same machine.
DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"


class OllamaSettings(BaseSettings):
    """Hold the address of an Ollama server.

    The value can come from the OLLAMA_BASE_URL environment variable. The model is not a setting: the model
    spec names it.
    """

    model_config = SettingsConfigDict(env_prefix="OLLAMA_")

    base_url: str = DEFAULT_OLLAMA_BASE_URL
