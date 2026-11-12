from pydantic_settings import BaseSettings, SettingsConfigDict


class OllamaSettings(BaseSettings):
    """Hold the address of an Ollama server.

    The value can come from the OLLAMA_BASE_URL environment variable. The model is not a setting: the model
    spec names it.
    """

    model_config = SettingsConfigDict(env_prefix="OLLAMA_")

    base_url: str = "http://localhost:11434"
