from platformdirs import user_data_path

from ostra.enums.database_backend import DatabaseBackend
from ostra.enums.embedding_provider_name import EmbeddingProviderName

APP_NAME = "ostra"
DEFAULT_DATABASE_PATH = user_data_path(APP_NAME) / "ostra.db"
DEFAULT_DATABASE_BACKEND = DatabaseBackend.SQLITE
DEFAULT_EMBEDDING_PROVIDER = EmbeddingProviderName.OLLAMA
DEFAULT_CHUNK_OVERLAP_TOKENS = 50
OWNER_ONLY_DIRECTORY_MODE = 0o700
OWNER_ONLY_FILE_MODE = 0o600

NO_DIRECTORIES_MESSAGE = "No folders are configured. Add folders to the directories list in the config file."
