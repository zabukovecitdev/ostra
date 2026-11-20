from platformdirs import user_cache_path, user_data_path

APP_NAME = "ostra"
DATA_DIRECTORY = user_data_path(APP_NAME)
# The models that run in the ostra process are downloaded here on first use.
MODEL_CACHE_DIRECTORY = user_cache_path(APP_NAME) / "models"
DEFAULT_DATABASE_PATH = DATA_DIRECTORY / "ostra.db"
DEFAULT_CHUNK_OVERLAP_TOKENS = 50
OWNER_ONLY_DIRECTORY_MODE = 0o700
OWNER_ONLY_FILE_MODE = 0o600

NO_DIRECTORIES_MESSAGE = (
    "No folders are configured. Run ostra init to add folders, or add them to the directories list in the config file."
)
