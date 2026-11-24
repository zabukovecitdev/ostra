import sys
from pathlib import Path

from ostra.cli import configure_logging
from ostra.cli.parser import build_parser
from ostra.constants import NO_CONFIG_MESSAGE
from ostra.errors.config_not_found_error import ConfigNotFoundError
from ostra.models.app_config import AppConfig
from ostra.repositories.config_repository import load_config
from ostra.services.index_session import open_index


def main(argv: list[str] | None = None) -> None:
    """Run the command given on the command line, or print the help when no command is given.

    Args:
        argv: Command line arguments without the program name. Read sys.argv when None.
    """
    configure_logging()
    parser = build_parser()
    arguments = parser.parse_args(argv)
    if arguments.command is None:
        parser.print_help()
        return

    if not arguments.needs_index:
        arguments.handler(arguments)
        return

    config = _apply_database_override(_load_config_or_exit(), arguments.database)
    with open_index(config, model_id=arguments.model, create=arguments.create_index) as session:
        arguments.handler(arguments, config, session)


def _load_config_or_exit() -> AppConfig:
    """Load the application settings, or stop with a message when the config file does not exist yet.

    Returns:
        The application settings.

    Raises:
        SystemExit: With status 1 when the config file does not exist.
    """
    try:
        return load_config()
    except ConfigNotFoundError:
        print(NO_CONFIG_MESSAGE, file=sys.stderr)
        raise SystemExit(1) from None


def _apply_database_override(config: AppConfig, database: Path | None) -> AppConfig:
    """Replace the database location of the config file with the one given on the command line.

    Args:
        config: The application settings.
        database: The database file from the --database option, or None to keep the configured one.

    Returns:
        The settings with the database path of the option, or the unchanged settings.
    """
    if database is None:
        return config

    return config.model_copy(update={"database_path": database.expanduser()})


if __name__ == "__main__":
    main()
