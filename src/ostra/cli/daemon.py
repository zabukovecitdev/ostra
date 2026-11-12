import argparse

from ostra.cli import configure_logging
from ostra.services.background_indexing_service import run_forever
from ostra.services.systemd_service import install_service, uninstall_service

DEFAULT_INTERVAL_MINUTES = 15


def main(argv: list[str] | None = None) -> None:
    """Run the background indexer, or install or remove its user service.

    Args:
        argv: Command line arguments without the program name. Read sys.argv when None.
    """
    configure_logging()
    arguments = _build_parser().parse_args(argv)
    if arguments.command == "install":
        print(install_service(interval_minutes=arguments.interval))
    elif arguments.command == "uninstall":
        print(uninstall_service())
    else:
        run_forever(interval_minutes=arguments.interval)


def _build_parser() -> argparse.ArgumentParser:
    """Build the command line parser with the run, install and uninstall commands.

    Returns:
        The parser. The run command is the default.
    """
    parser = argparse.ArgumentParser(prog="ostra-daemon", description="Index text files in the background.")
    parser.set_defaults(command="run", interval=DEFAULT_INTERVAL_MINUTES)
    commands = parser.add_subparsers(dest="command", metavar="COMMAND")
    for name, text in (
        ("run", "Index in a loop until stopped. This is the default."),
        ("install", "Install and start a systemd user service that runs this loop."),
    ):
        command = commands.add_parser(name, help=text, description=text)
        command.add_argument(
            "--interval",
            type=int,
            default=DEFAULT_INTERVAL_MINUTES,
            metavar="MINUTES",
            help=f"Minutes between runs. Default is {DEFAULT_INTERVAL_MINUTES}.",
        )
    commands.add_parser("uninstall", help="Stop and remove the systemd user service.")
    return parser


if __name__ == "__main__":
    main()
