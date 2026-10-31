import shutil
import subprocess
import sys
from pathlib import Path

SERVICE_NAME = "ostra-daemon.service"
UNIT_TEMPLATE = """[Unit]
Description=ostra background indexer

[Service]
ExecStart={executable} run --interval {interval_minutes}
Restart=on-failure
RestartSec=60

[Install]
WantedBy=default.target
"""


def unit_path() -> Path:
    """Return the location of the systemd user unit file.

    Returns:
        The path in the systemd user folder of the current user.
    """
    return Path.home() / ".config" / "systemd" / "user" / SERVICE_NAME


def install_service(interval_minutes: int) -> str:
    """Write the systemd user unit, then enable and start it.

    Args:
        interval_minutes: Minutes between indexing runs.

    Returns:
        A message that names the unit file.

    Raises:
        RuntimeError: When the operating system is not Linux with systemd, or when ostra-daemon is not on the PATH.
    """
    _require_systemd()
    executable = shutil.which("ostra-daemon")
    if executable is None:
        raise RuntimeError("The ostra-daemon command is not on the PATH. Install ostra first.")

    path = unit_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(UNIT_TEMPLATE.format(executable=executable, interval_minutes=interval_minutes))
    _systemctl("daemon-reload")
    _systemctl("enable", "--now", SERVICE_NAME)
    return f"Service installed and started. Unit={path}"


def uninstall_service() -> str:
    """Stop and disable the systemd user unit, then delete its file.

    Returns:
        A message that tells what was removed.

    Raises:
        RuntimeError: When the operating system is not Linux with systemd.
    """
    _require_systemd()
    path = unit_path()
    if not path.exists():
        return "Service is not installed."

    _systemctl("disable", "--now", SERVICE_NAME)
    path.unlink()
    _systemctl("daemon-reload")
    return f"Service removed. Unit={path}"


def _require_systemd() -> None:
    """Raise an error when the system cannot run a systemd user service."""
    if not sys.platform.startswith("linux") or shutil.which("systemctl") is None:
        raise RuntimeError("Service install supports Linux with systemd only. Run ostra-daemon directly instead.")


def _systemctl(*arguments: str) -> None:
    """Run a systemctl command for the user manager.

    Args:
        arguments: The systemctl arguments.
    """
    subprocess.run(["systemctl", "--user", *arguments], check=True)
