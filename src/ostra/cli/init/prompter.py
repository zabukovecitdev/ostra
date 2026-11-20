from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from dataclasses import dataclass

# A validator returns an error message for an answer it rejects, or None for an answer it accepts.
Validator = Callable[[str], str | None]
IntegerValidator = Callable[[int], str | None]


class SetupCancelled(Exception):
    """The user stopped the setup before it saved anything."""


@dataclass(frozen=True)
class Choice:
    """One option of a select question."""

    title: str
    value: str
    description: str = ""


class Prompter(ABC):
    """Ask the user questions in the terminal.

    The setup steps depend on this class, not on a prompt library, so that tests can answer the questions.
    Each method raises SetupCancelled when the user stops the setup, for example with Ctrl-C.
    """

    @abstractmethod
    def select(self, message: str, choices: Sequence[Choice], default: str | None = None) -> str:
        """Let the user pick one choice and return its value."""

    @abstractmethod
    def text(self, message: str, default: str = "", validate: Validator | None = None) -> str:
        """Ask for a line of text."""

    @abstractmethod
    def path(self, message: str, default: str = "", validate: Validator | None = None) -> str:
        """Ask for a file or folder path, with completion where the terminal supports it."""

    @abstractmethod
    def confirm(self, message: str, default: bool = True) -> bool:
        """Ask a yes or no question."""

    def integer(self, message: str, default: int, validate: IntegerValidator | None = None) -> int:
        """Ask for a whole number.

        Args:
            message: The question.
            default: The answer when the user presses Enter.
            validate: Rejects a number with an error message, or accepts it with None.

        Returns:
            The number.
        """

        def check(answer: str) -> str | None:
            try:
                value = int(answer)
            except ValueError:
                return "Enter a whole number."

            return validate(value) if validate else None

        return int(self.text(message, default=str(default), validate=check))


class DefaultsPrompter(Prompter):
    """Answer every question with its default, for a setup without questions."""

    def select(self, message: str, choices: Sequence[Choice], default: str | None = None) -> str:
        """Return the default choice, or the first choice when there is no default."""
        return default if default is not None else choices[0].value

    def text(self, message: str, default: str = "", validate: Validator | None = None) -> str:
        """Return the default text."""
        return default

    def path(self, message: str, default: str = "", validate: Validator | None = None) -> str:
        """Return the default path."""
        return default

    def confirm(self, message: str, default: bool = True) -> bool:
        """Return the default answer."""
        return default
