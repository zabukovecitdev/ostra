from collections.abc import Sequence
from typing import Any

import questionary
from prompt_toolkit.input import Input
from prompt_toolkit.output import Output

from ostra.cli.config.prompter import Choice, Prompter, SetupCancelled, Validator


class QuestionaryPrompter(Prompter):
    """Ask the questions with questionary: arrow-key lists, path completion and inline validation."""

    def __init__(self, input: Input | None = None, output: Output | None = None):
        """Create the prompter.

        Args:
            input: The terminal input. None means the real terminal. Tests pass a pipe.
            output: The terminal output. None means the real terminal.
        """
        self._io: dict[str, Any] = {"input": input, "output": output}

    def select(self, message: str, choices: Sequence[Choice], default: str | None = None) -> str:
        """Let the user pick one choice with the arrow keys and return its value."""
        options = [
            questionary.Choice(choice.title, value=choice.value, description=choice.description or None)
            for choice in choices
        ]
        return _ask(questionary.select(message, choices=options, default=default, **self._io))

    def text(self, message: str, default: str = "", validate: Validator | None = None) -> str:
        """Ask for a line of text."""
        return _ask(questionary.text(message, default=default, validate=_adapt(validate), **self._io))

    def path(self, message: str, default: str = "", validate: Validator | None = None) -> str:
        """Ask for a path with tab completion."""
        return _ask(questionary.path(message, default=default, validate=_adapt(validate), **self._io))

    def confirm(self, message: str, default: bool = True) -> bool:
        """Ask a yes or no question."""
        return _ask(questionary.confirm(message, default=default, **self._io))


def _ask(question: questionary.Question) -> Any:
    """Ask the question and return the answer.

    Raises:
        SetupCancelled: When the user pressed Ctrl-C, which makes questionary return None.
    """
    answer = question.ask(kbi_msg="")
    if answer is None:
        raise SetupCancelled

    return answer


def _adapt(validate: Validator | None) -> Any:
    """Turn a validator that returns an error message or None into the form questionary expects."""
    if validate is None:
        return None

    return lambda answer: validate(answer) or True
