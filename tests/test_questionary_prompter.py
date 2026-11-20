from collections.abc import Iterator

import pytest
from prompt_toolkit.input import PipeInput, create_pipe_input
from prompt_toolkit.output import DummyOutput

from ostra.cli.init.prompter import Choice, SetupCancelled
from ostra.cli.init.questionary_prompter import QuestionaryPrompter

DOWN = "\x1b[B"
ENTER = "\r"
CTRL_C = "\x03"
BACKSPACE = "\x7f"


@pytest.fixture
def pipe() -> Iterator[PipeInput]:
    """Provide a terminal input that the test types into."""
    with create_pipe_input() as pipe_input:
        yield pipe_input


def test_select__arrow_down_and_enter__returns_second_value(pipe: PipeInput):
    """Return the value of the choice that the user moved to."""
    pipe.send_text(DOWN + ENTER)
    prompter = QuestionaryPrompter(input=pipe, output=DummyOutput())

    answer = prompter.select("Model?", [Choice("One", "one"), Choice("Two", "two", "second")])

    assert answer == "two"


def test_text__enter__returns_default(pipe: PipeInput):
    """Return the default when the user presses Enter."""
    pipe.send_text(ENTER)

    assert QuestionaryPrompter(input=pipe, output=DummyOutput()).text("Address?", default="http://x") == "http://x"


def test_integer__invalid_then_valid__returns_number(pipe: PipeInput):
    """Keep asking until the answer passes the validation."""
    pipe.send_text(BACKSPACE + "abc" + ENTER + BACKSPACE * 3 + "500" + ENTER + BACKSPACE * 3 + "42" + ENTER)
    prompter = QuestionaryPrompter(input=pipe, output=DummyOutput())

    answer = prompter.integer("Size?", default=0, validate=lambda value: None if value < 100 else "Too large.")

    assert answer == 42


def test_confirm__ctrl_c__raises_cancelled(pipe: PipeInput):
    """Raise SetupCancelled when the user presses Ctrl-C."""
    pipe.send_text(CTRL_C)

    with pytest.raises(SetupCancelled):
        QuestionaryPrompter(input=pipe, output=DummyOutput()).confirm("Save?")
