from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Callable

from data_gradients.utils.utils import text_to_blue, text_to_yellow, text_to_red


class Question(ABC):
    """
    Abstract base class representing a generic question interface.

    Implementing classes should provide concrete implementation for the `ask` method.

    Example:
        >>> question_instance = SubclassOfAbstractQuestion()
        >>> answer = question_instance.ask()
    """

    @abstractmethod
    def ask(self, hint: str = "") -> Any:
        """Method to ask the question. Must be implemented by subclasses."""
        ...


class OpenEndedQuestion(Question):
    """
    Represents an open-ended question that can be posed to a user.

    :param question:    The main content of the question.
    :param validation: An optional callable for validating user input.

    Example:
        >>> open_question = OpenEndedQuestion("What is your name?")
        >>> name = open_question.ask()
    """

    def __init__(self, question: str, validation: Optional[Callable[[str], bool]] = None):
        self.question = question
        self.validation = validation

    def ask(self, hint: str = "") -> Any:
        """Pose the open-ended question and capture the user's response."""
        if is_notebook():
            return ask_open_ended_via_jupyter(question=self, hint=hint)
        else:
            return ask_open_ended_via_stdin(question=self, hint=hint)


class FixedOptionsQuestion(Question):
    """
    Represents a question with multiple options for the user to choose from.

    :param question:    The main content of the question.
    :param options:     A dictionary where keys represent options and values give descriptions or corresponding data.

    Example:
        >>> option_question = FixedOptionsQuestion("Choose a color:", {"R": "Red", "G": "Green", "B": "Blue"})
        >>> chosen_color = option_question.ask()
    """

    def __init__(self, question: str, options: Dict[str, Any]):
        self.question = question
        self.options = options

    def ask(self, hint: str = "") -> Any:
        """Pose the question with options and capture the user's choice."""
        if is_notebook():
            return ask_option_via_jupyter(question=self, hint=hint)
        else:
            return ask_option_via_stdin(question=self, hint=hint)


def is_notebook() -> bool:
    """Determines if the current environment is a Jupyter notebook."""
    try:
        from IPython import get_ipython

        shell = get_ipython().__class__.__name__
        if shell == "ZMQInteractiveShell":
            return True  # Jupyter notebook or qtconsole
        elif shell == "TerminalInteractiveShell":
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except ImportError:
        return False
    except NameError:
        return False  # Probably standard Python interpreter


def ask_via_stdin(question: str, optional_description: str, validate_func: Optional[callable] = None, input_message: str = "Enter your response >>> ") -> str:
    """
    Get user input from the command line with optional validation.

    :param question:                The main content of the question.
    :param optional_description:    Additional instruction or context for the question.
    :param validate_func:           An optional function to validate user input.
    :param input_message:           Message to display before the user provides an input.
    :return:                        User's input after successful validation or after the user provides an input.

    Example:
        >>> answer = ask_via_stdin("Enter a number:", "Choose a number between 1 to 10", lambda x: x.isdigit() and 1 <= int(x) <= 10)
    """

    print("\n" + "-" * 80)
    print(text_to_yellow(question))
    print("-" * 80)

    if optional_description:
        print(optional_description)

    answer = None
    while answer is None:
        try:
            answer = input(f"\n{input_message}")
            if validate_func and not validate_func(answer):
                print(f"{text_to_red(f'`{answer}` is not a valid input!')} Please check the instruction and try again.")
                answer = None
        except Exception as e:
            print(text_to_red(f"Oops! {e}. Let's try again."))
            answer = None
    return answer


def ask_open_ended_via_stdin(question: OpenEndedQuestion, hint: str) -> str:
    """
    Capture open-ended responses from the command line.

    :param question:    An instance of OpenEndedQuestion.
    :param hint:        A hint or additional instruction for the question.
    :return:            User's response to the question.
    """
    user_answer = ask_via_stdin(question=question.question, optional_description=hint, validate_func=question.validation)
    print(f"Great! {text_to_yellow(f'You chose: `{user_answer}`')}")
    return user_answer


def ask_option_via_stdin(question: FixedOptionsQuestion, hint: str) -> Any:
    """
    Capture responses for option-based questions from the command line.

    :param question:    An instance of FixedOptionsQuestion.
    :param hint:        A hint or additional instruction for the question.
    :return:            User's selected option.
    """
    options_display = "\n".join([f"[{text_to_blue(idx)}] | {option_description}" for idx, option_description in enumerate(question.options.keys())])
    description = f"{hint if hint else ''}\n{text_to_blue('Options')}:\n{options_display}"

    validate_func = lambda x: x.isdigit() and 0 <= int(x) <= len(question.options) - 1
    input_message = f"Your selection (Enter the {text_to_blue('corresponding number')}) >>> "
    selected_index = int(ask_via_stdin(question=question.question, optional_description=description, validate_func=validate_func, input_message=input_message))

    options_descriptions, options_values = zip(*question.options.items())

    selected_description = options_descriptions[selected_index]
    selected_value = options_values[selected_index]

    print(f"Great! {text_to_yellow(f'You chose: `{selected_description}`')}")
    return selected_value


def ask_open_ended_via_jupyter(question: OpenEndedQuestion, hint: str) -> str:
    """
    Capture open-ended responses within a Jupyter notebook environment.

    :param question:    An instance of OpenEndedQuestion.
    :param hint:        A hint or additional instruction for the question.
    :return:            User's response to the question.
    """
    from data_gradients.utils.jupyter_utils import ui_events

    import ipywidgets as widgets
    from IPython.display import display

    user_answer: Optional[str] = None
    validation_message = widgets.Label(value="")  # A label to display validation errors

    def on_submit(button):
        nonlocal user_answer
        if question.validation is None or question.validation(text_widget.value):
            user_answer = text_widget.value
            print(f"You entered: `{user_answer}`")
        else:
            validation_message.value = f"`{text_to_red(f'{text_widget.value} is not a valid input!')} Please check the instruction and try again."

    # If there's a hint, display it using a Label widget
    hint_label = widgets.Label(value=hint) if hint else None

    # Create a label for the question description
    description_label = widgets.Label(value=question.question)

    text_widget = widgets.Text(placeholder="Enter your response here...", layout=widgets.Layout(width="100%"))  # Adjust width to 100%
    submit_button = widgets.Button(description="Submit", button_style="info")
    submit_button.on_click(on_submit)

    box_items = (
        [hint_label, description_label, text_widget, submit_button, validation_message]
        if hint_label
        else [description_label, text_widget, submit_button, validation_message]
    )
    box = widgets.VBox(box_items)

    display(box)

    with ui_events() as poll:
        while user_answer is None:
            poll(10)

    return user_answer


def ask_option_via_jupyter(question: FixedOptionsQuestion, hint: str) -> str:
    """
    Capture responses for option-based questions within a Jupyter notebook environment using buttons.

    :param question:    An instance of FixedOptionsQuestion.
    :param hint:        A hint or additional instruction for the question.
    :return:            User's selected option.
    """
    from data_gradients.utils.jupyter_utils import ui_events

    import ipywidgets as widgets
    from IPython.display import display

    options: List[str] = list(question.options.keys())
    user_selected_index: Optional[int] = None

    # Create a box to group the hint, the options as buttons, and any outputs
    box = widgets.VBox([])

    if hint:
        hint_label = widgets.Label(value=hint)
        box.children = [hint_label]

    buttons = []
    outputs = []

    for idx, option in enumerate(options):
        button = widgets.Button(description=option, layout=widgets.Layout(width="100%"))
        output = widgets.Output()

        def create_callback(index):
            def callback(button):
                nonlocal user_selected_index
                user_selected_index = index

            return callback

        button.on_click(create_callback(idx))
        buttons.append(button)
        outputs.append(output)

    # Combine buttons and their outputs for display
    combined_widgets = [widget for pair in zip(buttons, outputs) for widget in pair]

    # Append the option buttons and outputs to the box
    box.children += tuple(combined_widgets)
    box.layout = widgets.Layout(margin="0px 0px 10px 0px", padding="0px")  # Adjust spacing between items in the VBox

    display(box)

    with ui_events() as poll:
        while user_selected_index is None:
            poll(10)

    potential_values = list(question.options.values())
    selected_value = potential_values[user_selected_index]
    print(f"Great! {text_to_yellow(f'You chose: `{selected_value}`')}")

    return selected_value


if __name__ == "__main__":
    # Example usage:
    option_q = FixedOptionsQuestion("Choose an option:", {"A": "Option A", "B": "Option B"})
    closed_response = option_q.ask()
    print(closed_response)

    open_q = OpenEndedQuestion("How many classes do you have in your dataset?", validation=lambda x: x.isdigit() and int(x) > 0)
    openended_response = open_q.ask()
    print(openended_response)
