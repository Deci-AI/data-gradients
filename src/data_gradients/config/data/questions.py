from dataclasses import dataclass
from time import sleep
from typing import Dict, Any, Optional, List


def text_to_blue(text: str) -> str:
    return f"\033[34;1m{text}\033[0m"


def text_to_yellow(text: str):
    return f"\033[33;1m{text}\033[0m"


@dataclass
class Question:
    """Model a Question with its options
    :attr question: The question string
    :attr options: The options for the question
    """

    question: str
    options: Dict[str, Any]


def ask_question(question: Optional[Question], hint: str = "") -> Any:
    """Method responsible for the whole logic of the class. Read class description for more information.

    :param question:    Question to ask the user for the parameter. This is only used when the parameter was not set in the `__init__` and was
                            not found in the cache.
    :param hint:        Hint to display to the user. This is only displayed when asking a question to the user, and aims at providing extra context,
                            such as showing a sample of data, to help the user answer the question.
    """
    if question is not None:
        answer = ask_user(question.question, options=list(question.options.keys()), optional_description=hint)
        return question.options[answer]


def is_notebook() -> bool:
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


def ask_user_via_stdin(main_question: str, options: List[str], optional_description: str = "") -> str:
    """Prompt the user to choose an option from a list of options.
    :param main_question:   The main question or instruction for the user.
    :param options:         List of options to chose from.
    :param optional_description:  Optional description to display to the user.
    :return:                The chosen option (key from the options_described dictionary).
    """

    numbers_to_chose_from = range(len(options))

    options_formatted = "\n".join([f"[{text_to_blue(number)}] | {option_description}" for number, option_description in zip(numbers_to_chose_from, options)])

    user_answer = None

    while user_answer not in numbers_to_chose_from:
        print("\n------------------------------------------------------------------------")
        print(f"{main_question}")
        print("------------------------------------------------------------------------")
        if optional_description:
            print(optional_description)
        print("\nOptions:")
        print(options_formatted)
        print("")

        try:
            user_answer = input(f"Your selection (Enter the {text_to_blue('corresponding number')}) >>> ")
            user_answer = int(user_answer)
        except Exception:
            user_answer = None

        if user_answer not in numbers_to_chose_from:
            print(f'Oops! "{text_to_blue(str(user_answer))}" is not a valid choice. Let\'s try again.')

    selected_option = options[user_answer]
    print(f"Great! You chose: {text_to_yellow(selected_option)}\n")


def ask_user_via_jupyter(main_question: str, options: List[str], optional_description: str = "") -> str:
    numbers_to_chose_from = range(len(options))

    options_formatted = "\n".join([f"[{text_to_blue(number)}] | {option_description}" for number, option_description in zip(numbers_to_chose_from, options)])

    user_answer = None

    print("\n------------------------------------------------------------------------")
    print(f"{main_question}")
    print("------------------------------------------------------------------------")
    if optional_description:
        print(optional_description)
    print("\nOptions:")
    print(options_formatted)
    print("")

    import ipywidgets as widgets
    from IPython.display import display
    from jupyter_ui_poll import ui_events

    for i, option in enumerate(options):
        button = widgets.Button(description=option)
        button.value = i
        output = widgets.Output()

        display(button, output)

        def on_button_clicked(b):
            with output:
                nonlocal user_answer
                user_answer = b.value
                print("You selected option: " + b.value)

        button.on_click(on_button_clicked)

    with ui_events() as poll:
        while user_answer is None:
            poll(10)

    selected_option = options[user_answer]
    print(f"Great! You chose: {text_to_yellow(selected_option)}\n")
    return selected_option


def ask_user(main_question: str, options: List[str], optional_description: str = "") -> str:
    """Prompt the user to choose an option from a list of options.
    Depending on the environment, the user will be prompted via stdin or via a Jupyter widget.
    :param main_question:   The main question or instruction for the user.
    :param options:         List of options to chose from.
    :param optional_description:  Optional description to display to the user.
    :return:                The chosen option (key from the options_described dictionary).
    """

    if is_notebook():
        return ask_user_via_jupyter(main_question, options, optional_description)
    else:
        return ask_user_via_stdin(main_question, options, optional_description)
