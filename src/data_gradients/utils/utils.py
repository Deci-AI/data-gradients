import os
import re
import shutil
from typing import Dict, Mapping, List


def class_id_to_name(mapping, hist: Dict):
    if mapping is None:
        return hist

    new_hist = {}
    for key in list(hist.keys()):
        try:
            new_hist.update({mapping[key]: hist[key]})
        except KeyError:
            new_hist.update({key: hist[key]})
    return new_hist


def fuzzy_keys(params: Mapping) -> List[str]:
    """
    Returns params.key() removing leading and trailing white space, lower-casing and dropping symbols.
    :param params: Mapping, the mapping containing the keys to be returned.
    :return: List[str], list of keys as discussed above.
    """
    return [fuzzy_str(s) for s in params.keys()]


def fuzzy_str(s: str):
    """
    Returns s removing leading and trailing white space, lower-casing and drops
    :param s: str, string to apply the manipulation discussed above.
    :return: str, s after the manipulation discussed above.
    """
    return re.sub(r"[^\w]", "", s).replace("_", "").lower()


def get_fuzzy_mapping_param(name: str, params: Mapping):
    """
    Returns parameter value, with key=name with no sensitivity to lowercase, uppercase and symbols.
    :param name: str, the key in params which is fuzzy-matched and retruned.
    :param params: Mapping, the mapping containing param.
    :return:
    """
    fuzzy_params = {fuzzy_str(key): params[key] for key in params.keys()}
    return fuzzy_params[fuzzy_str(name)]


def ask_user(main_question: str, options: List[str], optional_description: str = "") -> str:
    """Prompt the user to choose an option from a list of options.
    :param main_question:   The main question or instruction for the user.
    :param options:         List of options to chose from.
    :param optional_description:  Optional description to display to the user.
    :return:                The chosen option (key from the options_described dictionary).
    """
    numbers_to_chose_from = range(len(options))

    options_formatted = "\n".join([f"[{number}] {option_description}" for number, option_description in zip(numbers_to_chose_from, options)])

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
            user_answer = input("Your selection (Enter the corresponding number) >>> ")
            user_answer = int(user_answer)
        except Exception:
            user_answer = None

        if user_answer not in numbers_to_chose_from:
            print(f'Oops! "{user_answer}" is not a valid choice. Let\'s try again.')

    selected_option = options[user_answer]
    print(f"Great! You chose: {selected_option}\n")

    return selected_option


def copy_files_by_list(file_list: List[str], source_dir: str, dest_dir: str) -> None:
    """Copy a list of files from the source directory to the destination directory.

    :param file_list:   List of filenames to be copied.
    :param source_dir:  Path of the source directory.
    :param dest_dir:    Path of the destination directory.
    """
    for file_name in file_list:
        source_file_path = os.path.join(source_dir, file_name)
        os.makedirs(dest_dir, exist_ok=True)
        if os.path.isfile(source_file_path):
            dest_file_path = os.path.join(dest_dir, file_name)
            shutil.copy(source_file_path, dest_file_path)
