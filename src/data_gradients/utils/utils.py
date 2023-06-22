import os
import re
import shutil
import json
from typing import Dict, Mapping, List


def write_json(path: str, json_dict: Dict):
    dirname = os.path.dirname(path)
    os.makedirs(dirname, exist_ok=True)
    with open(path, "w") as f:
        json.dump(json_dict, f, indent=4)


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


def safe_json_load(path: str) -> Dict:
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r") as f:
            return json.load(f)
    except json.decoder.JSONDecodeError:
        return {}
