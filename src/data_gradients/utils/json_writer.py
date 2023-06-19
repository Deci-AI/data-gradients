import json
import os
import logging
import appdirs
from typing import Dict, List, Union

import data_gradients


logger = logging.getLogger(__name__)

CACHE_DIR = appdirs.user_cache_dir("DataGradients", "Deci")


def _safe_load_json(path: str, require_same_version: bool = False) -> Dict:
    """Load a json file if exists, otherwise return an empty dict. If not valid json, also return an empty dict.

    :param path:                    Path to the json file
    :param require_same_version:    If True, requires the cache file to have the same version as data-gradients
    :return:                        The dict representing the json file
    """
    try:
        if os.path.exists(path):
            with open(path, "r") as f:
                json_dict = json.load(f)
                if json_dict.get("__version__") == data_gradients.__version__ or not require_same_version:
                    return json_dict
                else:
                    logger.info(
                        f"{path} was not loaded from cache due to data-gradients missmatch between cache and current version"
                        f"cache={json_dict.get('__version__')}!={data_gradients.__version__}=installed"
                    )
        return {}
    except json.decoder.JSONDecodeError:
        return {}


def load_cache(path: str) -> Dict:
    """Load cache from a json file. If no valid cache, return an empty dict.
    :param path: Path to the json file
    :return:     The dict representing the cache of the data configuration
    """
    json_dict = _safe_load_json(path, require_same_version=True)
    return json_dict.get("cache", {})


def load_features(path: str, require_same_version: bool) -> List[Dict]:
    """Load cache from a json file. If no valid features, return an empty dict.
    :param path:                    Path to the json file
    :param require_same_version:    If True, requires the cache file to have the same version as data-gradients
    :return:                        List of feature data
    """
    json_dict = _safe_load_json(path, require_same_version=require_same_version)
    return json_dict.get("features", [])


def _log(title: str, data: Union[List, Dict], path: str):
    """Log data to a json file. Save the data-gradients version to the json file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)

    json_dict = _safe_load_json(path)
    json_dict["__version__"] = data_gradients.__version__
    json_dict[title] = data

    with open(path, "w") as f:
        json.dump(json_dict, f, indent=4)


def log_features(features_data: List[Dict], path: str):
    """Log features to a json file."""
    _log(title="features", data=features_data, path=path)


def log_cache(cache_data: Dict, path: str):
    """Log cache to a json file."""
    _log(title="cache", data=cache_data, path=path)


def log_errors(errors_data: List[Dict[str, List[str]]], path: str):
    """Log errors to a json file."""
    _log(title="errors", data=errors_data, path=path)
