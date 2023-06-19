import json
import os
import logging
import appdirs
from typing import Dict, List, Union

import data_gradients


logger = logging.getLogger(__name__)

MAIN_CACHE_DIR = appdirs.user_cache_dir("DataGradients", "Deci")


def _safe_load_json(path: str, require_version: bool = False) -> Dict:
    try:
        if os.path.exists(path):
            with open(path, "r") as f:
                json_dict = json.load(f)
                if json_dict.get("__version__") == data_gradients.__version__ or not require_version:
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
    json_dict = _safe_load_json(path, require_version=True)
    return json_dict.get("cache", {})


def load_features(path: str, require_version: bool) -> List[Dict]:
    json_dict = _safe_load_json(path, require_version=require_version)
    return json_dict.get("features", [])


def _log(title: str, data: Union[List, Dict], path):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    json_dict = _safe_load_json(path)
    json_dict["__version__"] = data_gradients.__version__
    json_dict[title] = data

    with open(path, "w") as f:
        json.dump(json_dict, f, indent=4)


def log_features(features_data: List[Dict], path: str):
    _log(title="features", data=features_data, path=path)


def log_cache(cache_data: Dict, path: str):
    _log(title="cache", data=cache_data, path=path)


def log_errors(errors_data: List[Dict[str, List[str]]], path: str):
    _log(title="errors", data=errors_data, path=path)
