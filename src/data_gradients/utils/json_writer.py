import json
import os
from typing import Dict

import data_gradients


class JsonWriter:
    def __init__(self, source_path: str):
        """
        :param source_path: Path where to load the cache from, if it exists.
        """
        os.makedirs(os.path.dirname(source_path), exist_ok=True)
        self.source = source_path
        self.data = {"__version__": data_gradients.__version__, "cache": self._load_cache(cache_path=source_path), "stats": {}}

    @staticmethod
    def _load_cache(cache_path: str) -> Dict:
        if os.path.exists(cache_path):
            with open(cache_path, "r") as f:
                previous_data = json.load(f)
                if previous_data.get("__version__") == data_gradients.__version__:
                    return previous_data.get("cache", {})
        return {}

    @property
    def cache(self):
        return self.data["cache"]

    @cache.setter
    def cache(self, cache: Dict):
        self.data["cache"] = cache

    def log_data(self, title: str, data: Dict):
        self.data["stats"][title] = data

    def write(self, output_path: str):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(self.data, f, indent=4)
