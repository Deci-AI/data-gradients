from collections import OrderedDict
from typing import Dict, List

import numpy as np


def merge_dict_splits(hist: Dict):
    for key in [*hist["train"], *hist["val"]]:
        if key not in hist["train"]:
            hist["train"][key] = type(hist["val"][key])()
        if key not in hist["val"]:
            hist["val"][key] = type(hist["train"][key])()

    hist["train"] = OrderedDict(sorted(hist["train"].items()))
    hist["val"] = OrderedDict(sorted(hist["val"].items()))


def normalize_values_to_percentages(counters: List[float], total_count: float) -> List[float]:
    """
    Normalize a list of count to percentages relative to a total value.

    :param counters:    Values to normalize.
    :param total_count: Total number of values, which will be used to calculate percentages.
    :return:            Values representing the percentages of each input value.
    """
    if total_count == 0:
        total_count = 1
    return [np.round(((100 * count) / total_count), 3) for count in counters]
