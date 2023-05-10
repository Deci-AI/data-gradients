from typing import Dict, List, Any, Tuple

import numpy as np


def align_histogram_keys(train_histogram: Dict[str, Any], val_histogram: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Enforces the keys of training and validation histograms to be the same.
    If one of the keys is missing, the histogram will filled with defaults value (0, 0.0, "") depending on the situation

    :param train_histogram:  Histogram representing metrics from training split.
    :param val_histogram:  Histogram representing metrics from validation split.
    :return: A merged dictionary containing key-value pairs from both "train" and "val" splits.
    """
    keys = set(train_histogram.keys()) | set(val_histogram.keys())

    aligned_train_histogram, aligned_val_histogram = {}, {}
    for key in keys:
        train_value = train_histogram.get(key)
        val_value = val_histogram.get(key)

        value_type = type(train_value) if train_value is not None else type(val_value)
        default_value = value_type()

        aligned_train_histogram[key] = train_value or default_value
        aligned_val_histogram[key] = val_value or default_value

    return aligned_train_histogram, aligned_val_histogram


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
