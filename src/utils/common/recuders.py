from functools import reduce
from numbers import Number
from typing import List


def sum_dictionaries(source: dict, others: List[dict]):
    return reduce(
        lambda a, b: {k: a[k] + b[k] for k in a},
        others,
        source
    )


def sum_scalars(source: Number, others: List[Number]):
    return reduce(
        lambda a, b: a + b,
        others,
        source
    )
