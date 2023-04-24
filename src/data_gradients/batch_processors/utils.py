from typing import Sequence
import torch


def channels_last_to_first(tensors: torch.Tensor) -> torch.Tensor:
    """
    Permute BS, W, H, C -> BS, C, W, H
            0   1  2  3 -> 0   3  1  2
    :param tensors: Tensor[BS, W, H, C]
    :return: Tensor[BS, C, W, H]
    """
    return tensors.permute(0, 3, 1, 2)


def check_all_integers(values: Sequence) -> bool:
    return all(v - int(v) == 0 for v in values)
