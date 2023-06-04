from typing import List
import torch


def channels_last_to_first(tensors: torch.Tensor) -> torch.Tensor:
    """
    Permute BS, W, H, C -> BS, C, H, W
            0   1  2  3 -> 0   3  1  2
    :param tensors: Tensor[BS, W, H, C]
    :return: Tensor[BS, C, H, W]
    """
    return tensors.permute(0, 3, 1, 2)


def check_all_integers(tensor: torch.Tensor) -> bool:
    return torch.all(torch.eq(tensor, tensor.to(torch.int))).item()


def to_one_hot(labels: torch.Tensor, class_ids: List[int]) -> torch.Tensor:
    """Method gets label with the shape of [BS, N, H, W] where N is the number of classes.
    :param labels:      Tensor of shape [BS, H, W]
    :param class_ids:   List of ids to class names. Ids not mapped will be ignored
    :return:            Labels tensor shaped as [BS, N, H, W]
    """

    # class_ids might not include every ids (in case the user want to ignore some ids).
    # This means we don't know the number of classes, so the workaround is to take the max value between batch ids and class_ids.
    n_dims = int(max(labels.max().item() + 1, *class_ids))

    masks = []
    labels = labels.to(torch.int64)
    for label in labels:
        label = torch.nn.functional.one_hot(label, n_dims).byte()
        masks.append(label)
    labels = torch.concat(masks, dim=0).permute(0, -1, 1, 2)
    return labels
