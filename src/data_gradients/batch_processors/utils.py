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


def to_one_hot(labels: torch.Tensor, n_classes: int) -> torch.Tensor:
    """Method gets label with the shape of [BS, N, H, W] where N is the number of classes.
    :param labels:      Tensor of shape [BS, H, W]
    :param n_classes:   Number of classes in the dataset.
    :return:            Labels tensor shaped as [BS, N, H, W]
    """

    masks = []
    labels = labels.to(torch.int64)
    for label in labels:
        label = torch.nn.functional.one_hot(label, n_classes).byte()
        masks.append(label)
    labels = torch.concat(masks, dim=0).permute(0, -1, 1, 2)
    return labels
