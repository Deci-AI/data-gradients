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
    """
    Converts labels to one-hot encoded representation.

    :param labels:      Tensor of shape [BS, H, W]
    :param n_classes:   Number of classes in the dataset.
    :return:            Labels tensor shaped as [BS, N, H, W]
    """
    batch_size, height, width = labels.shape

    # Expand dimensions and create a tensor filled with zeros
    labels_one_hot = torch.zeros(batch_size, n_classes, height, width, device=labels.device)

    # Set corresponding class index to 1 in one-hot representation
    labels_long = labels.long()  # Convert labels to long (int64) dtype
    labels_one_hot.scatter_(1, labels_long.unsqueeze(1), 1)

    return labels_one_hot
