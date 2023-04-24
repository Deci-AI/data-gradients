from typing import Optional, List, Tuple

import torch
from torch import Tensor

from data_gradients.batch_processors.validators.base import BatchValidator
from data_gradients.batch_processors.utils import check_all_integers, channels_last_to_first


class SegmentationBatchValidator(BatchValidator):
    """
    Segmentation validator class
    """

    def __init__(
        self,
        n_classes: int,
        n_image_channels: int,
        threshold_value: float,
        ignore_labels: Optional[List[int]] = None,
    ):
        """
        Constructor gets number of classes and ignore labels in order to understand how to data labels should look like
        :param n_classes: number of valid classes
        :param ignore_labels: list of numbers that we should avoid from analyzing as valid classes, such as background
        """
        self.n_classes_used = n_classes
        self.n_image_channels = n_image_channels
        self.ignore_labels = ignore_labels or []

        self.threshold_value = threshold_value
        self.is_input_soft_label = None

    def __call__(self, images: Tensor, labels: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Validating object came out of next() method activated on the iterator.
        Check & Fix length, type, dimensions, channels-first, pixel values and checks if onehot
        :param objects: Tuple from next(Iterator)
        :return: images, labels as Tuple[Tensor, Tensor] with shape [[BS, C, W, H], [BS, N, W, H]]
        """
        images = drop_nan(images)
        labels = drop_nan(labels)

        images = ensure_images_shape(images, n_image_channels=self.n_image_channels)
        labels = ensure_labels_shape(labels, n_classes=self.n_image_channels, ignore_labels=self.ignore_labels)

        images = ensure_channel_first(images, n_image_channels=self.n_image_channels)
        labels = ensure_channel_first(labels, n_image_channels=self.n_image_channels)

        labels = ensure_hard_labels(labels, n_classes_used=self.n_classes_used, ignore_labels=self.ignore_labels, threshold_value=self.threshold_value)

        if require_onehot(labels, n_classes_used=self.n_classes_used, total_n_classes=self.total_n_classes):
            labels = to_one_hot(labels, n_classes=self.total_n_classes)

        for ignore_label in self.ignore_labels:
            labels[:, ignore_label, ...] = torch.zeros_like(labels[:, ignore_label, ...])

        return images, labels

    @property
    def total_n_classes(self) -> int:
        return self.n_classes_used + len(self.ignore_labels)


def ensure_hard_labels(labels: Tensor, n_classes_used: int, ignore_labels: List[int], threshold_value: float) -> Tensor:
    if not check_all_integers(values=torch.unique(labels)):
        if n_classes_used > 1:
            raise NotImplementedError(
                f"Not supporting soft-labeling for number of classes > 1!\nGot {n_classes_used} classes, while ignore labels are {ignore_labels}."
            )
        labels = binary_mask_above_threshold(labels=labels, threshold_value=threshold_value)
    return labels


def is_soft_labels(labels: Tensor) -> bool:
    unique_values = torch.unique(labels)
    if check_all_integers(unique_values):
        return False
    elif 0 <= min(unique_values) and max(unique_values) <= 1 and check_all_integers(unique_values * 255):
        return False
    return True


def require_onehot(labels: Tensor, n_classes_used: int, total_n_classes: int) -> bool:
    is_binary = n_classes_used == 1
    is_onehot = labels.shape[1] == total_n_classes
    return not (is_binary or is_onehot)


def ensure_images_shape(images: Tensor, n_image_channels: int) -> Tensor:
    """
    Validating images dimensions are (BS, Channels, W, H)
    :param images: Tensor [BS, C, W, H]
    :return: images: Tensor [BS, C, W, H]
    """
    if images.dim() != 4:
        raise ValueError(f"Images batch shape should be (BatchSize x Channels x Width x Height). Got {images.shape}")

    if images.shape[1] != n_image_channels and images.shape[-1] != n_image_channels:
        raise ValueError(f"Images should have {n_image_channels} number of channels. Got {min(images[0].shape)}")

    return images


def ensure_labels_shape(labels: Tensor, n_classes: int, ignore_labels: List[int]) -> Tensor:
    """
    Validating labels dimensions are (BS, N, W, H) where N is either 1 or number of valid classes
    :param labels: Tensor [BS, N, W, H]
    :return: labels: Tensor [BS, N, W, H]
    """
    if labels.dim() == 3:
        labels = labels.unsqueeze(1)  # Probably (B, W, H)
        return labels
    elif labels.dim() == 4:
        total_n_classes = n_classes + len(ignore_labels)
        valid_n_classes = (total_n_classes, 1)
        input_n_classes = labels.shape[1]
        if input_n_classes not in valid_n_classes and labels.shape[-1] not in valid_n_classes:
            raise ValueError(
                f"Labels batch shape should be [BS, N, W, H] where N is either 1 or n_classes + len(ignore_labels)"
                f" ({total_n_classes}). Got: {input_n_classes}"
            )
        return labels
    else:
        raise ValueError(f"Labels batch shape should be [BatchSize x Channels x Width x Height]. Got {labels.shape}")


def to_one_hot(labels: torch.Tensor, n_classes: int) -> torch.Tensor:
    """
    Method gets label with the shape of [BS, N, W, H] where N is either 1 or n_classes, if is_one_hot=True.
    param label: Tensor
    param is_one_hot: Determine if labels are one-hot shaped
    :return: Labels tensor shaped as [BS, VC, W, H] where VC is Valid Classes only - ignores are omitted.
    """
    masks = []
    labels = labels.to(torch.int64)

    for label in labels:
        label = torch.nn.functional.one_hot(label, n_classes)
        masks.append(label)
    labels = torch.concat(masks, dim=0).permute(0, -1, 1, 2)

    return labels


def ensure_channel_first(images: Tensor, n_image_channels: int) -> Tensor:
    """Images should be [BS, C, W, H]. If [BS, W, H, C], permute

    :param images: Tensor
    :return: images: Tensor [BS, C, W, H]
    """
    if images.shape[1] != n_image_channels and images.shape[-1] == n_image_channels:
        images = channels_last_to_first(images)
    return images


def drop_nan(tensor: Tensor) -> Tensor:
    nans = torch.isnan(tensor)
    if nans.any():
        nan_indices = set(nans.nonzero()[:, 0].tolist())
        all_indices = set(i for i in range(tensor.shape[0]))
        valid_indices = all_indices - nan_indices
        return tensor[valid_indices]
    return tensor


def binary_mask_above_threshold(labels: Tensor, threshold_value: float) -> Tensor:
    # Support only for binary segmentation
    labels = torch.where(
        labels > threshold_value,
        torch.ones_like(labels),
        torch.zeros_like(labels),
    )
    return labels
