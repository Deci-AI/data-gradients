from typing import Optional, List, Tuple

import torch
from torch import Tensor

from data_gradients.batch_processors.formatters.base import BatchFormatter
from data_gradients.batch_processors.utils import check_all_integers, to_one_hot
from data_gradients.batch_processors.formatters.utils import ensure_images_shape, ensure_channel_first, drop_nan


class SegmentationBatchFormatter(BatchFormatter):
    """
    Segmentation formatter class
    """

    def __init__(
        self,
        n_classes: int,
        n_image_channels: int,
        threshold_value: float,
        ignore_labels: Optional[List[int]] = None,
    ):
        """
        :param n_classes:           Number of valid classes
        :param n_image_channels:    Number of image channels (3 for RGB, 1 for Gray Scale, ...)
        :param threshold_value:     Threshold
        :param ignore_labels:       Numbers that we should avoid from analyzing as valid classes, such as background
        """
        self.n_classes_used = n_classes
        self.n_image_channels = n_image_channels
        self.ignore_labels = ignore_labels or []

        self.threshold_value = threshold_value
        self.is_input_soft_label = None

    def format(self, images: Tensor, labels: Tensor) -> Tuple[Tensor, Tensor]:
        """Validate batch images and labels format, and ensure that they are in the relevant format for segmentation.

        :param images: Batch of images, in (BS, ...) format
        :param labels: Batch of labels, in (BS, ...) format
        :return:
            - images: Batch of images already formatted into (BS, C, H, W)
            - labels: Batch of labels already formatted into (BS, N, H, W)
        """
        images = drop_nan(images)
        labels = drop_nan(labels)

        images = ensure_channel_first(images, n_image_channels=self.n_image_channels)
        labels = ensure_channel_first(labels, n_image_channels=self.n_image_channels)

        images = ensure_images_shape(images, n_image_channels=self.n_image_channels)
        labels = self.ensure_labels_shape(labels, n_classes=self.n_image_channels, ignore_labels=self.ignore_labels)

        labels = self.ensure_hard_labels(labels, n_classes_used=self.n_classes_used, ignore_labels=self.ignore_labels, threshold_value=self.threshold_value)

        if self.require_onehot(labels, n_classes_used=self.n_classes_used, total_n_classes=self.total_n_classes):
            labels = to_one_hot(labels, n_classes=self.total_n_classes)

        for ignore_label in self.ignore_labels:
            labels[:, ignore_label, ...] = torch.zeros_like(labels[:, ignore_label, ...])

        if 0 <= images.min() and images.max() <= 1:
            images *= 255
            images = images.to(torch.uint8)

        if 0 <= images.min() and images.max() <= 1:
            images = images * 255
        images = images.astype(torch.uint8)

        return images, labels

    @property
    def total_n_classes(self) -> int:
        return self.n_classes_used + len(self.ignore_labels)

    @staticmethod
    def ensure_hard_labels(labels: Tensor, n_classes_used: int, ignore_labels: List[int], threshold_value: float) -> Tensor:
        unique_values = torch.unique(labels)

        if check_all_integers(unique_values):
            return labels
        elif 0 <= min(unique_values) and max(unique_values) <= 1 and check_all_integers(unique_values * 255):
            return labels * 255
        else:
            if n_classes_used > 1:
                raise NotImplementedError(
                    f"Not supporting soft-labeling for number of classes > 1!\nGot {n_classes_used} classes, while ignore labels are {ignore_labels}."
                )
            labels = SegmentationBatchFormatter.binary_mask_above_threshold(labels=labels, threshold_value=threshold_value)
        return labels

    @staticmethod
    def is_soft_labels(labels: Tensor) -> bool:
        unique_values = torch.unique(labels)
        if check_all_integers(unique_values):
            return False
        elif 0 <= min(unique_values) and max(unique_values) <= 1 and check_all_integers(unique_values * 255):
            return False
        return True

    @staticmethod
    def require_onehot(labels: Tensor, n_classes_used: int, total_n_classes: int) -> bool:
        is_binary = n_classes_used == 1
        is_onehot = labels.shape[1] == total_n_classes
        return not (is_binary or is_onehot)

    @staticmethod
    def ensure_labels_shape(labels: Tensor, n_classes: int, ignore_labels: List[int]) -> Tensor:
        """
        Validating labels dimensions are (BS, N, H, W) where N is either 1 or number of valid classes
        :param labels: Tensor [BS, N, W, H]
        :return: labels: Tensor [BS, N, W, H]
        """
        if labels.dim() == 3:
            labels = labels.unsqueeze(1)  # Probably (B, H, W)
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

    @staticmethod
    def binary_mask_above_threshold(labels: Tensor, threshold_value: float) -> Tensor:
        # Support only for binary segmentation
        labels = torch.where(
            labels > threshold_value,
            torch.ones_like(labels),
            torch.zeros_like(labels),
        )
        return labels
