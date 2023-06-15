from typing import Optional, List, Tuple

import torch
from torch import Tensor

from data_gradients.config.data.questions import ask_user
from data_gradients.batch_processors.formatters.base import BatchFormatter
from data_gradients.batch_processors.utils import check_all_integers, to_one_hot
from data_gradients.batch_processors.formatters.utils import DatasetFormatError, ensure_images_shape, ensure_channel_first, drop_nan


class SegmentationBatchFormatter(BatchFormatter):
    """
    Segmentation formatter class
    """

    def __init__(
        self,
        class_names: List[str],
        class_names_to_use: List[str],
        n_image_channels: int,
        threshold_value: float,
        ignore_labels: Optional[List[int]] = None,
    ):
        """
        :param class_names:         List of all class names in the dataset. The index should represent the class_id.
        :param class_names_to_use:  List of class names that we should use for analysis.
        :param n_image_channels:    Number of image channels (3 for RGB, 1 for Gray Scale, ...)
        :param threshold_value:     Threshold
        :param ignore_labels:       Numbers that we should avoid from analyzing as valid classes, such as background
        """
        class_names_to_use = set(class_names_to_use)

        self.class_names = class_names
        self.class_ids_to_ignore = [class_id for class_id, class_name in enumerate(class_names) if class_name not in class_names_to_use]

        self.n_image_channels = n_image_channels
        self.ignore_labels = ignore_labels or []

        self.threshold_value = threshold_value
        self.is_input_soft_label = None
        self.is_batch = None

    def format(self, images: Tensor, labels: Tensor) -> Tuple[Tensor, Tensor]:
        """Validate batch images and labels format, and ensure that they are in the relevant format for segmentation.

        :param images: Batch of images, in (BS, ...) format
        :param labels: Batch of labels, in (BS, ...) format
        :return:
            - images: Batch of images already formatted into (BS, C, H, W)
            - labels: Batch of labels already formatted into (BS, N, H, W)
        """

        if self.is_batch is None:
            # if less any dim is 4, we know it's a batch
            if images.ndim == 4 or labels.ndim == 4:
                self.is_batch = True
            # If image or mask only includes 2 dims, we can guess it's a single sample
            elif images.ndim == 2 or labels.ndim == 2:
                self.is_batch = False
            # Otherwise, we need to ask the user
            else:
                is_batch_descriptions = {"Batch Data": True, "Single Image Data": False}
                selected_option = ask_user(
                    main_question=(
                        f"Do your tensors represent a batch or a single image data?\n    - Image shape: {images.shape}\n    - Mask shape:  {labels.shape}"
                    ),
                    options=list(is_batch_descriptions.keys()),
                )
                self.is_batch = is_batch_descriptions[selected_option]

        if not self.is_batch:
            images = images.unsqueeze(0)
            labels = labels.unsqueeze(0)

        images = drop_nan(images)
        labels = drop_nan(labels)

        images = ensure_channel_first(images, n_image_channels=self.n_image_channels)
        labels = ensure_channel_first(labels, n_image_channels=self.n_image_channels)

        images = ensure_images_shape(images, n_image_channels=self.n_image_channels)
        labels = self.ensure_labels_shape(labels, n_classes=self.n_image_channels, ignore_labels=self.ignore_labels)

        labels = self.ensure_hard_labels(labels, n_classes=len(self.class_names), threshold_value=self.threshold_value)

        if self.require_onehot(labels=labels, n_classes=len(self.class_names)):
            labels = to_one_hot(labels, n_classes=len(self.class_names))

        for class_id_to_ignore in self.class_ids_to_ignore:
            labels[:, class_id_to_ignore, ...] = 0

        if 0 <= images.min() and images.max() <= 1:
            images *= 255
            images = images.to(torch.uint8)

        return images, labels

    @staticmethod
    def ensure_hard_labels(labels: Tensor, n_classes: int, threshold_value: float) -> Tensor:
        unique_values = torch.unique(labels)

        if check_all_integers(unique_values):
            return labels
        elif 0 <= min(unique_values) and max(unique_values) <= 1 and check_all_integers(unique_values * 255):
            return labels * 255
        else:
            if n_classes > 1:
                raise DatasetFormatError(f"Not supporting soft-labeling for number of classes > 1!\nGot {n_classes} classes.")
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
    def require_onehot(labels: Tensor, n_classes: int) -> bool:
        is_binary = n_classes == 1
        is_onehot = labels.shape[1] == n_classes
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
                raise DatasetFormatError(
                    f"Labels batch shape should be [BS, N, W, H] where N is either 1 or n_classes + len(ignore_labels)"
                    f" ({total_n_classes}). Got: {input_n_classes}"
                )
            return labels
        else:
            raise DatasetFormatError(f"Labels batch shape should be [BatchSize x Channels x Width x Height]. Got {labels.shape}")

    @staticmethod
    def binary_mask_above_threshold(labels: Tensor, threshold_value: float) -> Tensor:
        # Support only for binary segmentation
        labels = torch.where(
            labels > threshold_value,
            torch.ones_like(labels),
            torch.zeros_like(labels),
        )
        return labels
