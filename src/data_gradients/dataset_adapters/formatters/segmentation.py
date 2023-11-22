from typing import Optional, List, Tuple

import torch
from torch import Tensor

from data_gradients.dataset_adapters.formatters.base import BatchFormatter
from data_gradients.dataset_adapters.utils import check_all_integers
from data_gradients.dataset_adapters.config.data_config import SegmentationDataConfig
from data_gradients.dataset_adapters.formatters.utils import DatasetFormatError, check_images_shape, ensure_channel_first


class SegmentationBatchFormatter(BatchFormatter):
    """
    Segmentation formatter class
    """

    def __init__(
        self,
        data_config: SegmentationDataConfig,
        threshold_value: float,
        ignore_labels: Optional[List[int]] = None,
    ):
        """
        :param threshold_value:     Threshold
        :param ignore_labels:       Numbers that we should avoid from analyzing as valid classes, such as background
        """
        self.class_ids_to_ignore: Optional[List[str]] = None  # This will be initialized in `format()`
        self.ignore_labels = ignore_labels or []

        self.threshold_value = threshold_value
        self.is_input_soft_label = None
        self.data_config = data_config
        super().__init__(data_config=data_config)

    def format(self, images: Tensor, labels: Tensor) -> Tuple[Tensor, Tensor]:
        """Validate batch images and labels format, and ensure that they are in the relevant format for segmentation.

        :param images: Batch of images, in (BS, ...) format, or single sample
        :param labels: Batch of labels, in (BS, ...) format, or single sample
        :return:
            - images: Batch of images already formatted into (BS, C, H, W)
            - labels: Batch of labels already formatted into (BS, H, W) - categorical representation
        """

        if self.class_ids_to_ignore is None:
            # This may trigger questions to the user, so we prefer to set it inside `former()` and not `__init__`
            # to avoid asking questions even before the analysis starts.
            classes_to_ignore = set(self.data_config.get_class_names().values()) - set(self.data_config.get_class_names_to_use())
            self.class_ids_to_ignore = []
            for class_id, class_name in self.data_config.get_class_names().items():
                for class_name_to_ignore in classes_to_ignore:
                    if class_name == class_name_to_ignore:
                        self.class_ids_to_ignore.append(class_name)

        if not self.check_is_batch(images=images, labels=labels):
            images = images.unsqueeze(0)
            labels = labels.unsqueeze(0)

        images = self._format_images(images)
        labels = self._format_labels(labels)

        for class_id_to_ignore in self.class_ids_to_ignore:
            labels[labels == class_id_to_ignore] = -1

        return images, labels

    def check_is_batch(self, images: Tensor, labels: Tensor) -> bool:
        if images.ndim == 4 or labels.ndim == 4:
            self.data_config.is_batch = True
        elif images.ndim == 2 or labels.ndim == 2:
            self.data_config.is_batch = False
        else:
            hint = f"Image shape: {images.shape}\nMask shape: {labels.shape}"
            self.data_config.is_batch = self.data_config.get_is_batch(hint=hint)
        return self.data_config.is_batch

    def _format_images(self, images: Tensor) -> Tensor:
        images = ensure_channel_first(images, n_image_channels=self.get_n_image_channels(images=images))
        images = check_images_shape(images, n_image_channels=self.get_n_image_channels(images=images))
        images = adjust_image_values(images)
        return images

    def _format_labels(self, labels: Tensor) -> Tensor:
        labels = labels.squeeze(1).squeeze(-1)  # If (BS, 1, H, W) or (BS, H, W, 1) -> (BS, H, W)
        if labels.ndim == 3:
            labels = ensure_hard_labels(labels, n_classes=self.data_config.get_n_classes(), threshold_value=self.threshold_value)
        elif labels.ndim == 4:
            labels = convert_to_categorical(labels, n_classes=self.data_config.get_n_classes())
        else:
            raise DatasetFormatError(f"Labels should be either 3D (categorical) or 4D (onehot), but got {labels.ndim}D")
        return labels


def adjust_image_values(images: Tensor) -> Tensor:
    if 0 <= images.min() and images.max() <= 1:
        return (images * 255).to(torch.uint8)
    elif images.min() < 0:
        images = (images - images.min()) / images.max() * 255
        return images.to(torch.uint8)
    return images


def ensure_hard_labels(labels: Tensor, n_classes: int, threshold_value: float) -> Tensor:
    unique_values = torch.unique(labels)
    if check_all_integers(unique_values):
        return labels
    elif 0 <= min(unique_values) and max(unique_values) <= 1 and check_all_integers(unique_values * 255):
        return labels * 255
    elif n_classes == 1:
        return binary_mask_above_threshold(labels, threshold_value)
    raise DatasetFormatError(f"Not supporting soft-labeling for number of classes > 1! Got {n_classes} classes.")


def convert_to_categorical(labels: Tensor, n_classes: int) -> Tensor:
    if labels.shape[1] == n_classes:
        labels = labels.permute(0, 2, 3, 1)  # (BS, C, H, W) -> (BS, H, W, C)
    return torch.argmax(labels, dim=-1)


def binary_mask_above_threshold(labels: Tensor, threshold_value: float) -> Tensor:
    return torch.where(labels > threshold_value, torch.ones_like(labels), torch.zeros_like(labels))
