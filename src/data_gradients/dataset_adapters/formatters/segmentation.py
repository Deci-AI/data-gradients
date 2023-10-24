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
            classes_to_ignore = set(self.data_config.get_class_names()) - set(self.data_config.get_class_names_to_use())
            self.class_ids_to_ignore = [self.data_config.get_class_names().index(class_name_to_ignore) for class_name_to_ignore in classes_to_ignore]

        if not self.check_is_batch(images=images, labels=labels):
            images = images.unsqueeze(0)
            labels = labels.unsqueeze(0)
        # We have images/labels of shape (BS, ...)

        images = ensure_channel_first(images, n_image_channels=self.get_n_image_channels(images=images))
        images = check_images_shape(images, n_image_channels=self.get_n_image_channels(images=images))

        if 0 <= images.min() and images.max() <= 1:
            images *= 255
            images = images.to(torch.uint8)
        elif images.min() < 0:  # images were normalized with some unknown mean and std
            images -= images.min()
            images /= images.max()
            images *= 255
            images = images.to(torch.uint8)

        # Here, We have labels of shape (BS, ...)
        labels = labels.squeeze()  # Avoid dim of size 1
        if labels.ndim == 3:
            # Case if classes are floats (e.g. [0.2, 0.6, 0.8] -> [1, 3, 4] because it was normalized by factor of 5
            labels = self.ensure_hard_labels(labels, n_classes=self.data_config.get_n_classes(), threshold_value=self.threshold_value)
        elif labels.ndim == 4:
            n_classes = self.data_config.get_n_classes()
            # if (BS, C, H, W), -> (BS, H, W, C)
            if labels.shape[1] == n_classes:
                labels.permute(0, 3, 1, 2)

            # To categorical (BS, H, W, C), -> (BS, H, W)
            labels = torch.argmax(labels, dim=1)
        # We have labels of shape (Bs, H, W)

        for class_id_to_ignore in self.class_ids_to_ignore:
            labels[labels == class_id_to_ignore] = -1  # Set the ignored class labels to -1 or any other value indicating 'ignore'.

        return images, labels

    def check_is_batch(self, images: Tensor, labels: Tensor) -> bool:
        if images.ndim == 4 or labels.ndim == 4:
            # if less any dim is 4, we know it's a batch
            self.data_config.is_batch = True
            return self.data_config.is_batch
        elif images.ndim == 2 or labels.ndim == 2:
            # If image or mask only includes 2 dims, we can guess it's a single sample
            self.data_config.is_batch = False
            return self.data_config.is_batch
        else:
            # Otherwise, we need to ask the user
            hint = f"    - Image shape: {images.shape}\n    - Mask shape:  {labels.shape}"
            return self.data_config.get_is_batch(hint=hint)

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
    def binary_mask_above_threshold(labels: Tensor, threshold_value: float) -> Tensor:
        # Support only for binary segmentation
        labels = torch.where(
            labels > threshold_value,
            torch.ones_like(labels),
            torch.zeros_like(labels),
        )
        return labels
