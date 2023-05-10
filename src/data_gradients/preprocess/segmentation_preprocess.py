from typing import List, Optional, Tuple

import torch
from torch import Tensor

from data_gradients.utils import SegmentationBatchData
from data_gradients.preprocess import PreprocessorAbstract
from data_gradients.preprocess import contours


class SegmentationPreprocessor(PreprocessorAbstract):
    """
    Segmentation preprocessor class
    """

    def __init__(
        self,
        num_classes,
        ignore_labels,
        images_extractor,
        labels_extractor,
        num_image_channels,
        threshold_value,
    ):
        """
        Constructor gets number of classes and ignore labels in order to understand how to data labels should look like
        :param num_classes: number of valid classes
        :param ignore_labels: list of numbers that we should avoid from analyzing as valid classes, such as background
        """
        super().__init__(num_classes, images_extractor, labels_extractor, num_image_channels)
        self._onehot: bool = False

        self._ignore_labels: List[int] = ignore_labels
        self.threshold_value = threshold_value
        self._soft_labels = False
        self._binary = False

    @property
    def ignore_labels(self) -> List[int]:
        return self._ignore_labels if self._ignore_labels is not None else []

    def _type_validate(self, objs):
        """
        Required: A Tuple (Sequence) with length 2, representing (images, labels).
        If any of (images, labels) are not Tensors, convert them.
        :param objs: output of next(iterator)
        :return: (images, labels) as Tuple[Tensor, Tensor]
        """
        if isinstance(objs, Tuple) or isinstance(objs, List):
            if len(objs) == 2:
                images = objs[0] if isinstance(objs[0], torch.Tensor) else self._to_tensor(objs[0], "first")
                labels = objs[1] if isinstance(objs[1], torch.Tensor) else self._to_tensor(objs[1], "second")
            else:
                raise NotImplementedError(f"Got tuple/list object with length {len(objs)}! Supporting only len == 2")
        elif isinstance(objs, dict):
            images = self._handle_dict(objs, "first")
            labels = self._handle_dict(objs, "second")
        else:
            raise NotImplementedError(f"Got object {type(objs)} from Iterator - supporting dict, tuples and lists Only!")
        return images, labels

    def _dim_validate_images(self, images: Tensor):
        """
        Validating images dimensions are (BS, Channels, W, H)
        :param images: Tensor [BS, C, W, H]
        :return: images: Tensor [BS, C, W, H]
        """
        if images.dim() != 4:
            raise ValueError(f"Images batch shape should be (BatchSize x Channels x Width x Height). Got {images.shape}")

        if images.shape[1] != self._num_image_channels and images.shape[-1] != self._num_image_channels:
            raise ValueError(f"Images should have {self._num_image_channels} number of channels. Got {min(images[0].shape)}")
        return images

    def _dim_validate_labels(self, labels: Tensor):
        """
        Validating labels dimensions are (BS, N, W, H) where N is either 1 or number of valid classes
        :param labels: Tensor [BS, N, W, H]
        :return: labels: Tensor [BS, N, W, H]
        """
        if labels.dim() == 3:
            # Probably (B, W, H)
            labels = labels.unsqueeze(1)
            return labels

        if labels.dim() != 4:
            raise ValueError(f"Labels batch shape should be [BatchSize x Channels x Width x Height]. Got {labels.shape}")

        valid = [self.number_of_classes + len(self.ignore_labels), 1]
        if labels.shape[1] not in valid and labels.shape[-1] not in valid:
            raise ValueError(
                f"Labels batch shape should be [BS, N, W, H] where N is either 1 or num_classes + len(ignore_labels)"
                f" ({self.number_of_classes + len(self.ignore_labels)}). Got: {labels.shape[1]}"
            )

        return labels

    def _normalize_validate(self, labels: Tensor):
        """
        Pixel values for labels are representing class id's, hence they are in the range of [0, 255] or normalized
        in [0, 1] representing (1/255, 2/255, ...).
        :param labels: Tensor [BS, N, W, H]
        :return: labels: Tensor [BS, N, W, H]
        """
        unique_values = torch.unique(labels)

        if self._check_all_integers(unique_values):
            pass
        elif 0 <= min(unique_values) and max(unique_values) <= 1 and self._check_all_integers(unique_values * 255):
            labels = labels * 255
        else:
            print(f"\nFound Soft labels! There are {len(unique_values)} unique values! max is: {max(unique_values)}," f" min is {min(unique_values)}")
            print(f"Thresholding to [0, 1] with threshold value {self.threshold_value}")
            if self.number_of_classes > 1:
                raise NotImplementedError(
                    "Not supporting soft-labeling for number of classes > 1! "
                    f"Got {self.number_of_classes} # classes,"
                    f" while ignore labels are {self.ignore_labels}."
                )
            self._soft_labels = True
            labels = self._thresh(labels)
        return labels

    def _thresh(self, labels: Tensor) -> Tensor:
        # Support only for binary segmentation
        labels = torch.where(
            labels > self.threshold_value,
            torch.ones_like(labels),
            torch.zeros_like(labels),
        )
        return labels

    def _channels_first_validate_images(self, images: Tensor):
        """
        Images should be [BS, C, W, H]. If [BS, W, H, C], permute
        :param images: Tensor
        :return: images: Tensor [BS, C, W, H]
        """
        if images.shape[1] != self._num_image_channels and images.shape[-1] == self._num_image_channels:
            images = self.channels_last_to_first(images)
        return images

    def _channels_first_validate_labels(self, labels: Tensor):
        """
        Labels should be [BS, N, W, H]. If [BS, W, H, N], permute
        :param labels: Tensor
        :return: labels: Tensor [BS, N, W, H]
        """
        # B, W, H, C-> B, C, W, H
        if labels.shape[1] not in [self.number_of_classes + len(self.ignore_labels), 1]:
            labels = self.channels_last_to_first(labels)
        return labels

    @staticmethod
    def _nan_validate(objects):
        nans = torch.isnan(objects)
        if nans.any():
            nan_indices = set(nans.nonzero()[:, 0].tolist())
            all_indices = set(i for i in range(objects.shape[0]))
            valid_indices = all_indices - nan_indices
            return objects[valid_indices]
        return objects

    def validate(self, objects: Optional[Tuple]) -> Tuple[Tensor, Tensor]:
        """
        Validating object came out of next() method activated on the iterator.
        Check & Fix length, type, dimensions, channels-first, pixel values and checks if onehot
        :param objects: Tuple from next(Iterator)
        :return: images, labels as Tuple[Tensor, Tensor] with shape [[BS, C, W, H], [BS, N, W, H]]
        """
        images, labels = self._type_validate(objects)

        images = self._nan_validate(images)
        labels = self._nan_validate(labels)

        images = self._dim_validate_images(images)
        labels = self._dim_validate_labels(labels)

        images = self._channels_first_validate_images(images)
        labels = self._channels_first_validate_labels(labels)

        if self._soft_labels:
            labels = self._thresh(labels)

        labels = self._normalize_validate(labels)

        self._binary = self.number_of_classes == 1
        self._onehot = labels.shape[1] == (self.number_of_classes + len(self.ignore_labels)) and not self._binary

        return images, labels

    def preprocess(self, images: Tensor, labels: Tensor) -> SegmentationBatchData:
        """
        Preprocess method gets images and labels tensors and returns a segmentation dedicated data class.
        Images are tensor with [BS, C, W, H], Labels are without ignore labels representation and with the
        squeeze-by-class format, which means every VALID class gets a designated channel, with the unique values of
        <class-num> and 0.
        <class-num> pixel says "this is object of class <class-num>", 0 says "no object of class <class-num>".
        :param images: Tensor
        :param labels: Tensor
        :return: SegBatchData
        """
        # To One Hot
        if not self._binary and not self._onehot:
            labels = self._to_one_hot(labels)

        # Remove ignore label
        for ignore_label in self.ignore_labels:
            labels[:, ignore_label, ...] = torch.zeros_like(labels[:, ignore_label, ...])

        all_contours = [contours.get_contours(onehot_label) for onehot_label in labels]

        sbd = SegmentationBatchData(images=images, labels=labels, contours=all_contours, split="")

        return sbd

    def _to_one_hot(self, labels: torch.Tensor) -> torch.Tensor:
        """
        Method gets label with the shape of [BS, N, W, H] where N is either 1 or num_classes, if is_one_hot=True.
        param label: Tensor
        param is_one_hot: Determine if labels are one-hot shaped
        :return: Labels tensor shaped as [BS, VC, W, H] where VC is Valid Classes only - ignores are omitted.
        """
        masks = []
        labels = labels.to(torch.int64)

        for label in labels:
            label = torch.nn.functional.one_hot(label, self.number_of_classes + len(self.ignore_labels))
            masks.append(label)
        labels = torch.concat(masks, dim=0).permute(0, -1, 1, 2)

        return labels
