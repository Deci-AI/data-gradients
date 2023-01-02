from typing import List, Optional, Tuple, Sequence

import torch
from torch import Tensor

from src.utils import SegBatchData
from src.preprocess import PreprocessorAbstract, squeeze_by_class
from src.preprocess import contours


class SegmentationPreprocessor(PreprocessorAbstract):
    """
    Segmentation preprocessor class
    """
    def __init__(self, num_classes, ignore_labels):
        """
        Constructor gets number of classes and ignore labels in order to understand how to data labels should look like
        :param num_classes: number of valid classes
        :param ignore_labels: list of numbers that we should avoid from analyzing as valid classes, such as background
        """
        super().__init__(num_classes)
        self._onehot: bool = False
        self._ignore_labels: List[int] = ignore_labels if ignore_labels is not None else [0]

    @property
    def ignore_labels(self) -> List[int]:
        return self._ignore_labels

    def _type_validate(self, objs):
        """
        Required: A Tuple (Sequence) with length 2, representing (images, labels).
        If any of (images, labels) are not Tensors, convert them.
        :param objs: output of next(iterator)
        :return: (images, labels) as Tuple[Tensor, Tensor]
        """
        if isinstance(objs, Sequence):
            if len(objs) == 2:
                images = objs[0] if isinstance(objs[0], torch.Tensor) else self._to_tensor(objs[0], 'first')
                labels = objs[1] if isinstance(objs[1], torch.Tensor) else self._to_tensor(objs[1], 'second')
            else:
                raise NotImplementedError
        elif isinstance(objs, dict):
            raise NotImplementedError
        else:
            raise NotImplementedError
        return images, labels

    def _dim_validate_images(self, images: Tensor):
        """
        Validating images dimensions are (BS, Channels, W, H)
        :param images: Tensor [BS, C, W, H]
        :return: images: Tensor [BS, C, W, H]
        """
        if images.dim() != 4:
            raise ValueError(
                f"Images batch shape should be (BatchSize x Channels x Width x Height). Got {images.shape}")

        if images.shape[1] != self._number_of_channels and images.shape[-1] != self._number_of_channels:
            raise ValueError(
                f"Images should have {self._number_of_channels} number of channels. Got {min(images[0].shape)}")
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

        if labels.dim() != 4:
            raise ValueError(
                f"Labels batch shape should be [BatchSize x Channels x Width x Height]. Got {labels.shape}")
        if labels.shape[1] != 1 and labels.shape[1] != self.number_of_classes:
            raise ValueError(
                f"Labels batch shape should be [BS, N, W, H] where N is either 1 or num_classes"
                f" ({self.number_of_classes}). Got: {labels.shape[1]}")

        return labels

    @staticmethod
    def _normalize_validate(labels: Tensor):
        """
        Pixel values for labels are representing class id's, hence they are in the range of [0, 255] or normalized
        in [0, 1] representing (1/255, 2/255, ...).
        Soft labels (continues in [0, 1] are not supported yet.
        :param labels: Tensor [BS, N, W, H]
        :return: labels: Tensor [BS, N, W, H]
        """
        unique_values = torch.unique(labels)
        if 0 <= min(unique_values) and max(unique_values) < 1:
            # TODO:
            #  If resize uses BiLinear, I'll get here with an error message
            if any(u not in range(0, 255) for u in unique_values * 255):
                raise NotImplementedError("Values were not matching for integer numbers even after inverse"
                                          "normalization.\nYou might using resize transformation with bilinear "
                                          "interpolation mode - please change it to 'nearest'")
            labels = labels * 255
        elif any(unique_values < 0) or max(unique_values) > 255:
            raise ValueError("Labels pixel-values should be either floats in [0, 1] or integers in [0, 255]")
        return labels

    def _channels_first_validate_images(self, images: Tensor):
        """
        Images should be [BS, C, W, H]. If [BS, W, H, C], permute
        :param images: Tensor
        :return: images: Tensor [BS, C, W, H]
        """
        if images.shape[1] != self._number_of_channels and images.shape[-1] == self._number_of_channels:
            images = self.channels_last_to_first(images)
        return images

    def _channels_first_validate_labels(self, labels: Tensor):
        """
        Labels should be [BS, N, W, H]. If [BS, W, H, N], permute
        :param labels: Tensor
        :return: labels: Tensor [BS, N, W, H]
        """
        # B, W, H, C-> B, C, W, H
        if labels.shape[1] != self.number_of_classes and labels.shape[1] > 1:
            labels = self.channels_last_to_first(labels)
        return labels

    def _pixel_values_validate(self, values: Tensor):
        """
        Inner-use validation method. Unique values of pixels in the labels tensor.
        Will be deprecated!
        :param values: Tensor of unique pixel values of labels
        """
        if not self._onehot:
            if len(values) == self.number_of_classes + 1:
                # Regular
                pass
            elif len(values) > self.number_of_classes + 1:
                pass
                # print(f'Got {len(values)} > {self.number_of_classes + 1} which is number of classes + 1')
        else:
            if len(values) != 2:
                print(f'Weird, OneHot should have two values only! Got {len(values)} != 2 ({values})')

    @staticmethod
    def _nan_validate(objects):
        # TODO: Keep that nan indices in mind
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

        labels = self._normalize_validate(labels)

        self._onehot = labels.shape[1] == self.number_of_classes and self.number_of_classes > 1
        self._pixel_values_validate(torch.unique(labels))

        return images, labels

    def _remove_ignore_labels(self, labels):
        """
        For user-given ignore labels, put zeros in the specific ignore labels representation, either a whole channel in
        the one-hot case or only pixels in the one channel annotation case
        :param labels: Tensor with unique values of all classes, including ignored ones.
        :return: Tensor with no representation of the ignored classes (all are zeros).
        """
        if 0 not in self._ignore_labels:
            return labels

        for ignore_label in self.ignore_labels:
            if self._onehot:
                # TODO: Check it out (Right channel (ignore level channel) should be ignore label values
                # Turn specific channel into zeros
                labels[:, ignore_label, ...] = 0
            else:
                # Get all labels in the only channel that are ignored and turn them into zeros
                labels = torch.where((labels > 0) & (labels == ignore_label), torch.zeros_like(labels), labels)
        return labels

    def preprocess(self, images: Tensor, labels: Tensor) -> SegBatchData:
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
        labels = self._remove_ignore_labels(labels)

        labels = [squeeze_by_class.squeeze_by_classes(label,
                                                      is_one_hot=self._onehot,
                                                      ignore_labels=self.ignore_labels) for label in labels]

        # TODO: Debug convexity things
        # contours.debug_convexity_things(labels, images)
        # exit(0)
        all_contours = [contours.get_contours(onehot_label) for onehot_label in labels]

        # bbox_areas = contours.get_bbox_area(all_contours)
        sbd = SegBatchData(images=images,
                           labels=labels,
                           contours=all_contours,
                           # bbox_areas=bbox_areas,
                           split="")

        return sbd
