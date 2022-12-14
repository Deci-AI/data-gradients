from typing import List, Optional, Tuple, Sequence

import torch
from torch import Tensor

import preprocess
from utils.data_classes import BatchData
from preprocess import PreprocessorAbstract, contours, squeeze_by_class


class SegmentationPreprocessor(PreprocessorAbstract):
    def __init__(self):
        super().__init__()
        self._onehot: bool = False
        self._ignore_labels: List[int] = []

    @property
    def ignore_labels(self) -> List[int]:
        return self._ignore_labels

    @ignore_labels.setter
    def ignore_labels(self, ignore_labels: Optional[List[int]]):
        self._ignore_labels = ignore_labels if ignore_labels is not None else []

    def _type_validate(self, objs):
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

    def _dim_validate_images(self, images):
        if images.dim() != 4:
            raise ValueError(
                f"Images batch shape should be (BatchSize x Channels x Width x Height). Got {images.shape}")

        if images.shape[1] != self._number_of_channels and images.shape[-1] != self._number_of_channels:
            raise ValueError(
                f"Images should have {self._number_of_channels} number of channels. Got {min(images[0].shape)}")
        return images

    def _dim_validate_labels(self, labels):
        if labels.dim() == 3:
            # Probably (B, W, H)
            labels = labels.unsqueeze(1)

        if labels.dim() != 4:
            raise ValueError(
                f"Labels batch shape should be (BatchSize x Channels x Width x Height). Got {labels.shape}")
        return labels

    @staticmethod
    def _normalize_validate(labels):
        unique_values = torch.unique(labels)
        if 0 <= min(unique_values) and max(unique_values) < 1:
            labels = labels * 255
        elif any(unique_values < 0) or max(unique_values) > 255:
            raise ValueError("Labels pixel-values should be either floats in [0, 1] or integers in [0, 255]")
        return labels

    def _channels_first_validate_images(self, images):
        if images.shape[1] != self._number_of_channels:
            images = self.channels_last_to_first(images)
        return images

    def _channels_first_validate_labels(self, labels):
        # B, W, H, C-> B, C, W, H
        if labels.shape[1] != self.number_of_classes and labels.shape[1] > 1:
            labels = self.channels_last_to_first(labels)
        return labels

    def _pixel_values_validate(self, values):
        # TODO: All the below is weird
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

    def validate(self, objects: Optional[Tuple]) -> Tuple[Tensor, Tensor]:
        images, labels = self._type_validate(objects)

        images = self._dim_validate_images(images)
        labels = self._dim_validate_labels(labels)

        images = self._channels_first_validate_images(images)
        labels = self._channels_first_validate_labels(labels)

        labels = self._normalize_validate(labels)

        self._onehot = labels.shape[1] == self.number_of_classes and self.number_of_classes > 1
        self._pixel_values_validate(torch.unique(labels))

        return images, labels

    def _remove_ignore_labels(self, labels):
        for ignore_label in self.ignore_labels:
            if self._onehot:
                # Turn specific channel into zeros
                labels[:, ignore_label, ...] = 0
            else:
                # Get all labels in the only channel that are ignored and turn them into zeros
                labels = torch.where((labels > 0) & (labels == ignore_label), torch.zeros_like(labels), labels)
        return labels

    def preprocess(self, images: Tensor, labels: Tensor) -> BatchData:
        labels = self._remove_ignore_labels(labels)

        labels = [squeeze_by_class.squeeze_by_classes(label, is_one_hot=self._onehot) for label in labels]

        if False:
            for label, image in zip(labels, images):
                temp = preprocess.get_contours(label, image)

        all_contours = [contours.get_contours(onehot_label) for onehot_label in labels]

        bd = BatchData(images=images,
                       labels=labels,
                       contours=all_contours)

        return bd
