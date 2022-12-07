from typing import List, Optional

import torch
from torch import Tensor

import preprocessing
from utils.data_classes import BatchData
from preprocess.preprocessor_abstract import PreprocessorAbstract


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

    def validate(self, images: Tensor, labels: Tensor) -> (Tensor, Tensor):
        if images.dim() != 4:
            raise ValueError(
                f"Images batch shape should be (BatchSize x Channels x Width x Height). Got {images.shape}")
        # TODO: Talk with Natan about Errors ?
        if labels.dim() == 3:
            # Probably (B, W, H)
            labels = labels.unsqueeze(1)
        if labels.dim() != 4:
            raise ValueError(
                f"Labels batch shape should be (BatchSize x Channels x Width x Height). Got {labels.shape}")
        if images.shape[1] != self._number_of_channels and images.shape[-1] != self._number_of_channels:
            raise ValueError(
                f"Images should have {self._number_of_channels} number of channels. Got {min(images[0].shape)}")

        # B, W, H, C-> B, C, W, H
        if images.shape[1] != self._number_of_channels:
            images = self.channels_last_to_first(images)
        if labels.shape[1] != self.number_of_classes and labels.shape[1] > 1:
            labels = self.channels_last_to_first(labels)

        self._onehot = labels.shape[1] == self.number_of_classes and self.number_of_classes > 1
        unique_values = torch.unique(labels)
        if not self._onehot:
            if len(unique_values) == self.number_of_classes + 1:
                # Regular
                pass
            elif len(unique_values) > self.number_of_classes + 1:
                # TODO
                print(f'Got {len(unique_values)} > {self.number_of_classes + 1} which is number of classes + 1')
        else:
            if len(unique_values) != 2:
                # TODO
                print(f'Weird, OneHot should have two values only! Got {len(unique_values)} != 2 ({unique_values})')
        if 1 < max(unique_values) < 255:
            labels = labels / 255
            # print(f"Normalizing labels to [0, 1] - Max unique {max(unique_values)}")
        # TODO: Check ignore labels
        # TODO: If channels is not number of classes but its hte number of channels in each label...
        # Ignore labels -> number of classes + 1
        if self._ignore_labels:
            ignore_mask = torch.zeros(labels.size(), dtype=torch.bool)
            ignore_mask = ignore_mask | (labels == (label for label in self._ignore_labels))
            labels[ignore_mask] = self.number_of_classes + 1

        return images, labels

    def preprocess(self, images: Tensor, labels: Tensor) -> BatchData:
        onehot_labels = labels if self._onehot else [preprocessing.get_onehot(label) for label in labels]
        # if True:
        #     for label, image in zip(onehot_labels, images):
        #         temp = preprocessing.get_contours(label, image)
                # break
        onehot_contours = [preprocessing.get_contours(onehot_label) for onehot_label in labels]

        bd = BatchData(images=images,
                       labels=labels,
                       batch_onehot_contours=onehot_contours,
                       batch_onehot_labels=onehot_labels)

        return bd
