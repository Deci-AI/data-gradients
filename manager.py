import concurrent
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional, Callable, Iterable

import yaml
import tqdm
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader

from feature_extractors import FEATURE_EXTRACTORS, FeatureExtractorBuilder
from preprocessing import contours, onehot
from tensorboard_logger import TensorBoardLogger
from batch_data import BatchData


debug_mode = True


class AnalysisManager:
    def __init__(self, args, train_dataloader: DataLoader, val_dataloader: Optional[DataLoader] = None):
        self._train_only: bool = True
        self._train_extractors: List[FeatureExtractorBuilder] = []
        self._val_extractors: List[FeatureExtractorBuilder] = []
        self._task: str = args.task
        self._yaml_path: str = args.yaml_path
        self._threads = ThreadPoolExecutor()
        self._logger = TensorBoardLogger()
        self.reformat: Optional[Callable] = None
        self._train_dataloader = train_dataloader
        self._val_dataloader = val_dataloader
        self._images_channels_last: bool = False
        # TODO: Remove hard code
        self._number_of_channels = 3

    def _get_feature_extractors(self, task: str) -> List:
        with open(self._yaml_path, "r") as stream:
            try:
                for key, value in yaml.safe_load(stream).items():
                    if key == task:
                        return value
            except yaml.YAMLError as exc:
                raise exc
        raise ValueError(f"Did not find right task, arg is {task} and couldn't find it in yaml")

    def build(self):
        fe_list = self._get_feature_extractors(self._task) + self._get_feature_extractors('general')
        for fe in fe_list:
            if isinstance(fe, str):
                self._train_extractors += [FEATURE_EXTRACTORS[fe](True)]
                self._val_extractors += [FEATURE_EXTRACTORS[fe](False)]
            elif isinstance(fe, dict):
                fe, params = next(iter(fe.items()))
                self._train_extractors += [FEATURE_EXTRACTORS[fe](True, params)]
                self._val_extractors += [FEATURE_EXTRACTORS[fe](False, params)]
            else:
                raise ValueError

    @staticmethod
    def _validate_classification_dataloader():
        pass

    @staticmethod
    def _validate_detection_dataloader():
        pass

    def _validate_segmentation_dataloader(self, dataloader):
        images, labels = next(iter(dataloader))

        assert images.dim() == 4,\
            f"Images batch shape should be (BatchSize x Channels x Width x Height). Got {images.shape}"
        assert labels.dim() == 4, \
            f"Labels batch shape should be (BatchSize x Channels x Width x Height). Got {labels.shape}"

        assert images[0].shape[0] == self._number_of_channels or images[0].shape[-1] == self._number_of_channels, \
            f"Images should have {self._number_of_channels} number of channels. Got {min(images[0].shape)}"

        if images[0].shape[0] != self._number_of_channels:
            self._images_channels_last = True

        image_shape = images[0].shape[:2] if self._images_channels_last else images[0].shape[1:]
        labels_shape = labels[0].shape
        n_classes = min(labels[0].shape)
        if labels_shape == (1, *image_shape):
            # TODO: Check for binary
            # TODO: Check for values range
            # TODO: Think of how to save current format
            # TODO: Build reformats if needed
            pass
        elif labels_shape == (*image_shape, 1):
            pass
        elif labels_shape == (n_classes, *image_shape):
            pass
        elif labels_shape == (*image_shape, n_classes):
            pass

    def _validate_dataloader(self, dataloader):
        if self._task == 'semantic-segmentation':
            self._validate_segmentation_dataloader(dataloader)
        elif self._task == 'object-detection':
            self._validate_detection_dataloader()
        elif self._task == 'classification':
            self._validate_classification_dataloader()
        else:
            raise NotImplementedError(f"Task {self._task} is not implemented!")

    def _preprocess(self, images, labels) -> BatchData:
        onehot_labels = [onehot.get_onehot(label) for label in labels]

        if debug_mode:
            for label, image in zip(onehot_labels, images):
                temp = contours.get_contours(label, image)
                break

        onehot_contours = [contours.get_contours(onehot_label) for onehot_label in onehot_labels]

        if self._images_channels_last:
            # TODO: Check that works
            images = images.permute(0, -1, 1, 2)

        bd = BatchData(images=images,
                       labels=labels,
                       batch_onehot_contours=onehot_contours,
                       batch_onehot_labels=onehot_labels)

        return bd

    def _get_batch(self, dataloader) -> BatchData:
        images, labels = next(dataloader)
        if self.reformat is not None:
            images = self.reformat(images)
            labels = self.reformat(labels)
        bd = self._preprocess(images, labels)
        return bd

    def _get_iter(self, dataloader) -> Iterable:
        self._validate_dataloader(dataloader)
        iterable = iter(dataloader)
        return iterable

    def execute(self):
        # Validate dataloader
        train_iter = self._get_iter(self._train_dataloader)
        if self._val_dataloader is not None:
            self._train_only = False
            valid_iter = self._get_iter(self._val_dataloader)

        # Check how to get length of iterator
        for train_batch in tqdm.trange(len(self._train_dataloader)):
            if train_batch > 3 and debug_mode:
                continue

            batch_data = self._get_batch(train_iter)

            for extractor in self._train_extractors:
                if not debug_mode:
                    futures = [self._threads.submit(extractor.execute, batch_data) for extractor in
                               self._train_extractors]
                else:
                    extractor.execute(batch_data)

            if train_batch < len(self._val_dataloader) and not self._train_only:
                batch_data = self._get_batch(valid_iter)
                for extractor in self._val_extractors:
                    if not debug_mode:
                        futures += [self._threads.submit(extractor.execute, batch_data) for extractor in
                                    self._val_extractors]
                    else:
                        extractor.execute(batch_data)

            if not debug_mode:
                # Wait for all threads to finish
                concurrent.futures.wait(futures, return_when=concurrent.futures.ALL_COMPLETED)

    def post_process(self):
        for val_extractor, train_extractor in zip(self._val_extractors, self._train_extractors):
            fig, ax = plt.subplots()

            # First val - because graph params will be overwritten by latest (train) and we want it's params
            if not self._train_only:
                val_extractor.process(ax)

            train_extractor.process(ax)

            fig.tight_layout()

            self._logger.graph_to_tensorboard(val_extractor.__class__.__name__ + "/fig", fig)

    def close(self):
        self._logger.close()

    def run(self):
        self.build()
        self.execute()
        self.post_process()
        self.close()
