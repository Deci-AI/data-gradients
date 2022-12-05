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


debug_mode = False


class AnalysisManager:
    def __init__(self, args):
        self._train_only: bool = True
        self._train_extractors: List[FeatureExtractorBuilder] = []
        self._val_extractors: List[FeatureExtractorBuilder] = []
        self._task: str = args.task
        self._yaml_path: str = args.yaml_path
        self._threads = ThreadPoolExecutor()
        self._logger = TensorBoardLogger()
        self.reformat: Optional[Callable] = None

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

    @staticmethod
    def _validate_segmentation_dataloader(dataloader):
        images, labels = next(iter(dataloader))

        assert images.dim() == 4 and labels.dim() == 4,\
            f"Batch shape should be (BatchSize x Channels x Width x Height). Got: Images - {images.shape}, {labels.shape}"

        # TODO: Check if this is the desired format
        assert labels[0].shape[0] == 1,\
            f"Segmentation label should be (1 x Width x Height). Got: {labels[0].shape}"

        # TODO: Add channels first/last + number of channels
        number_of_channels = 3
        channels_first = True
        assert images[0].shape[0 if channels_first else -1] == number_of_channels,\
            f"Segmentation image should be ({number_of_channels} x Width x Height). Got: {images[0].shape}"

    def _validate_dataloader(self, dataloader):
        if self._task == 'semantic-segmentation':
            self._validate_segmentation_dataloader(dataloader)
        elif self._task == 'object-detection':
            self._validate_detection_dataloader()
        elif self._task == 'classification':
            self._validate_classification_dataloader()
        else:
            raise NotImplementedError(f"Task {self._task} is not implemented!")

    @staticmethod
    def _preprocess(images, labels) -> BatchData:
        onehot_labels = [onehot.get_onehot(label) for label in labels]

        if debug_mode:
            for label, image in zip(onehot_labels, images):
                temp = contours.get_contours(label, image)
                break

        onehot_contours = [contours.get_contours(onehot_label) for onehot_label in onehot_labels]

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

    def execute(self, train_dataloader: DataLoader, val_dataloader: Optional[DataLoader] = None):
        # Validate dataloader
        train_iter = self._get_iter(train_dataloader)
        if val_dataloader is not None:
            self._train_only = False
            valid_iter = self._get_iter(val_dataloader)

        # Check how to get length of iterator
        for train_batch in tqdm.trange(len(train_dataloader)):
            if train_batch > 3 and debug_mode:
                continue

            batch_data = self._get_batch(train_iter)

            for extractor in self._train_extractors:
                if not debug_mode:
                    futures = [self._threads.submit(extractor.execute, batch_data) for extractor in
                               self._train_extractors]
                else:
                    extractor.execute(batch_data)

            if train_batch < len(val_dataloader) and not self._train_only:
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
