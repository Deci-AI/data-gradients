import concurrent
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional, Iterator, Iterable, Union

import hydra
import tqdm
import numpy as np
import torch
from matplotlib import pyplot as plt

from feature_extractors import FeatureExtractorAbstract
from preprocess.preprocessor_abstract import PreprocessorAbstract
from logger.tensorboard_logger import TensorBoardLogger
from utils.data_classes import BatchData

debug_mode = False


class AnalysisManager:
    def __init__(self, cfg,
                 train_data: Union[Iterable, Iterator],
                 val_data: Optional[Union[Iterable, Iterator]] = None):
        self._train_extractors: List[FeatureExtractorAbstract] = []
        self._val_extractors: List[FeatureExtractorAbstract] = []
        self._threads = ThreadPoolExecutor()

        self.cfg = cfg

        # Users Data Iterators
        self._train_iter: Iterator = train_data if isinstance(train_data, Iterator) else iter(train_data)
        if val_data is not None:
            self._train_only = False
            self._val_iter: Iterator = val_data if isinstance(train_data, Iterator) else iter(train_data)
        else:
            self._train_only = True
            self._val_iter = None

        # Logger
        self._logger = TensorBoardLogger()

        # Task Data Preprocessor
        self._preprocessor: PreprocessorAbstract = PreprocessorAbstract.get_preprocessor(cfg.task)
        self._preprocessor.number_of_classes = cfg.number_of_classes
        self._preprocessor.ignore_labels = cfg.ignore_labels

    def build(self):
        cfg = hydra.utils.instantiate(self.cfg)
        self._train_extractors = cfg.common + cfg[cfg.task]
        # Create another instances for same classes
        cfg = hydra.utils.instantiate(self.cfg)
        self._val_extractors = cfg.common + cfg[cfg.task]

    def _get_batch(self, data_iterator) -> BatchData:
        batch = next(data_iterator)
        batch = tuple(batch) if isinstance(batch, list) else batch

        # # TODO: Check dictionaries
        # images, labels = batch
        # new_labels_dict = dict()
        # new_labels_dict['all_labels'] = dict()
        # new_labels_dict['all_labels']['not_good_torch_labels'] = torch.zeros((1, 1))
        # new_labels_dict['all_labels']['not_good_np_labels'] = np.zeros_like(labels)
        # new_labels_dict['all_labels']['good_torch_labels'] = labels
        # new_labels_dict['something_other_then_labels'] = np.zeros_like(images)
        # images_dict = dict()
        # images_dict['only_images'] = images
        # batch = images_dict, new_labels_dict

        images, labels = self._preprocessor.validate(batch)

        bd = self._preprocessor.preprocess(images, labels)
        return bd

    def execute(self):
        pbar = tqdm.tqdm(desc='Working on batch #')
        train_batch = 0

        while True:
            if train_batch > 2:
                break
            try:
                batch_data = self._get_batch(self._train_iter)
            except StopIteration:
                break
            else:
                futures = [self._threads.submit(extractor.execute, batch_data) for extractor in
                           self._train_extractors]

            if not self._train_only:
                try:
                    batch_data = self._get_batch(self._val_iter)
                except StopIteration:
                    self._train_only = True
                else:
                    futures += [self._threads.submit(extractor.execute, batch_data) for extractor in
                                self._val_extractors]

            # Wait for all threads to finish
            concurrent.futures.wait(futures, return_when=concurrent.futures.ALL_COMPLETED)

            pbar.update()
            train_batch += 1

    def post_process(self):
        for val_extractor, train_extractor in zip(self._val_extractors, self._train_extractors):
            axes = dict()
            if train_extractor.single_axis:
                fig, ax = plt.subplots(figsize=(10, 5))
                axes['train'] = axes['val'] = ax
            else:
                fig, ax = plt.subplots(1, 2, figsize=(10, 5))
                axes['train'], axes['val'] = ax

            # First val - because graph params will be overwritten by latest (train) and we want it's params
            val_extractor.process(axes['val'], train=False)

            train_extractor.process(axes['train'], train=True)

            fig.tight_layout()

            self._logger.log_graph(val_extractor.__class__.__name__ + "/fig", fig)

    def close(self):
        self._logger.close()

    def run(self):
        self.build()
        self.execute()
        self.post_process()
        self.close()
