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

    def execute(self):
        datasets = [('train', self._train_iter, self._train_extractors)]
        if not self._train_only:
            datasets.append(('val', self._val_iter, self._val_extractors))

        for dataset_name, dataset_iterator, feature_extractors in datasets:
            for batch in tqdm.tqdm(dataset_iterator, desc=f'Working on {dataset_name} dataset'):
                images, labels = self._preprocessor.validate(batch)
                bd: BatchData = self._preprocessor.preprocess(images, labels)
                futures = [self._threads.submit(extractor.execute, bd) for extractor in feature_extractors]
                concurrent.futures.wait(futures, return_when=concurrent.futures.ALL_COMPLETED)

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
