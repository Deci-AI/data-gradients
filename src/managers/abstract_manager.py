import concurrent
from concurrent.futures import ThreadPoolExecutor
from typing import Iterator, Iterable, Optional, Union, List

import hydra
import tqdm
from matplotlib import pyplot as plt

from src.feature_extractors import FeatureExtractorAbstract
from src.logger.tensorboard_logger import TensorBoardLogger
from src.preprocess import PreprocessorAbstract
from src.utils import BatchData


class AnalysisManagerAbstract:
    def __init__(self, train_data: Union[Iterable, Iterator],
                 val_data: Optional[Union[Iterable, Iterator]],
                 task: str):

        self._train_extractors: List[FeatureExtractorAbstract] = []
        self._val_extractors: List[FeatureExtractorAbstract] = []

        self._threads = ThreadPoolExecutor()

        # Users Data Iterators
        self._train_iter: Iterator = train_data if isinstance(train_data, Iterator) else iter(train_data)
        if val_data is not None:
            self._train_only = False
            self._val_iter: Iterator = val_data if isinstance(val_data, Iterator) else iter(val_data)

        else:
            self._train_only = True
            self._val_iter = None

        # Logger
        self._loggers = [TensorBoardLogger(), JsonLogger()]

        self._preprocessor: PreprocessorAbstract = Optional[None]
        self._cfg = None

        self._task = task

    def build(self):
        cfg = hydra.utils.instantiate(self._cfg)
        self._train_extractors = cfg.common + cfg[self._task]
        # Create another instances for same classes
        cfg = hydra.utils.instantiate(self._cfg)
        self._val_extractors = cfg.common + cfg[self._task]

    def _get_batch(self, data_iterator) -> BatchData:
        batch = next(data_iterator)
        batch = tuple(batch) if isinstance(batch, list) else batch

        images, labels = self._preprocessor.validate(batch)

        bd = self._preprocessor.preprocess(images, labels)
        return bd

    def execute(self):
        pbar = tqdm.tqdm(desc='Working on batch #')
        train_batch = 0

        while True:
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
            val_hist = val_extractor.process(axes['val'], train=False)

            train_hist = train_extractor.process(axes['train'], train=True)

            fig.tight_layout()

            for logger in self._loggers:
                title = val_extractor.__class__.__name__
                logger.log(title, fig if isinstance(logger, TensorBoardLogger) else [train_hist, val_hist])

    def close(self):
        self._logger.close()

    def run(self):
        self.build()
        self.execute()
        self.post_process()
        self.close()
