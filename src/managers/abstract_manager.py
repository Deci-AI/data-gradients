from abc import abstractmethod
from concurrent.futures import ThreadPoolExecutor
from typing import Iterator, Iterable, Optional, Union, List

import hydra
from matplotlib import pyplot as plt

from src.feature_extractors import FeatureExtractorAbstract
from src.logger.tensorboard_logger import TensorBoardLogger
from src.preprocess import PreprocessorAbstract
from src.utils import BatchData


class AnalysisManagerAbstract:
    def __init__(self, train_data: Union[Iterable, Iterator],
                 val_data: Optional[Union[Iterable, Iterator]] = None):

        self._train_extractors: List[FeatureExtractorAbstract] = []
        self._val_extractors: List[FeatureExtractorAbstract] = []

        self._threads = ThreadPoolExecutor()

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
        self._preprocessor: PreprocessorAbstract = Optional[None]
        self._cfg = None
        self._task = ""

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

    @abstractmethod
    def execute(self):
        pass

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
