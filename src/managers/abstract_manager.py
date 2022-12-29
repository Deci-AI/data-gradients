import concurrent
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Iterator, Iterable, Optional, List

import hydra
import tqdm

from src.feature_extractors import FeatureExtractorAbstract
from src.logger.json_logger import JsonLogger
from src.logger.tensorboard_logger import TensorBoardLogger
from src.preprocess import PreprocessorAbstract
from src.utils import BatchData


class AnalysisManagerAbstract:
    """
    Main dataset analyzer manager abstract class.
    """

    def __init__(self, train_data: Iterable,
                 val_data: Optional[Iterable],
                 task: str,
                 samples_to_visualize: int,
                 id_to_name):

        self._extractors: List[FeatureExtractorAbstract] = []

        self._threads = ThreadPoolExecutor()

        self._dataset_size = len(train_data) if hasattr(train_data, '__len__') else None
        # Users Data Iterator
        self._train_iter: Iterator = train_data if isinstance(train_data, Iterator) else iter(train_data)
        if val_data is not None:
            self._train_only = False
            self._val_iter: Iterator = val_data if isinstance(val_data, Iterator) else iter(val_data)

        else:
            self._train_only = True
            self._val_iter = None

        # Logger
        self._loggers = {'TB': TensorBoardLogger(iter(train_data), samples_to_visualize),
                         'JSON': JsonLogger()}

        self._preprocessor: PreprocessorAbstract = Optional[None]
        self._cfg = None

        self._task = task
        self.id_to_name = id_to_name

    def build(self):
        """
        Build method for hydra configuration file initialized and composed in manager constructor.
        Create lists of feature extractors, both to train and val iterables.
        """
        cfg = hydra.utils.instantiate(self._cfg)
        self._extractors = cfg.common + cfg[self._task]

    def visualize(self):
        self._loggers['TB'].visualize()

    def _get_batch(self, data_iterator: Iterator) -> BatchData:
        """
        Iterates iterable, get a Tuple out of it, validate format and preprocess due to task preprocessor.
        :param data_iterator: Iterable for getting next item out of it
        :return: BatchData object, holding images, labels and preprocessed objects in accordance to task
        """
        batch = next(data_iterator)
        batch = tuple(batch) if isinstance(batch, list) else batch

        images, labels = self._preprocessor.validate(batch)

        bd = self._preprocessor.preprocess(images, labels)
        return bd

    def execute(self):
        """
        Execute method take batch from train & val data iterables, submit a thread to it and runs the extractors.
        Method finish it work after both train & val iterables are exhausted.
        """
        pbar = tqdm.tqdm(desc='Working on batch # ', total=self._dataset_size)
        train_batch = 0
        val_batch_data = None
        while True:
            try:
                train_batch_data = self._get_batch(self._train_iter)
                train_batch_data.split = 'train'
            except StopIteration:
                break
            # Try to get val batch
            if not self._train_only:
                try:
                    val_batch_data = self._get_batch(self._val_iter)
                    val_batch_data.split = 'val'
                except StopIteration:
                    self._train_only = True

            # Run threads
            futures = [self._threads.submit(extractor.execute, train_batch_data) for extractor in
                       self._extractors]

            if not self._train_only:
                futures = [self._threads.submit(extractor.execute, val_batch_data) for extractor in
                           self._extractors]

            concurrent.futures.wait(futures, return_when=concurrent.futures.ALL_COMPLETED)

            pbar.update()
            train_batch += 1

    def post_process(self):
        """
        Post process method runs on all feature extractors, concurrently on valid and train extractors, send each
        of them a matplotlib ax(es) and gets in return the ax filled with the feature extractor information.
        Then, it logs the information through the logger.
        :return:
        """
        for extractor in self._extractors:
            extractor.process(self._loggers, self.id_to_name)

    def close(self):
        """
        Safe logger closing
        """
        [self._loggers[logger].close() for logger in self._loggers.keys()]
        print(f'{"*" * 100}'
              f'\nWe have finished evaluating your dataset!'
              f'\nThe results can be seen in {list(self._loggers.values())[0].logdir}'
              f'\n\nShow tensorboard by writing in terminal:'
              f'\n\ttensorboard --logdir={os.path.join(os.getcwd() ,list(self._loggers.values())[0].logdir)} --bind_all'
              f'\n')

    def run(self):
        """
        Run method activating build, execute, post process and close the manager.
        """
        self.build()
        self.visualize()
        self.execute()
        self.post_process()
        self.close()
