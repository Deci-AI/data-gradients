import abc
import concurrent
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Iterator, Iterable, Optional, List, Dict, Optional

import hydra
import tqdm

from data_gradients.feature_extractors import FeatureExtractorAbstract
from data_gradients.logging.logger import Logger
from data_gradients.preprocess.preprocessor_abstract import PreprocessorAbstract
from data_gradients.utils.data_classes.batch_data import BatchData
from data_gradients.utils.common.stopwatch import Stopwatch


logger = logging.getLogger(__name__)


class AnalysisManagerAbstract(abc.ABC):
    """
    Main dataset analyzer manager abstract class.
    """

    def __init__(
        self,
        *,
        train_data: Iterable,
        val_data: Optional[Iterable],
        logger: Logger,
        id_to_name: Dict,
        batches_early_stop: Optional[int],
        short_run: bool,
    ):
        """
        :param train_data:          Iterable object contains images and labels of the training dataset
        :param val_data:            Iterable object contains images and labels of the validation dataset
        :param logger:              Logger object for logging information during analysis
        :param id_to_name:          Dictionary mapping class IDs to class names
        :param batches_early_stop:  Maximum number of batches to run in training (early stop)
        :param short_run:           Flag indicating whether to run for a single epoch first to estimate total duration,
                                    before choosing the number of epochs.
        """
        self._extractors: List[FeatureExtractorAbstract] = []

        self._threads = ThreadPoolExecutor()

        self._train_dataset_size = len(train_data) if hasattr(train_data, "__len__") else None
        self._val_dataset_size = len(val_data) if hasattr(val_data, "__len__") else None
        # Users Data Iterator
        self._train_iter: Iterator = train_data if isinstance(train_data, Iterator) else iter(train_data)
        if val_data is not None:
            self._train_only = False
            self._val_iter: Iterator = val_data if isinstance(val_data, Iterator) else iter(val_data)

        else:
            self._train_only = True
            self._val_iter = None

        # Logger
        self._logger = logger

        self._preprocessor: PreprocessorAbstract = Optional[None]
        self._cfg = None

        self.id_to_name = id_to_name

        self.sw: Optional[Stopwatch] = None
        self.batches_early_stop = batches_early_stop
        self.short_run = short_run

    @abc.abstractmethod
    def _create_logger(self) -> Logger:
        raise NotImplementedError

    def build(self):
        """
        Build method for hydra configuration file initialized and composed in manager constructor.
        Create lists of feature extractors, both to train and val iterables.
        """
        cfg = hydra.utils.instantiate(self._cfg)
        self._extractors = cfg.feature_extractors + cfg.common.feature_extractors

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
        pbar = tqdm.tqdm(desc="Analyzing...", total=self._train_dataset_size)
        train_batch = 0
        val_batch_data = None
        self.sw = Stopwatch()
        while True:
            # Try to get train batch
            if train_batch > self.batches_early_stop:
                break
            try:
                train_batch_data = self._get_batch(self._train_iter)
                train_batch_data.split = "train"
                self._logger.visualize(train_batch_data)  # maybe there's a better place to put this?
                self.sw.tick()
            except StopIteration:
                break
            # Try to get val batch
            if not self._train_only:
                try:
                    val_batch_data = self._get_batch(self._val_iter)
                    val_batch_data.split = "val"
                    self.sw.tick()
                except StopIteration:
                    self._train_only = True

            # Run threads
            futures = [self._threads.submit(extractor.execute, train_batch_data) for extractor in self._extractors]

            if not self._train_only:
                futures = [self._threads.submit(extractor.execute, val_batch_data) for extractor in self._extractors]

            concurrent.futures.wait(futures, return_when=concurrent.futures.ALL_COMPLETED)
            self.sw.tick()

            if train_batch < 1 and self.short_run:
                self.measure()

            pbar.update()
            train_batch += 1

    def measure(self):
        total_time = self.sw.estimate_total_time(self._train_dataset_size, self._val_dataset_size)
        print(f"\n\nEstimated time for the whole analyze is {total_time}")
        inp = input(f"Do you want to shorten the amount of data to analyze? [y / n]\n")
        if inp == "y":
            inp = input("Please provide amount of data to analyze [%]\n")
            self.batches_early_stop = int(self._train_dataset_size * (int(inp) / 100))
            print(f"Running for {self.batches_early_stop} batches!")

    def post_process(self):
        """
        Post process method runs on all feature extractors, concurrently on valid and train extractors, send each
        of them a matplotlib ax(es) and gets in return the ax filled with the feature extractor information.
        Then, it logs the information through the logging.
        :return:
        """

        # Post process each feature executor to json / tensorboard
        for extractor in self._extractors:
            extractor.process(self._logger, self.id_to_name)

        # Write meta data to json file
        self._logger.log_meta_data(self._preprocessor)

        # Write all text data to json file
        self._logger.to_json()

    def close(self):
        """
        Safe logging closing
        """
        self._logger.close()
        print(
            f'{"*" * 100}'
            f"\nWe have finished evaluating your dataset!"
            f"\nThe results can be seen in {self._logger.results_dir()}"
            f"\n\nShow tensorboard by writing in terminal:"
            f"\n\ttensorboard --logdir={os.path.join(os.getcwd(), self._logger.results_dir())} --bind_all"
            f"\n"
        )

    def run(self):
        """
        Run method activating build, execute, post process and close the manager.
        """
        self.build()
        self.execute()
        self.post_process()
        self.close()
