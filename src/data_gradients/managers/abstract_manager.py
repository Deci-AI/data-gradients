import abc
import concurrent
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Iterator, Iterable, Optional, List, Dict, Optional
from itertools import zip_longest
import datetime
import time
from contextlib import contextmanager

import hydra
import tqdm

from data_gradients.feature_extractors import FeatureExtractorAbstract
from data_gradients.logging.logger import Logger
from data_gradients.preprocess.preprocessor_abstract import PreprocessorAbstract
from data_gradients.utils.data_classes.batch_data import BatchData
from data_gradients.utils.common.stopwatch import Stopwatch, Timer

from logging import getLogger

logging.basicConfig(level=logging.WARNING)

logger = getLogger(__name__)


class AnalysisManagerAbstract:
    """
    Main dataset analyzer manager abstract class.
    """

    def __init__(
        self,
        *,
        train_data: Iterable,
        val_data: Optional[Iterable] = None,
        log_writer: Logger,
        id_to_name: Dict,
        batches_early_stop: Optional[int],
        short_run: bool,
    ):

        self._extractors: List[FeatureExtractorAbstract] = []

        short_run = True
        self.batches_early_stop = 1
        # self.train_size = len(train_data) if hasattr(train_data, "__len__") else None
        # self.val_size = len(train_data) if hasattr(val_data, "__len__") else None

        self.train_size = None
        self.val_size = None

        self.train_iter = iter(train_data)
        self.val_iter = iter(val_data) if val_data is not None else iter([])

        # Logger
        self._log_writer = log_writer

        self._cfg = None
        self._preprocessor: PreprocessorAbstract = Optional[None]

        self.id_to_name = id_to_name

        if short_run and self.n_iterations is None:
            logger.warning(
                "`short_run=True` will be ignored because it expects your dataloaders to implement `__len__`, or you to set `early_stop=...`"
            )
            short_run = False
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

    def _preprocess_batch(self, batch: Iterator, split: str) -> BatchData:
        batch = tuple(batch) if isinstance(batch, list) else batch
        images, labels = self._preprocessor.validate(batch)
        preprocessed_batch = self._preprocessor.preprocess(images, labels)
        preprocessed_batch.split = split
        return preprocessed_batch

    def execute(self):
        """
        Execute method take batch from train & val data iterables, submit a thread to it and runs the extractors.
        Method finish it work after both train & val iterables are exhausted.
        """
        thread_manager = ThreadManager()

        datasets_iterator = tqdm.tqdm(
            zip_longest(self.train_iter, self.val_iter, fillvalue=None),
            desc="Analyzing...",
            total=self.n_iterations,
        )

        for i, (train_batch, val_batch) in enumerate(datasets_iterator):

            if i == self.batches_early_stop:
                break

            with timer() as train_batch_timer:
                if train_batch is not None:
                    preprocessed_batch = self._preprocess_batch(train_batch, "train")
                    for extractor in self._extractors:
                        thread_manager.submit(extractor.execute, preprocessed_batch)
                    self._log_writer.visualize(preprocessed_batch)

            # For the first batch, we want to measure the processing time on train/val individually
            if i == 0 and self.short_run:
                thread_manager.wait_complete()

            with timer() as val_batch_timer:
                if val_batch is not None:
                    preprocessed_batch = self._preprocess_batch(val_batch, "val")
                    for extractor in self._extractors:
                        thread_manager.submit(extractor.execute, preprocessed_batch)

            if i == 0 and self.short_run:
                thread_manager.wait_complete()
                self.reevaluate_early_stop(
                    train_batch_timer.elapsed, val_batch_timer.elapsed
                )

    def reevaluate_early_stop(self, train_batch_time: float, val_batch_time: float):
        self.n_iterations
        remaining_time = 0
        remaining_time += (self.n_iterations -1) * (train_batch_time + val_batch_time)

        if self.train_size is not None and self.val_size is not None:
            remaining_time += (self.batches_early_stop -1) * train_batch_time
            remaining_time += (self.batches_early_stop -1) * val_batch_time

        remaining_time = min(remaining_time, max(train_size, val_size))
        total_time = str(datetime.timedelta(seconds=remaining_time))

        print(f"\n\nEstimated time for the whole analyze is {total_time}")
        inp = input(f"Do you want to shorten the amount of data to analyze? [y / n]\n")
        if inp.lower() in ("y", "yes"):
            train_ratio = input("Please provide amount of data to analyze (in %)\n")
            self.batches_early_stop = int(self.train_size * (int(train_ratio) / 100))
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
            extractor.process(self._log_writer, self.id_to_name)

        # Write meta data to json file
        self._log_writer.log_meta_data(self._preprocessor)

        # Write all text data to json file
        self._log_writer.to_json()

    def close(self):
        """
        Safe logging closing
        """
        self._log_writer.close()
        print(
            f'{"*" * 100}'
            f"\nWe have finished evaluating your dataset!"
            f"\nThe results can be seen in {self._log_writer.results_dir()}"
            f"\n\nShow tensorboard by writing in terminal:"
            f"\n\ttensorboard --logdir={os.path.join(os.getcwd(), self._log_writer.results_dir())} --bind_all"
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

    @property
    def n_iterations(self):
        # TODO: check if this is correct
        if not (
            self.train_size is None
            and self.val_size is None
            and self.batches_early_stop is None
        ):
            return min(
                self.batches_early_stop or float("inf"),
                self.train_size or float("inf"),
                self.val_size or float("inf"),
            )


@contextmanager
def timer() -> Iterator[Timer]:
    _timer = Timer()
    yield _timer
    _timer.stop()


class ThreadManager:
    def __init__(self):
        self.thread = ThreadPoolExecutor()
        self.futures = []

    def submit(self, fn, *args):
        self.futures.append(self.thread.submit(fn, *args))

    def wait_complete(self):
        concurrent.futures.wait(
            self.futures, return_when=concurrent.futures.ALL_COMPLETED
        )
