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
        batches_early_stop: Optional[int] = None,
        short_run: bool = False,
    ):
        self._extractors: List[FeatureExtractorAbstract] = []

        if batches_early_stop:
            logger.info(
                "Running with `batches_early_stop={batches_early_stop}`: Only the first {batches_early_stop} batches will be analyzed."
            )
        self.batches_early_stop = batches_early_stop
        self.train_size = len(train_data) if hasattr(train_data, "__len__") else None
        self.val_size = len(train_data) if hasattr(val_data, "__len__") else None

        self.train_iter = iter(train_data)
        self.val_iter = iter(val_data) if val_data is not None else iter([])

        # Logger
        self._log_writer = log_writer

        self._cfg = None
        self._preprocessor: PreprocessorAbstract = Optional[None]

        self.id_to_name = id_to_name

        if short_run and self.n_batches is None:
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
        datasets_tqdm = tqdm.tqdm(
            zip_longest(self.train_iter, self.val_iter, fillvalue=None),
            desc="Analyzing... ",
            total=self.n_batches,
        )

        for i, (train_batch, val_batch) in enumerate(datasets_tqdm):

            if i == self.batches_early_stop:
                break

            if train_batch is not None:
                preprocessed_batch = self._preprocess_batch(train_batch, "train")
                for extractor in self._extractors:
                    thread_manager.submit(extractor.execute, preprocessed_batch)
                self._log_writer.visualize(preprocessed_batch)

            if val_batch is not None:
                preprocessed_batch = self._preprocess_batch(val_batch, "val")
                for extractor in self._extractors:
                    thread_manager.submit(extractor.execute, preprocessed_batch)

            if i == 0 and self.short_run:
                thread_manager.wait_complete()
                datasets_tqdm.refresh()
                single_batch_duration = datasets_tqdm.format_dict["elapsed"]
                self.reevaluate_early_stop(
                    remaining_time=(self.n_batches - 1) * single_batch_duration
                )

    def reevaluate_early_stop(self, remaining_time: float) -> None:
        """Give option to the user to reevaluate the early stop criteria.

        :param remaining_time: Time remaining for the whole analyze."""

        print(
            f"\nEstimated remaining time for the whole analyze is {remaining_time} (1/{self.n_batches} done)"
        )
        inp = input(
            f"Do you want to shorten the amount of data to analyze? (Yes/No) : "
        )
        if inp.lower() in ("y", "yes"):
            early_stop_ratio_100 = input(
                "What percentage of the remaining data do you want to process? (0-100) : "
            )
            early_stop_ratio = float(early_stop_ratio_100) / 100
            remaining_batches = self.n_batches - 1
            self.batches_early_stop = int(remaining_batches * early_stop_ratio + 1)
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
    def n_batches(self) -> Optional[int]:
        """Number of batches to analyze if available, None otherwise."""
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
