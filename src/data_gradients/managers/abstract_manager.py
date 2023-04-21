import abc
import logging
from typing import Iterator, Iterable, List, Dict, Optional
from itertools import zip_longest
from logging import getLogger

import tqdm

from data_gradients.feature_extractors import FeatureExtractorAbstract
from data_gradients.logging.log_writer import LogWriter
from data_gradients.preprocess.preprocessor_abstract import PreprocessorAbstract
from data_gradients.utils.data_classes.batch_data import BatchData
from data_gradients.utils.thread_manager import ThreadManager
from data_gradients.visualize.image_visualizer import ImageVisualizer

logging.basicConfig(level=logging.WARNING)

logger = getLogger(__name__)


class AnalysisManagerAbstract(abc.ABC):
    """
    Main dataset analyzer manager abstract class.
    """

    def __init__(
        self,
        *,
        train_data: Iterable,
        val_data: Optional[Iterable] = None,
        log_dir: Optional[str] = None,
        preprocessor: PreprocessorAbstract,
        extractors: List[FeatureExtractorAbstract],
        id_to_name: Dict,
        batches_early_stop: Optional[int] = None,
        short_run: bool = False,
        visualizer: ImageVisualizer,
    ):
        """
        :param train_data:          Iterable object contains images and labels of the training dataset
        :param val_data:            Iterable object contains images and labels of the validation dataset
        :param log_dir:             Directory where to save the logs. By default uses the current working directory
        :param preprocessor:        Preprocessor object to be used before extracting features
        :param extractors:          List of feature extractors to be used
        :param id_to_name:          Dictionary mapping class IDs to class names
        :param batches_early_stop:  Maximum number of batches to run in training (early stop)
        :param short_run:           Flag indicating whether to run for a single epoch first to estimate total duration,
                                    before choosing the number of epochs.
        :param visualizer:          Visualizer object to be used for visualizing images.
        """

        if batches_early_stop:
            logger.info(f"Running with `batches_early_stop={batches_early_stop}`: Only the first {batches_early_stop} batches will be analyzed.")
        self.batches_early_stop = batches_early_stop
        self.train_size = len(train_data) if hasattr(train_data, "__len__") else None
        self.val_size = len(train_data) if hasattr(val_data, "__len__") else None

        self.train_iter = iter(train_data)
        self.val_iter = iter(val_data) if val_data is not None else iter([])

        # Logger
        self._log_writer = LogWriter(log_dir=log_dir)

        self.preprocessor = preprocessor
        self.extractors = extractors

        self.id_to_name = id_to_name

        if short_run and self.n_batches is None:
            logger.warning("`short_run=True` will be ignored because it expects your dataloaders to implement `__len__`, or you to set `early_stop=...`")
            short_run = False
        self.short_run = short_run
        self.visualizer = visualizer

    def _preprocess_batch(self, batch: Iterator, split: str) -> BatchData:
        batch = tuple(batch) if isinstance(batch, list) else batch
        images, labels = self.preprocessor.validate(batch)
        preprocessed_batch = self.preprocessor.preprocess(images, labels)
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
                for extractor in self.extractors:
                    thread_manager.submit(extractor.update, preprocessed_batch)
                self.visualizer.update(preprocessed_batch)

            if val_batch is not None:
                preprocessed_batch = self._preprocess_batch(val_batch, "val")
                for extractor in self.extractors:
                    thread_manager.submit(extractor.update, preprocessed_batch)

            if i == 0 and self.short_run:
                thread_manager.wait_complete()
                datasets_tqdm.refresh()
                single_batch_duration = datasets_tqdm.format_dict["elapsed"]
                self.reevaluate_early_stop(remaining_time=(self.n_batches - 1) * single_batch_duration)

    def reevaluate_early_stop(self, remaining_time: float) -> None:
        """Give option to the user to reevaluate the early stop criteria.

        :param remaining_time: Time remaining for the whole analyze."""

        print(f"\nEstimated remaining time for the whole analyze is {remaining_time} (1/{self.n_batches} done)")
        inp = input("Do you want to shorten the amount of data to analyze? (Yes/No) : ")
        if inp.lower() in ("y", "yes"):
            early_stop_ratio_100 = input("What percentage of the remaining data do you want to process? (0-100) : ")
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
        for extractor in self.extractors:
            extractor.aggregate_and_write(self._log_writer, self.id_to_name)

        for i, sample_to_visualize in enumerate(self.visualizer.samples):
            title = f"Data Visualization/{len(self.visualizer.samples) - i}"
            self._log_writer.log_image(title=title, image=sample_to_visualize)

        # Write meta data to json file
        self._log_writer.log_meta_data(image_route=self.preprocessor.images_route, labels_route=self.preprocessor.labels_route)

        # Write all text data to json file
        self._log_writer.to_json()

    def close(self):
        """
        Safe logging closing
        """
        self._log_writer.close()

    def run(self):
        """
        Run method activating build, execute, post process and close the manager.
        """
        self.execute()
        self.post_process()
        self.close()

    @property
    def n_batches(self) -> Optional[int]:
        """Number of batches to analyze if available, None otherwise."""
        if not (self.train_size is None and self.val_size is None and self.batches_early_stop is None):
            return min(
                self.batches_early_stop or float("inf"),
                self.train_size or float("inf"),
                self.val_size or float("inf"),
            )
