import os
import abc
import logging
import traceback
from typing import Iterable, List, Dict, Optional
from itertools import zip_longest
from logging import getLogger
from tqdm import tqdm

from data_gradients.feature_extractors import AbstractFeatureExtractor
from data_gradients.batch_processors.base import BatchProcessor
from data_gradients.feature_extractors.common import SummaryStats
from data_gradients.visualize.seaborn_renderer import SeabornRenderer
from data_gradients.utils.pdf_writer import ResultsContainer, Section, FeatureSummary
from data_gradients.utils.summary_writer import SummaryWriter
from data_gradients.config.data.data_config import DataConfig


logging.basicConfig(level=logging.INFO)

logger = getLogger(__name__)


class AnalysisManagerAbstract(abc.ABC):
    """
    Main dataset analyzer manager abstract class.
    """

    def __init__(
        self,
        *,
        report_title: str,
        data_config: DataConfig,
        train_data: Iterable,
        val_data: Optional[Iterable] = None,
        report_subtitle: Optional[str] = None,
        log_dir: Optional[str] = None,
        batch_processor: BatchProcessor,
        grouped_feature_extractors: Dict[str, List[AbstractFeatureExtractor]],
        batches_early_stop: Optional[int] = None,
    ):
        """
        :param report_title:        Title of the report. Will be used to save the report
        :param report_subtitle:     Subtitle of the report
        :param train_data:          Iterable object contains images and labels of the training dataset
        :param val_data:            Iterable object contains images and labels of the validation dataset
        :param log_dir:             Directory where to save the logs. By default uses the current working directory
        :param batch_processor:     Batch processor object to be used before extracting features
        :param grouped_feature_extractors:  List of feature extractors to be used
        :param id_to_name:          Dictionary mapping class IDs to class names
        :param batches_early_stop:  Maximum number of batches to run in training (early stop)
        """
        self.renderer = SeabornRenderer()
        self.summary_writer = SummaryWriter(report_title=report_title, report_subtitle=report_subtitle, log_dir=log_dir)

        self.data_config_cache_name = f"{self.summary_writer.run_name}.json"
        self.data_config = data_config
        self.data_config.fill_missing_params_with_cache(cache_filename=self.data_config_cache_name)

        # DATA
        if batches_early_stop:
            logger.info(f"Running with `batches_early_stop={batches_early_stop}`: Only the first {batches_early_stop} batches will be analyzed.")
        self.batches_early_stop = batches_early_stop
        self.train_size = len(train_data) if hasattr(train_data, "__len__") else None
        self.val_size = len(val_data) if hasattr(val_data, "__len__") else None

        self.train_iter = iter(train_data)
        self.val_iter = iter(val_data) if val_data is not None else iter([])

        # FEATURES
        self.batch_processor = batch_processor
        self.grouped_feature_extractors = grouped_feature_extractors

        self._train_iters_done = 0
        self._val_iters_done = 0
        self._train_batch_size = None
        self._val_batch_size = None
        self._stopped_early = None

    def execute(self):
        """
        Execute method take batch from train & val data iterables, submit a thread to it and runs the extractors.
        Method finish it work after both train & val iterables are exhausted.
        """

        print(
            f"  - Executing analysis with: \n"
            f"  - batches_early_stop: {self.batches_early_stop} \n"
            f"  - len(train_data): {self.train_size} \n"
            f"  - len(val_data): {self.val_size} \n"
            f"  - log directory: {self.summary_writer.log_dir} \n"
            f"  - Archive directory: {self.summary_writer.archive_dir} \n"
            f"  - feature extractor list: {self.grouped_feature_extractors}"
        )

        datasets_tqdm = tqdm(
            zip_longest(self.train_iter, self.val_iter, fillvalue=None),
            desc="Analyzing... ",
            total=self.n_batches,
        )

        self._train_iters_done, self._val_iters_done = 0, 0
        self._stopped_early = False

        for i, (train_batch, val_batch) in enumerate(datasets_tqdm):

            if i == self.batches_early_stop:
                self._stopped_early = True
                break

            if train_batch is not None:
                for sample in self.batch_processor.process(train_batch, split="train"):
                    for feature_extractors in self.grouped_feature_extractors.values():
                        for feature_extractor in feature_extractors:
                            feature_extractor.update(sample)
                    self._train_iters_done += 1

            if self._train_batch_size is None:
                self._train_batch_size = self._train_iters_done

            if val_batch is not None:
                for sample in self.batch_processor.process(val_batch, split="val"):
                    for feature_extractors in self.grouped_feature_extractors.values():
                        for feature_extractor in feature_extractors:
                            feature_extractor.update(sample)
                    self._val_iters_done += 1

            if self._val_batch_size is None:
                self._val_batch_size = self._val_iters_done

    def post_process(self, interrupted=False):
        """
        Post process method runs on all feature extractors, concurrently on valid and train extractors, send each
        of them a matplotlib ax(es) and gets in return the ax filled with the feature extractor information.
        Then, it logs the information through the logging.
        :return:
        """
        images_created = []

        summary = ResultsContainer()
        for section_name, feature_extractors in tqdm(self.grouped_feature_extractors.items(), desc="Summarizing... "):
            section = Section(section_name)
            for feature_extractor in feature_extractors:
                try:
                    feature = feature_extractor.aggregate()
                    f = self.renderer.render(feature.data, feature.plot_options)
                    feature_json = feature.json
                    feature_error = ""
                except Exception as e:
                    f = None
                    error_description = traceback.format_exception(type(e), e, e.__traceback__)
                    feature_json = {"error": error_description}
                    feature_error = f"Feature extraction error. Check out the log file for more details:<br/>" f"<em>{self.summary_writer.errors_path}</em>"
                    self.summary_writer.add_error(title=feature_extractor.title, error=error_description)

                if f is not None:
                    image_name = feature_extractor.__class__.__name__ + ".png"
                    image_path = os.path.join(self.summary_writer.archive_dir, image_name)
                    f.savefig(image_path, dpi=300)
                    images_created.append(image_path)
                else:
                    image_path = None

                self.summary_writer.add_feature_stats(title=feature_extractor.title, stats=feature_json)

                if feature_error:
                    warning = feature_error
                elif isinstance(feature_extractor, SummaryStats) and (interrupted or (self.batches_early_stop and self._stopped_early)):
                    warning = self._create_samples_iterated_warning()
                else:
                    warning = feature_extractor.warning

                section.add_feature(
                    FeatureSummary(
                        name=feature_extractor.title,
                        description=feature_extractor.description,
                        image_path=image_path,
                        warning=warning,
                        notice=feature_extractor.notice,
                    )
                )
            summary.add_section(section)

        print("Dataset successfully analyzed!")
        print("Starting to write the report, this may take around 10 seconds...")
        self.summary_writer.set_pdf_summary(pdf_summary=summary)
        self.summary_writer.set_data_config(data_config_dict=self.data_config.to_json())
        self.summary_writer.write()

        # Save cache in a specific Folder
        self.data_config.write_to_json(filename=self.data_config_cache_name)

        # Cleanup of generated images
        for image_created in images_created:
            os.remove(image_created)

    def close(self):
        """Safe logging closing"""
        print(f'{"*" * 100}')
        print("We have finished evaluating your dataset!")
        print()
        print("The cache of your DataConfig object can be found in:")
        print(f"    - {os.path.join(self.data_config.DEFAULT_CACHE_DIR, self.data_config_cache_name)}")
        print()
        print("The results can be seen in:")
        print(f"    - {self.summary_writer.log_dir}")
        print(f"    - {self.summary_writer.archive_dir}")

    def run(self):
        """
        Run method activating build, execute, post process and close the manager.
        """
        interrupted = False
        try:
            self.execute()
        except KeyboardInterrupt as e:
            logger.info(
                "[EXECUTION HAS BEEN INTERRUPTED]... "
                "Please wait until SOFT-TERMINATION process finishes and saves the report and log files before terminating..."
            )
            logger.info("For HARD Termination - Stop the process again")
            interrupted = e is not None
        self.post_process(interrupted=interrupted)
        self.close()

    @property
    def n_batches(self) -> Optional[int]:
        """Number of batches to analyze if available, None otherwise."""
        if self.train_size is None or self.val_size is None:
            return self.batches_early_stop

        n_batches_available = max(self.train_size, self.val_size)
        n_batches_early_stop = self.batches_early_stop or float("inf")
        return min(n_batches_early_stop, n_batches_available)

    def _create_samples_iterated_warning(self) -> str:
        if self.train_size is None or self._train_batch_size is None:
            total_train_samples = "unknown amount of "
            portion_train = ""
        else:
            total_train_samples = self.train_size * self._train_batch_size
            portion_train = f" ({self._train_iters_done/total_train_samples:.1%})"

        if self.val_size is None or self._val_batch_size is None:
            total_val_samples = "unknown amount of "
            portion_val = ""

        else:
            total_val_samples = self.val_size * self._val_batch_size
            portion_val = f" ({self._val_iters_done/total_val_samples:.1%})"

        msg_head = "The results presented in this report cover only a subset of the data.\n"
        msg_train = f"Train set: {self._train_iters_done} out of {total_train_samples} samples were analyzed{portion_train}.\n"
        msg_val = f"Validation set: {self._val_iters_done} out of {total_val_samples} samples were analyzed{portion_val}.\n "
        return msg_head + msg_train + msg_val
