import os
import abc
import logging
from typing import Iterable, List, Dict, Optional
from itertools import zip_longest
from logging import getLogger
from datetime import datetime
import tqdm

from data_gradients.feature_extractors import AbstractFeatureExtractor
from data_gradients.logging.log_writer import LogWriter
from data_gradients.batch_processors.base import BatchProcessor
from data_gradients.visualize.seaborn_renderer import SeabornRenderer

from data_gradients.utils.pdf_writer import ResultsContainer, Section, FeatureSummary, PDFWriter, assets
from data_gradients.config.interactive_config import BaseInteractiveConfig

logging.basicConfig(level=logging.WARNING)

logger = getLogger(__name__)

from line_profiler_pycharm import profile

class AnalysisManagerAbstract(abc.ABC):
    """
    Main dataset analyzer manager abstract class.
    """

    def __init__(
        self,
        *,
        report_title: str,
        config: BaseInteractiveConfig,
        train_data: Iterable,
        val_data: Optional[Iterable] = None,
        report_subtitle: Optional[str] = None,
        log_dir: Optional[str] = None,
        batch_processor: BatchProcessor,
        grouped_feature_extractors: Dict[str, List[AbstractFeatureExtractor]],
        id_to_name: Dict,
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

        if batches_early_stop:
            logger.info(f"Running with `batches_early_stop={batches_early_stop}`: Only the first {batches_early_stop} batches will be analyzed.")
        self.batches_early_stop = batches_early_stop
        self.train_size = len(train_data) if hasattr(train_data, "__len__") else None
        self.val_size = len(val_data) if hasattr(val_data, "__len__") else None

        self.train_iter = iter(train_data)
        self.val_iter = iter(val_data) if val_data is not None else iter([])

        self.renderer = SeabornRenderer()
        self.report_title = report_title
        self.config = config

        report_subtitle = report_subtitle or datetime.strftime(datetime.now(), "%m:%H %B %d, %Y")
        self.html_writer = PDFWriter(title=report_title, subtitle=report_subtitle, html_template=assets.html.doc_template)
        self._log_writer = LogWriter(log_dir=log_dir)
        self.output_folder = self._log_writer.log_dir

        self.batch_processor = batch_processor
        self.grouped_feature_extractors = grouped_feature_extractors

        self.id_to_name = id_to_name

    @profile
    def execute(self):
        """
        Execute method take batch from train & val data iterables, submit a thread to it and runs the extractors.
        Method finish it work after both train & val iterables are exhausted.
        """

        print(
            f"Executing analysis with: \n"
            f"batches_early_stop: {self.batches_early_stop} \n"
            f"len(train_data): {self.train_size} \n"
            f"len(val_data): {self.val_size} \n"
            f"log directory: {self._log_writer.log_dir} \n"
            f"feature extractor list: {self.grouped_feature_extractors}"
        )

        datasets_tqdm = tqdm.tqdm(
            zip_longest(self.train_iter, self.val_iter, fillvalue=None),
            desc="Analyzing... ",
            total=self.n_batches,
        )

        for i, (train_batch, val_batch) in enumerate(datasets_tqdm):

            if i == self.batches_early_stop:
                break

            if train_batch is not None:
                for sample in self.batch_processor.process(train_batch, split="train"):
                    for feature_extractors in self.grouped_feature_extractors.values():
                        for feature_extractor in feature_extractors:
                            feature_extractor.update(sample)

            if val_batch is not None:
                for sample in self.batch_processor.process(val_batch, split="val"):
                    for feature_extractors in self.grouped_feature_extractors.values():
                        for feature_extractor in feature_extractors:
                            feature_extractor.update(sample)

    def post_process(self):
        """
        Post process method runs on all feature extractors, concurrently on valid and train extractors, send each
        of them a matplotlib ax(es) and gets in return the ax filled with the feature extractor information.
        Then, it logs the information through the logging.
        :return:
        """
        images_created = []

        summary = ResultsContainer()
        for section_name, feature_extractors in self.grouped_feature_extractors.items():
            section = Section(section_name)
            for feature_extractor in feature_extractors:
                feature = feature_extractor.aggregate()

                self._log_writer.log_json(title=feature_extractor.title, data=feature.json)

                f = self.renderer.render(feature.data, feature.plot_options)
                if f is not None:
                    image_name = feature_extractor.__class__.__name__ + ".png"
                    image_path = os.path.join(self.output_folder, image_name)
                    f.savefig(image_path)
                    images_created.append(image_path)
                else:
                    image_path = None

                section.add_feature(
                    FeatureSummary(
                        name=feature_extractor.title,
                        description=feature_extractor.description,
                        image_path=image_path,
                    )
                )
            summary.add_section(section)

        output_path = os.path.join(self.output_folder, "report.pdf")
        logger.info(f"Writing the result of the Data Analysis into: {output_path}")
        self.html_writer.write(results_container=summary, output_filename=output_path)

        # Cleanup of generated images
        for image_created in images_created:
            os.remove(image_created)

        # Write all text data to json file
        self._log_writer.save_as_json()

    def close(self):
        """Safe logging closing"""
        self._log_writer.close()
        self.config.save_cache()
        print(f'{"*" * 100}' f"\nWe have finished evaluating your dataset!" f"\nThe results can be seen in {self.output_folder}" f"\n")

    def run(self):
        """
        Run method activating build, execute, post process and close the manager.
        """
        try:
            self.execute()
        except KeyboardInterrupt:
            logger.info(
                "[EXECUTION HAS BEEN INTERRUPTED]... "
                "Please wait until SOFT-TERMINATION process finishes and saves the report and log files before terminating..."
            )
            logger.info("For HARD Termination - Stop the process again")
        self.post_process()
        self.close()

    @property
    def n_batches(self) -> Optional[int]:
        """Number of batches to analyze if available, None otherwise."""
        if self.train_size is None or self.val_size is None:
            return self.batches_early_stop

        n_batches_available = max(self.train_size, self.val_size)
        n_batches_early_stop = self.batches_early_stop or float("inf")
        return min(n_batches_early_stop, n_batches_available)
