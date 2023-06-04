import os
import abc
import logging
from typing import Iterable, List, Dict, Optional
from itertools import zip_longest
from logging import getLogger

import tqdm

from data_gradients.feature_extractors import AbstractFeatureExtractor
from data_gradients.logging.log_writer import LogWriter
from data_gradients.batch_processors.base import BatchProcessor
from data_gradients.visualize.image_samplers.base import ImageSampleManager
from data_gradients.visualize.seaborn_renderer import SeabornRenderer

from data_gradients.utils.pdf_writer import ResultsContainer, Section, FeatureSummary, PDFWriter, assets

logging.basicConfig(level=logging.WARNING)

logger = getLogger(__name__)


class AnalysisManagerAbstract(abc.ABC):
    """
    Main dataset analyzer manager abstract class.
    """

    def __init__(
        self,
        *,
        report_title: str,
        train_data: Iterable,
        val_data: Optional[Iterable] = None,
        report_subtitle: Optional[str] = None,
        log_dir: Optional[str] = None,
        batch_processor: BatchProcessor,
        feature_extractors: List[AbstractFeatureExtractor],
        id_to_name: Dict,
        batches_early_stop: Optional[int] = None,
        image_sample_manager: ImageSampleManager,
    ):
        """
        :param report_title:        Title of the report. Will be used to save the report
        :param report_subtitle:     Subtitle of the report
        :param train_data:          Iterable object contains images and labels of the training dataset
        :param val_data:            Iterable object contains images and labels of the validation dataset
        :param log_dir:             Directory where to save the logs. By default uses the current working directory
        :param batch_processor:     Batch processor object to be used before extracting features
        :param feature_extractors:  List of feature extractors to be used
        :param id_to_name:          Dictionary mapping class IDs to class names
        :param batches_early_stop:  Maximum number of batches to run in training (early stop)
        :param image_sample_manager:     Object responsible for collecting images
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
        from datetime import datetime

        report_subtitle = report_subtitle or datetime.strftime(datetime.now(), "%m:%H %B %d, %Y")
        self.html_writer = PDFWriter(title=report_title, subtitle=report_subtitle, html_template=assets.html.doc_template)
        self._log_writer = LogWriter(log_dir=log_dir)
        self.output_folder = self._log_writer.log_dir

        self.batch_processor = batch_processor
        self.feature_extractors = feature_extractors

        self.id_to_name = id_to_name

        self.image_sample_manager = image_sample_manager

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
            f"feature extractor list: {self.feature_extractors}"
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
                    self.image_sample_manager.update(sample)
                    for extractor in self.feature_extractors:
                        extractor.update(sample)

            if val_batch is not None:
                for sample in self.batch_processor.process(val_batch, split="val"):
                    for extractor in self.feature_extractors:
                        extractor.update(sample)

    def post_process(self):
        """
        Post process method runs on all feature extractors, concurrently on valid and train extractors, send each
        of them a matplotlib ax(es) and gets in return the ax filled with the feature extractor information.
        Then, it logs the information through the logging.
        :return:
        """

        summary = ResultsContainer()
        section = Section("Features")  # TODO: add section title for each section
        for feature_extractor in self.feature_extractors:
            feature = feature_extractor.aggregate()

            self._log_writer.log_json(title=feature_extractor.title, data=feature.json)

            f = self.renderer.render(feature.data, feature.plot_options)
            image_name = feature_extractor.__class__.__name__ + ".png"
            image_path = os.path.join(self.output_folder, image_name)
            f.savefig(image_path)

            section.add_feature(
                FeatureSummary(
                    name=feature_extractor.title,
                    description=feature_extractor.description,
                    image_path=image_path,
                )
            )
        summary.add_section(section)

        formatted_tite = self.report_title.lower().replace(" ", "_")
        output_path = os.path.join(self.output_folder, f"{formatted_tite}.pdf")
        logger.info(f"Writing the result of the Data Analysis into: {output_path}")
        self.html_writer.write(results_container=summary, output_filename=output_path)

        # TODO: add images to the report...
        for i, sample_to_visualize in enumerate(self.image_sample_manager.samples):
            title = f"Data Visualization/{len(self.image_sample_manager.samples) - i}"
            self._log_writer.log_image(title=title, image=sample_to_visualize)

        if self.batch_processor.images_route is not None:
            self._log_writer.log_json(title="Get images out of dictionary", data=self.batch_processor.images_route)
        if self.batch_processor.labels_route is not None:
            self._log_writer.log_json(title="Get labels out of dictionary", data=self.batch_processor.labels_route)

        # Write all text data to json file
        self._log_writer.save_as_json()

    def close(self):
        """Safe logging closing"""
        self._log_writer.close()
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
