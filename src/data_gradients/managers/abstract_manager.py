import os
import abc
import logging
import json
from typing import Iterable, List, Dict, Optional
from itertools import zip_longest
from logging import getLogger
from datetime import datetime
import tqdm

from data_gradients.feature_extractors import AbstractFeatureExtractor
from data_gradients.batch_processors.base import BatchProcessor
from data_gradients.visualize.seaborn_renderer import SeabornRenderer

from data_gradients.utils.pdf_writer import ResultsContainer, Section, FeatureSummary, PDFWriter, assets
from data_gradients.config.interactive_config import DataConfig

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
        data_config: DataConfig,
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
        # Static parameters
        if log_dir is None:
            log_dir = os.path.join(os.getcwd(), "logs", report_title.replace(" ", "_"))
            logger.info(f"`log_dir` was not set, so the logs will be saved in {log_dir}")

        session_id = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.log_dir = log_dir  # Main logging directory. Latest run results will be saved here.
        self.archive_dir = os.path.join(log_dir, "archive_" + session_id)  # A duplicate of the results will be saved here as well.

        self.report_title = report_title
        self.report_subtitle = report_subtitle or datetime.strftime(datetime.now(), "%m:%H %B %d, %Y")

        # WRITERS
        self.renderer = SeabornRenderer()
        self.pdf_writer = PDFWriter(title=report_title, subtitle=report_subtitle, html_template=assets.html.doc_template)
        self.data_config = data_config

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
            f"  - log directory: {self.log_dir} \n"
            f"  - Archive directory: {self.archive_dir} \n"
            f"  - feature extractor list: {self.grouped_feature_extractors}"
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

                # Save in the main directory and in the archive directory
                self.write_json(data=dict(title=feature_extractor.title, data=feature.json), output_dir=self.log_dir, filename="stats.json")
                self.write_json(data=dict(title=feature_extractor.title, data=feature.json), output_dir=self.archive_dir, filename="stats.json")

                f = self.renderer.render(feature.data, feature.plot_options)
                if f is not None:
                    image_name = feature_extractor.__class__.__name__ + ".png"
                    image_path = os.path.join(self.archive_dir, image_name)
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

        # Save in the main directory and in the archive directory
        self.pdf_writer.write(results_container=summary, output_filename=os.path.join(self.log_dir, "Report.pdf"))
        self.pdf_writer.write(results_container=summary, output_filename=os.path.join(self.archive_dir, "Report.pdf"))

        # Cleanup of generated images
        for image_created in images_created:
            os.remove(image_created)

    def close(self):
        """Safe logging closing"""
        print(f'{"*" * 100}')
        print("We have finished evaluating your dataset!")
        print("The results can be seen in:")
        print(f"    - {self.log_dir}")
        print(f"    - {self.archive_dir}")

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

    @staticmethod
    def write_json(data: Dict, output_dir: str, filename: str):
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, filename)
        with open(output_path, "a") as f:
            json.dump(data, f, indent=4)
