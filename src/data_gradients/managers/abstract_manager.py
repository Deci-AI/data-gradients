import os
import abc
import logging
import traceback
from typing import List, Dict, Optional, Iterable, Sized
from itertools import zip_longest
from logging import getLogger

from tqdm import tqdm

from data_gradients.dataset_adapters.config.typing_utils import SupportedDataType
from data_gradients.feature_extractors import AbstractFeatureExtractor
from data_gradients.feature_extractors.common import SummaryStats
from data_gradients.utils.utils import print_in_box
from data_gradients.visualize.seaborn_renderer import SeabornRenderer
from data_gradients.utils.pdf_writer import ResultsContainer, Section, FeatureSummary
from data_gradients.utils.summary_writer import SummaryWriter
from data_gradients.sample_preprocessor.base_sample_preprocessor import AbstractSamplePreprocessor

logging.basicConfig(level=logging.INFO)

logger = getLogger(__name__)


class AnalysisManagerAbstract(abc.ABC):
    """
    Main dataset analyzer manager abstract class.
    """

    def __init__(
        self,
        *,
        train_data: Iterable[SupportedDataType],
        val_data: Optional[Iterable[SupportedDataType]],
        sample_preprocessor: AbstractSamplePreprocessor,
        summary_writer: SummaryWriter,
        grouped_feature_extractors: Dict[str, List[AbstractFeatureExtractor]],
        batches_early_stop: Optional[int] = None,
        remove_plots_after_report: Optional[bool] = True,
    ):
        """
        :param train_data:                  Iterable object contains images and labels of the training dataset
        :param val_data:                    Iterable object contains images and labels of the validation dataset
        :param grouped_feature_extractors:  List of feature extractors to be used
        :param batches_early_stop:          Maximum number of batches to run in training (early stop)
        :param remove_plots_after_report:   Delete the plots from the report directory after the report is generated. By default, True
        """

        self.renderer = SeabornRenderer()
        self.summary_writer = summary_writer
        self.data_config = sample_preprocessor.data_config

        # DATA
        if batches_early_stop:
            logger.info(f"Running with `batches_early_stop={batches_early_stop}`: Only the first {batches_early_stop} batches will be analyzed.")
        self.batches_early_stop = batches_early_stop

        val_data = val_data or iter([])
        self.train_size = len(train_data) if isinstance(train_data, Sized) else None
        self.val_size = len(val_data) if isinstance(val_data, Sized) else None

        self.train_samples_iterator = sample_preprocessor.preprocess_samples(train_data, split="train")
        self.val_samples_iterator = sample_preprocessor.preprocess_samples(val_data, split="val")

        # FEATURES
        self.grouped_feature_extractors = grouped_feature_extractors
        self._remove_plots_after_report = remove_plots_after_report
        for _, grouped_feature_list in self.grouped_feature_extractors.items():
            for feature_extractor in grouped_feature_list:
                feature_extractor.setup_data_sources(train_data, val_data)
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

        print_in_box(
            "To better understand how to tackle the data issues highlighted in this report, explore our comprehensive course on analyzing "
            "computer vision datasets. click here: https://hubs.ly/Q01XpHBT0"
        )

        datasets_tqdm = tqdm(
            zip_longest(self.train_samples_iterator, self.val_samples_iterator, fillvalue=None),
            desc="Analyzing... ",
            total=self.n_batches,
        )

        self._train_iters_done, self._val_iters_done = 0, 0
        self._stopped_early = False

        for i, (train_sample, val_sample) in enumerate(datasets_tqdm):

            if i == self.batches_early_stop:
                self._stopped_early = True
                break

            if train_sample is not None:
                for feature_extractors in self.grouped_feature_extractors.values():
                    for feature_extractor in feature_extractors:
                        feature_extractor.update(train_sample)
                self._train_iters_done += 1

            if self._train_batch_size is None:
                self._train_batch_size = self._train_iters_done

            if val_sample is not None:
                for feature_extractors in self.grouped_feature_extractors.values():
                    for feature_extractor in feature_extractors:
                        feature_extractor.update(val_sample)
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
                    logger.error(f"Feature extractor {feature_extractor} error: {error_description}")

                if f is not None:
                    image_name = feature_extractor.__class__.__name__ + ".png"
                    image_path = os.path.join(self.summary_writer.archive_dir, image_name)
                    f.savefig(image_path, dpi=200)
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
                        description=self._format_feature_description(feature_extractor.description),
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

        # Cleanup of generated images
        if self._remove_plots_after_report:
            for image_created in images_created:
                os.remove(image_created)

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

        self.data_config.dump_cache_file()

        self.print_summary()

    def print_summary(self):
        print()
        print(f'{"=" * 100}')
        print("Your dataset evaluation has been completed!")
        print()
        print(f'{"-" * 100}')
        print("Training Configuration...")
        print(self.data_config.get_caching_info())
        print()
        print(f'{"-" * 100}')
        print("Report Location:")
        print("    - Temporary Folder (will be overwritten next run):")
        print(f"        └─ {self.summary_writer.log_dir}")
        print(f"                ├─ {os.path.basename(self.summary_writer.report_archive_path)}")
        print(f"                └─ {os.path.basename(self.summary_writer.summary_archive_path)}")
        print("    - Archive Folder:")
        print(f"        └─ {self.summary_writer.archive_dir}")
        print(f"                ├─ {os.path.basename(self.summary_writer.report_archive_path)}")
        print(f"                └─ {os.path.basename(self.summary_writer.summary_archive_path)}")
        print("")
        print(f'{"=" * 100}')
        print("Seen a glitch? Have a suggestion? Visit https://github.com/Deci-AI/data-gradients !")

    @property
    def n_batches(self):
        # If either train_size or val_size is None (indicating we don't know its size),
        # we will prioritize the value we know. If both are unknown, we cannot determine the max.
        if self.train_size is None and self.val_size is None:
            # If batches_early_stop is set, return that. Otherwise, it's undeterminable.
            return self.batches_early_stop if self.batches_early_stop is not None else float("inf")

        if self.train_size is None:
            max_size = self.val_size
        elif self.val_size is None:
            max_size = self.train_size
        else:
            max_size = max(self.train_size, self.val_size)

        # If batches_early_stop is set, take the minimum of batches_early_stop and the max_size
        # Otherwise, return max_size
        if self.batches_early_stop is not None:
            return min(max_size, self.batches_early_stop)
        return max_size

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

    @staticmethod
    def _format_feature_description(description: str) -> str:
        """
        Formats the feature extractor's description string for a vieable display in HTML.
        """
        if AnalysisManagerAbstract._is_html(description):
            return description
        else:
            return description.replace("\n", "<br />")

    @staticmethod
    def _is_html(description: str) -> bool:
        return description.startswith("<") and description.endswith(">")
