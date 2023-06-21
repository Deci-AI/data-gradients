import os
from typing import Optional, Iterable, Callable, List
import torch

from data_gradients.managers.abstract_manager import AnalysisManagerAbstract
from data_gradients.config.utils import load_report_feature_extractors
from data_gradients.batch_processors.segmentation import SegmentationBatchProcessor
from data_gradients.config.data.data_config import SegmentationDataConfig
from data_gradients.config.data.typing import SupportedDataType


class SegmentationAnalysisManager(AnalysisManagerAbstract):
    """
    Main semantic segmentation manager class.
    Definition of task name, task-related preprocessor and parsing related configuration file
    """

    def __init__(
        self,
        *,
        report_title: str,
        train_data: Iterable,
        val_data: Optional[Iterable] = None,
        report_subtitle: Optional[str] = None,
        config_path: Optional[str] = None,
        log_dir: Optional[str] = None,
        use_cache: bool = False,
        class_names: Optional[List[str]] = None,
        class_names_to_use: Optional[List[str]] = None,
        n_classes: Optional[int] = None,
        images_extractor: Optional[Callable[[SupportedDataType], torch.Tensor]] = None,
        labels_extractor: Optional[Callable[[SupportedDataType], torch.Tensor]] = None,
        num_image_channels: int = 3,
        threshold_soft_labels: float = 0.5,
        batches_early_stop: int = 999,
    ):
        """
        Constructor of semantic-segmentation manager which controls the analyzer

        :param report_title:            Title of the report. Will be used to save the report
        :param report_subtitle:         Subtitle of the report
        :param class_names:             List of all class names in the dataset. The index should represent the class_id. Mutually exclusive with `n_classes`
        :param class_names_to_use:      List of class names that we should use for analysis.
        :param n_classes:               Number of classes. Mutually exclusive with `class_names`. If set, `class_names` will be a list of `class_ids`.
        :param train_data:              Iterable object contains images and labels of the training dataset
        :param val_data:                Iterable object contains images and labels of the validation dataset
        :param config_path:             Full path the hydra configuration file. If None, the default configuration will be used.
        :param log_dir:                 Directory where to save the logs. By default uses the current working directory
        :param id_to_name:              Class ID to class names mapping (Dictionary)
        :param batches_early_stop:      Maximum number of batches to run in training (early stop)
        :param use_cache:               Whether to use cache or not for the configuration of the data.
        :param images_extractor:        Function extracting the image(s) out of the data output.
        :param labels_extractor:        Function extracting the label(s) out of the data output.
        :param num_image_channels:      Number of channels for each image in the dataset
        :param threshold_soft_labels:   Threshold for converting soft labels to binary labels
        """
        data_config = SegmentationDataConfig(use_cache=use_cache, images_extractor=images_extractor, labels_extractor=labels_extractor)

        # Check values of `n_classes` and `class_names` to define `class_names`.
        if n_classes and class_names:
            raise RuntimeError("`class_names` and `n_classes` cannot be specified at the same time")
        elif n_classes is None and class_names is None:
            raise RuntimeError("Either `class_names` or `n_classes` must be specified")
        class_names = class_names if class_names else list(map(str, range(n_classes)))

        # Define `class_names_to_use`
        if class_names_to_use:
            invalid_class_names_to_use = set(class_names_to_use) - set(class_names)
            if invalid_class_names_to_use != set():
                raise RuntimeError(f"You defined `class_names_to_use` with classes that are not listed in `class_names`: {invalid_class_names_to_use}")
        class_names_to_use = class_names_to_use or class_names

        # Resolve `config_dir` and `config_name` defining the feature extractors.
        if config_path is None:
            config_dir, config_name = None, "segmentation"
        else:
            config_path = os.path.abspath(config_path)
            config_dir, config_name = os.path.dirname(config_path), os.path.basename(config_path).split(".")[0]

        batch_processor = SegmentationBatchProcessor(
            data_config=data_config,
            class_names=class_names,
            class_names_to_use=class_names_to_use,
            n_image_channels=num_image_channels,
            threshold_value=threshold_soft_labels,
        )

        grouped_feature_extractors = load_report_feature_extractors(config_name=config_name, config_dir=config_dir)

        super().__init__(
            data_config=data_config,
            report_title=report_title,
            report_subtitle=report_subtitle,
            train_data=train_data,
            val_data=val_data,
            batch_processor=batch_processor,
            grouped_feature_extractors=grouped_feature_extractors,
            log_dir=log_dir,
            batches_early_stop=batches_early_stop,
        )
