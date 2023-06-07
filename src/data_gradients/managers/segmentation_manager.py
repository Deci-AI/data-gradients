from typing import Optional, Iterable, Dict, Callable, List

from data_gradients.managers.abstract_manager import AnalysisManagerAbstract
from data_gradients.config.utils import load_report_feature_extractors
from data_gradients.batch_processors.segmentation import SegmentationBatchProcessor
from data_gradients.config.interactive_config import InteractiveConfig


class SegmentationAnalysisManager(AnalysisManagerAbstract):
    """
    Main semantic segmentation manager class.
    Definition of task name, task-related preprocessor and parsing related configuration file
    """

    def __init__(
        self,
        *,
        report_title: str,
        config: InteractiveConfig,
        class_names: Optional[List[str]] = None,
        class_names_to_use: Optional[List[str]] = None,
        n_classes: Optional[int] = None,
        train_data: Iterable,
        val_data: Optional[Iterable] = None,
        report_subtitle: Optional[str] = None,
        config_name: str = "semantic_segmentation",
        log_dir: Optional[str] = None,
        id_to_name: Optional[Dict] = None,
        batches_early_stop: int = 999,
        images_extractor: Callable = None,
        labels_extractor: Callable = None,
        num_image_channels: int = 3,
        threshold_soft_labels: float = 0.5,
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
        :param config_name:             Name of the hydra configuration file
        :param log_dir:                 Directory where to save the logs. By default uses the current working directory
        :param id_to_name:              Class ID to class names mapping (Dictionary)
        :param batches_early_stop:      Maximum number of batches to run in training (early stop)
        :param images_extractor:
        :param labels_extractor:
        :param num_image_channels:      Number of channels for each image in the dataset
        :param threshold_soft_labels:   Threshold for converting soft labels to binary labels
        """

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

        batch_processor = SegmentationBatchProcessor(
            config=config,
            class_names=class_names,
            class_names_to_use=class_names_to_use,
            n_image_channels=num_image_channels,
            threshold_value=threshold_soft_labels,
        )

        grouped_feature_extractors = load_report_feature_extractors(config_name=config_name)

        super().__init__(
            config=config,
            report_title=report_title,
            report_subtitle=report_subtitle,
            train_data=train_data,
            val_data=val_data,
            batch_processor=batch_processor,
            grouped_feature_extractors=grouped_feature_extractors,
            log_dir=log_dir,
            id_to_name=id_to_name,
            batches_early_stop=batches_early_stop,
        )
