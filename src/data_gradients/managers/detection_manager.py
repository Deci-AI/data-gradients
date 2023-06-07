from typing import Optional, Iterable, Dict, List

from data_gradients.managers.abstract_manager import AnalysisManagerAbstract
from data_gradients.config.utils import load_report_feature_extractors
from data_gradients.batch_processors.detection import DetectionBatchProcessor
from data_gradients.config.interactive_config import InteractiveConfig


class DetectionAnalysisManager(AnalysisManagerAbstract):
    """Main detection manager class.
    Definition of task name, task-related preprocessor and parsing related configuration file
    """

    def __init__(
        self,
        *,
        report_title: str,
        config: InteractiveConfig,
        train_data: Iterable,
        val_data: Optional[Iterable] = None,
        report_subtitle: Optional[str] = None,
        class_names: Optional[List[str]] = None,
        class_names_to_use: Optional[List[str]] = None,
        n_classes: Optional[int] = None,
        config_name: str = "detection",
        log_dir: Optional[str] = None,
        id_to_name: Optional[Dict] = None,
        batches_early_stop: int = 999,
        n_image_channels: int = 3,
    ):
        """
        Constructor of detection manager which controls the analyzer
        :param report_title:            Title of the report. Will be used to save the report
        :param report_subtitle:         Subtitle of the report
        :param class_names:             List of all class names in the dataset. The index should represent the class_id.
        :param class_names_to_use:      List of class names that we should use for analysis.
        :param n_classes:               Number of classes. Mutually exclusive with `class_names`.
        :param train_data:              Iterable object contains images and labels of the training dataset
        :param val_data:                Iterable object contains images and labels of the validation dataset
        :param config_name:             Name of the hydra configuration file
        :param log_dir:                 Directory where to save the logs. By default uses the current working directory
        :param id_to_name:              Class ID to class names mapping (Dictionary)
        :param batches_early_stop:      Maximum number of batches to run in training (early stop)
        :param images_extractor:
        :param labels_extractor:
        :param n_image_channels:      Number of channels for each image in the dataset
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

        batch_processor = DetectionBatchProcessor(
            config=config,
            n_image_channels=n_image_channels,
            class_names=class_names,
            class_names_to_use=class_names_to_use,
        )

        feature_extractors = load_report_feature_extractors(config_name=config_name)

        super().__init__(
            config=config,
            report_title=report_title,
            report_subtitle=report_subtitle,
            train_data=train_data,
            val_data=val_data,
            batch_processor=batch_processor,
            grouped_feature_extractors=feature_extractors,
            log_dir=log_dir,
            id_to_name=id_to_name,
            batches_early_stop=batches_early_stop,
        )
