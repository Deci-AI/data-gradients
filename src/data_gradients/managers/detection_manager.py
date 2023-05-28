from typing import Optional, Iterable, Dict, Callable

from data_gradients.managers.abstract_manager import AnalysisManagerAbstract
from data_gradients.config.utils import load_extractors
from data_gradients.batch_processors.detection import DetectionBatchProcessor
from data_gradients.visualize.image_samplers.detection import DetectionImageSampleManager


class DetectionAnalysisManager(AnalysisManagerAbstract):
    """Main detection manager class.
    Definition of task name, task-related preprocessor and parsing related configuration file
    """

    def __init__(
        self,
        *,
        n_classes: int,
        train_data: Iterable,
        val_data: Optional[Iterable] = None,
        config_name: str = "detection",
        log_dir: Optional[str] = None,
        id_to_name: Optional[Dict] = None,
        batches_early_stop: int = 999,
        images_extractor: Callable = None,
        labels_extractor: Callable = None,
        n_image_channels: int = 3,
        short_run: bool = False,
        samples_to_visualize: int = 10,
    ):
        """
        Constructor of detection manager which controls the analyzer
        :param n_classes:             Number of valid classes to analyze
        :param train_data:              Iterable object contains images and labels of the training dataset
        :param val_data:                Iterable object contains images and labels of the validation dataset
        :param config_name:             Name of the hydra configuration file
        :param log_dir:                 Directory where to save the logs. By default uses the current working directory
        :param id_to_name:              Class ID to class names mapping (Dictionary)
        :param batches_early_stop:      Maximum number of batches to run in training (early stop)
        :param images_extractor:
        :param labels_extractor:
        :param n_image_channels:      Number of channels for each image in the dataset
        :param short_run:               Flag indicating whether to run for a single epoch first to estimate total duration,
                                        before choosing the number of epochs.
        :param samples_to_visualize:    Number of samples to visualize at tensorboard [0-n]
        """

        batch_processor = DetectionBatchProcessor(
            images_extractor=images_extractor,
            labels_extractor=labels_extractor,
            n_image_channels=n_image_channels,
        )

        feature_extractors = load_extractors(config_name=config_name, overrides={"number_of_classes": n_classes})

        image_sample_manager = DetectionImageSampleManager(n_samples=samples_to_visualize)

        super().__init__(
            train_data=train_data,
            val_data=val_data,
            batch_processor=batch_processor,
            feature_extractors=feature_extractors,
            log_dir=log_dir,
            id_to_name=id_to_name,
            batches_early_stop=batches_early_stop,
            short_run=short_run,
            image_sample_manager=image_sample_manager,
        )