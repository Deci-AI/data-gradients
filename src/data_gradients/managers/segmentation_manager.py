from typing import Optional, Iterable, List, Dict, Callable

import hydra
from omegaconf import OmegaConf

from data_gradients.logging.logger import Logger
from data_gradients.logging.segmentation.tensorboard_logger import (
    SegmentationTensorBoardLogger,
)
from data_gradients.managers.abstract_manager import AnalysisManagerAbstract
from data_gradients.preprocess.segmentation_preprocess import SegmentationPreprocessor

OmegaConf.register_new_resolver("merge", lambda x, y: x + y)


class SegmentationAnalysisManager(AnalysisManagerAbstract):
    """
    Main semantic segmentation manager class.
    Definition of task name, task-related preprocessor and parsing related configuration file
    """

    TASK = "semantic_segmentation"

    def __init__(
        self,
        *,
        num_classes: int,
        train_data: Iterable,
        ignore_labels: List[int] = None,
        val_data: Optional[Iterable] = None,
        samples_to_visualize: int = 10,
        id_to_name: Optional[Dict] = None,
        batches_early_stop: int = 999,
        images_extractor: Callable = None,
        labels_extractor: Callable = None,
        num_image_channels: int = 3,
        threshold_soft_labels: float = 0.5,
        short_run: bool = False
    ):
        """
        Constructor of semantic-segmentation manager which controls the analyzer
        :param num_classes: Number of valid classes to analyze
        :param train_data: Iterable object contains images and labels of the training dataset
        :param ignore_labels: List of not-valid labeled classes such as background.
        :param val_data: Iterable object contains images and labels of the validation dataset
        :param samples_to_visualize: Number of samples to visualize at tensorboard [0-n]
        :param id_to_name: Class ID to class names mapping (Dictionary)
        """
        super().__init__(
            train_data=train_data,
            val_data=val_data,
            logger=Logger(tb_logger=SegmentationTensorBoardLogger(samples_to_visualize)),
            id_to_name=id_to_name,
            batches_early_stop=batches_early_stop,
            short_run=short_run,
        )

        self._preprocessor = SegmentationPreprocessor(
            num_classes=num_classes,
            ignore_labels=ignore_labels,
            images_extractor=images_extractor,
            labels_extractor=labels_extractor,
            num_image_channels=num_image_channels,
            threshold_value=threshold_soft_labels,
        )

        self._parse_cfg()

    def _parse_cfg(self) -> None:
        """
        Parsing semantic segmentation configuration file with number of classes and ignore labels
        """
        hydra.initialize(config_path="../config/", version_base="1.2")
        self._cfg = hydra.compose(config_name=self.TASK)
        # Could add those parameters with no defining them in if disabling strict mode
        self._cfg.number_of_classes = self._preprocessor.number_of_classes
        self._cfg.ignore_labels = self._preprocessor.ignore_labels
