from typing import List, Optional, Callable

from data_gradients.batch_processors.base import BatchProcessor
from data_gradients.batch_processors.adapters.dataset_adapter import DatasetAdapter
from data_gradients.batch_processors.preprocessors.segmentation import SegmentationBatchPreprocessor
from data_gradients.batch_processors.formatters.segmentation import SegmentationBatchFormatter


class SegmentationBatchProcessor(BatchProcessor):
    def __init__(
        self,
        *,
        n_classes: Optional[int] = None,
        class_names: Optional[List[str]] = None,
        n_image_channels: int,
        threshold_value: float,
        ignore_labels: Optional[List[int]] = None,
        images_extractor: Optional[Callable] = None,
        labels_extractor: Optional[Callable] = None,
    ):
        if n_classes is None and class_names is None:
            raise RuntimeError("Either `n_classes` or `class_names` must be specified")

        if n_classes and class_names:
            if len(class_names) != n_classes:
                raise RuntimeError(f"`len(class_names) != n_classes ({len(class_names)} != {n_classes})")

        n_classes = n_classes or len(class_names)

        dataset_adapter = DatasetAdapter(
            images_extractor=images_extractor,
            labels_extractor=labels_extractor,
        )
        formatter = SegmentationBatchFormatter(
            n_classes=n_classes,
            n_image_channels=n_image_channels,
            threshold_value=threshold_value,
            ignore_labels=ignore_labels,
        )
        preprocessor = SegmentationBatchPreprocessor(class_names=class_names)

        super().__init__(dataset_adapter=dataset_adapter, batch_formatter=formatter, batch_preprocessor=preprocessor)
