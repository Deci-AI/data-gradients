from tqdm import tqdm

from data_gradients.batch_processors.detection import DetectionBatchProcessor

from data_gradients.feature_extractors.common.image_resolution import ImagesResolution
from data_gradients.feature_extractors.common.image_average_brightness import ImagesAverageBrightness
from data_gradients.feature_extractors.common.image_color_distribution import ImageColorDistribution
from data_gradients.feature_extractors.object_detection.bounding_boxes_area import DetectionBoundingBoxArea
from data_gradients.feature_extractors.object_detection.bounding_boxes_resolution import DetectionBoundingBoxSize
from data_gradients.feature_extractors.object_detection.classes_count import DetectionObjectsPerClass
from data_gradients.feature_extractors.object_detection.classes_per_image_count import DetectionClassesPerImageCount
from data_gradients.feature_extractors.object_detection.bounding_boxes_per_image_count import DetectionBoundingBoxPerImageCount

from data_gradients.visualize.seaborn_renderer import SeabornRenderer
import numpy as np
from torch.utils.data import DataLoader

# Note: This example will require you to install the super-gradients package
from super_gradients.training.datasets import YoloDarknetFormatDetectionDataset


class PadTarget:
    """Transform targets (Compatible with Sg DetectionDatasets)"""

    def __init__(self, max_targets: int):
        self.max_targets = max_targets

    def __call__(self, sample):
        targets = sample["target"]
        targets = np.ascontiguousarray(targets, dtype=np.float32)
        padded_targets = np.zeros((self.max_targets, targets.shape[-1]))
        padded_targets[range(len(targets))[: self.max_targets]] = targets[: self.max_targets]
        padded_targets = np.ascontiguousarray(padded_targets, dtype=np.float32)
        sample["target"] = padded_targets
        return sample


data_dir = "/Users/Louis.Dupont/Downloads/avatar_recognition.v2-release.yolov8"
classes = ["Character"]

# Create torch DataSet
train_dataset = YoloDarknetFormatDetectionDataset(
    data_dir=data_dir,
    images_dir="train/images",
    labels_dir="train/labels",
    classes=classes,
    transforms=[PadTarget(max_targets=50)],
)
val_dataset = YoloDarknetFormatDetectionDataset(
    data_dir=data_dir,
    images_dir="valid/images",
    labels_dir="valid/labels",
    classes=classes,
    transforms=[PadTarget(max_targets=50)],
)

# Create torch DataLoader
train_loader = DataLoader(train_dataset, batch_size=8)
val_loader = DataLoader(val_dataset, batch_size=8)
batch_processor = DetectionBatchProcessor(n_image_channels=3)

feature_extractors = [
    ImagesResolution(),
    ImagesAverageBrightness(),
    ImageColorDistribution(),
    DetectionBoundingBoxPerImageCount(),
    DetectionBoundingBoxArea(),
    DetectionBoundingBoxSize(),
    DetectionObjectsPerClass(),
    DetectionClassesPerImageCount(),
]

sns = SeabornRenderer()

for train_batch in tqdm(train_loader):
    for sample in batch_processor.process(train_batch, split="train"):
        for feature_extractor in feature_extractors:
            feature_extractor.update(sample)

for val_batch in tqdm(val_loader):
    for sample in batch_processor.process(val_batch, split="valid"):
        for feature_extractor in feature_extractors:
            feature_extractor.update(sample)

for feature_extractor in feature_extractors:
    feature = feature_extractor.aggregate()
    f = sns.render(feature.data, feature.plot_options)
    f.show()
