from tqdm import tqdm

from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor

from data_gradients.example_dataset.bdd_dataset import BDDDataset
from data_gradients.batch_processors.segmentation import SegmentationBatchProcessor

from data_gradients.feature_extractors.common.image_resolution import ImagesResolution
from data_gradients.feature_extractors.common.image_average_brightness import ImagesAverageBrightness
from data_gradients.feature_extractors.common.image_color_distribution import ImageColorDistribution
from data_gradients.feature_extractors.segmentation.bounding_boxes_area import SegmentationBoundingBoxArea
from data_gradients.feature_extractors.segmentation.bounding_boxes_resolution import SegmentationBoundingBoxResolution
from data_gradients.feature_extractors.segmentation.classes_count import SegmentationClassesCount
from data_gradients.feature_extractors.segmentation.classes_per_image_count import SegmentationClassesPerImageCount
from data_gradients.feature_extractors.segmentation.components_per_image_count import SegmentationComponentsPerImageCount
from data_gradients.feature_extractors.segmentation.components_convexity import SegmentationComponentsConvexity
from data_gradients.feature_extractors.segmentation.components_erosion import SegmentationComponentsErosion
from data_gradients.feature_extractors.segmentation.classes_heatmap_per_class import SegmentationClassHeatmap

from data_gradients.visualize.seaborn_renderer import SeabornRenderer

# Create torch DataSet
train_dataset = BDDDataset(
    data_folder="../../src/data_gradients/example_dataset/bdd_example",
    split="train",
    transform=Compose([ToTensor()]),
    target_transform=Compose([ToTensor()]),
)
val_dataset = BDDDataset(
    data_folder="../../src/data_gradients/example_dataset/bdd_example",
    split="val",
    transform=Compose([ToTensor()]),
    target_transform=Compose([ToTensor()]),
)

train_loader = DataLoader(train_dataset, batch_size=8)
val_loader = DataLoader(val_dataset, batch_size=8)

batch_processor = SegmentationBatchProcessor(
    class_names=BDDDataset.CLASS_NAMES,
    threshold_value=0.5,
    n_image_channels=3,
)

feature_extractors = [
    ImagesResolution(),
    ImagesAverageBrightness(),
    ImageColorDistribution(),
    SegmentationComponentsPerImageCount(),
    SegmentationBoundingBoxArea(),
    SegmentationBoundingBoxResolution(),
    SegmentationClassesCount(),
    SegmentationClassesPerImageCount(),
    SegmentationComponentsConvexity(),
    SegmentationComponentsErosion(),
    SegmentationClassHeatmap(),
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
