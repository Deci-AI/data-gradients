import os
from data_gradients.managers.detection_manager import DetectionAnalysisManager

from data_gradients.feature_extractors import (
    SummaryStats,
    ImagesResolution,
    ImageColorDistribution,
    ImagesAverageBrightness,
    ImageDuplicates,
    DetectionSampleVisualization,
    DetectionClassHeatmap,
    DetectionBoundingBoxArea,
    DetectionBoundingBoxPerImageCount,
    DetectionBoundingBoxSize,
    DetectionClassFrequency,
    DetectionClassesPerImageCount,
    DetectionBoundingBoxIoU,
)

DATASETS = (
    "acl-x-ray",
    "circuit-voltages",
    "furniture-ngpea",
    "gynecology-mri",
    "hand-gestures-jps7z",
    "mitosis-gjs3g",
    "number-ops",
    "road-signs-6ih4y",
    "shark-teeth-5atku",
    "sign-language-sokdr",
    "tweeter-profile",
    "wall-damage",
)
if __name__ == "__main__":

    analyzer = DetectionAnalysisManager.from_coco(root_dir="/data/coco", year=2017, report_title="COCO")
    analyzer.run()

    analyzer = DetectionAnalysisManager.from_voc(root_dir="/data/voc/VOCdevkit", year=2012, report_title="VOC")
    analyzer.run()

    # Running on all the Roboflow100 datasets
    for dataset_name in os.listdir("/data/rf100"):
        dataset_path = os.path.join("/data/rf100", dataset_name)

        # Features defined manually in order to dynamically define `ImageDuplicates(train_image_dir=..., valid_image_dir=...)`
        features = [
            SummaryStats(),
            ImagesResolution(),
            ImageColorDistribution(),
            ImagesAverageBrightness(),
            ImageDuplicates(train_image_dir=f"{dataset_path}/train/", valid_image_dir=f"{dataset_path}/valid/"),
            DetectionSampleVisualization(n_rows=3, n_cols=4, stack_splits_vertically=True),
            DetectionClassHeatmap(n_rows=6, n_cols=2, heatmap_shape=(200, 200)),
            DetectionBoundingBoxArea(topk=30, prioritization_mode="train_val_diff"),
            DetectionBoundingBoxPerImageCount(),
            DetectionBoundingBoxSize(),
            DetectionClassFrequency(topk=30, prioritization_mode="train_val_diff"),
            DetectionClassesPerImageCount(topk=30, prioritization_mode="train_val_diff"),
            DetectionBoundingBoxIoU(num_bins=10, class_agnostic=True),
        ]

        analyzer = DetectionAnalysisManager.from_coco_format(
            root_dir=dataset_path,
            feature_extractors=features,
            train_images_subdir="train",
            train_annotation_file_path="train/_annotations.coco.json",
            val_images_subdir="valid",
            val_annotation_file_path="valid/_annotations.coco.json",
            report_title=f"DET RF100 {dataset_name}",
        )
        analyzer.run()
