"""
Script for INTERNAL USE ONLY.
Generate a list of reports for detection datasets.
The script requires a clear dataset directory structures and will not work in environments not setup the same way.
"""

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


def _get_all_report_features(train_image_dir: str, valid_image_dir: str):
    """Features defined manually in order to dynamically define `ImageDuplicates(train_image_dir=..., valid_image_dir=...)`"""
    features = [
        SummaryStats(),
        ImagesResolution(),
        ImageColorDistribution(),
        ImagesAverageBrightness(),
        ImageDuplicates(train_image_dir=train_image_dir, valid_image_dir=valid_image_dir),
        DetectionSampleVisualization(n_rows=3, n_cols=4, stack_splits_vertically=True),
        DetectionClassHeatmap(n_rows=6, n_cols=2, heatmap_shape=(200, 200)),
        DetectionBoundingBoxArea(topk=30, prioritization_mode="train_val_diff"),
        DetectionBoundingBoxPerImageCount(),
        DetectionBoundingBoxSize(),
        DetectionClassFrequency(topk=30, prioritization_mode="train_val_diff"),
        DetectionClassesPerImageCount(topk=30, prioritization_mode="train_val_diff"),
        DetectionBoundingBoxIoU(num_bins=10, class_agnostic=True),
    ]
    return features


if __name__ == "__main__":

    DetectionAnalysisManager.analyze_coco(
        root_dir="/data/coco",
        year=2017,
        report_title="COCO",
        feature_extractors=_get_all_report_features(train_image_dir="/data/coco/images/train2017/", valid_image_dir="/data/coco/images/val2017/"),
    )

    # VOC dataset does not clearly split the train/valid sets so we cannot run duplicate analysis
    DetectionAnalysisManager.analyze_voc(
        root_dir="/data/voc/VOCdevkit",
        year=2012,
        report_title="VOC",
    )

    # Running on all the Roboflow100 datasets
    for dataset_name in os.listdir("/data/rf100"):
        dataset_path = os.path.join("/data/rf100", dataset_name)

        DetectionAnalysisManager.analyze_coco_format(
            root_dir=dataset_path,
            feature_extractors=_get_all_report_features(train_image_dir=f"{dataset_path}/train/", valid_image_dir=f"{dataset_path}/valid/"),
            train_images_subdir="train",
            train_annotation_file_path="train/_annotations.coco.json",
            val_images_subdir="valid",
            val_annotation_file_path="valid/_annotations.coco.json",
            report_title=f"DET RF100 {dataset_name}",
        )
