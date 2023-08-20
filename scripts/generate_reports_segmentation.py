"""
Script for INTERNAL USE ONLY.
Generate a list of reports for segmentation datasets.
The script requires a clear dataset directory structures and will not work in environments not setup the same way.
"""

from data_gradients.managers.segmentation_manager import SegmentationAnalysisManager
from super_gradients.training.dataloaders import cityscapes_train, cityscapes_val

from data_gradients.feature_extractors import (
    SummaryStats,
    ImagesResolution,
    ImageColorDistribution,
    ImagesAverageBrightness,
    ImageDuplicates,
    SegmentationBoundingBoxArea,
    SegmentationBoundingBoxResolution,
    SegmentationClassFrequency,
    SegmentationClassHeatmap,
    SegmentationClassesPerImageCount,
    SegmentationComponentsConvexity,
    SegmentationComponentsPerImageCount,
    SegmentationSampleVisualization,
)


def _get_all_report_features(train_image_dir: str, valid_image_dir: str):
    """Features defined manually in order to dynamically define `ImageDuplicates(train_image_dir=..., valid_image_dir=...)`"""
    features = [
        SummaryStats(),
        ImagesResolution(),
        ImageColorDistribution(),
        ImagesAverageBrightness(),
        ImageDuplicates(train_image_dir=train_image_dir, valid_image_dir=valid_image_dir),
        SegmentationSampleVisualization(n_rows=3, n_cols=3, stack_splits_vertically=True, stack_mask_vertically=True),
        SegmentationClassHeatmap(n_rows=6, n_cols=2, heatmap_shape=(200, 200)),
        SegmentationClassFrequency(topk=30, prioritization_mode="train_val_diff"),
        SegmentationClassesPerImageCount(topk=30, prioritization_mode="train_val_diff"),
        SegmentationComponentsPerImageCount(),
        SegmentationBoundingBoxResolution(),
        SegmentationBoundingBoxArea(topk=30, prioritization_mode="train_val_diff"),
        SegmentationComponentsConvexity(),
    ]
    return features


if __name__ == "__main__":

    # SegmentationAnalysisManager.analyze_coco(
    #     root_dir="/data/coco",
    #     year=2017,
    #     report_title="SEG COCO",
    #     feature_extractors=_get_all_report_features(train_image_dir="/data/coco/images/train2017/", valid_image_dir="/data/coco/images/val2017/"),
    #     batches_early_stop=10
    # )
    #
    # # VOC dataset does not clearly split the train/valid sets so we cannot run duplicate analysis
    # SegmentationAnalysisManager.analyze_voc(
    #     root_dir="/data/voc/VOCdevkit",
    #     year=2012,
    #     report_title="SEG VOC",
    #     batches_early_stop=10
    # )

    # Cityscape does not support image duplicate out of the box because it is made of multiple image sets
    trainset = cityscapes_train()
    val_set = cityscapes_val()
    SegmentationAnalysisManager(
        train_data=trainset,
        val_data=val_set,
        report_title="SEG Cityspace",
        class_names=trainset.dataset.classes + ["Ignore"],
        feature_extractors=_get_all_report_features(train_image_dir="/data/coco/images/train2017/", valid_image_dir="/data/coco/images/val2017/"),
        batches_early_stop=10,
    ).run()
