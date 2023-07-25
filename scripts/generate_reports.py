import os
from data_gradients.managers.detection_manager import DetectionAnalysisManager


if __name__ == "__main__":

    # analyzer = DetectionAnalysisManager.from_coco(root_dir="/data/coco", year=2017, report_title="COCO")
    # analyzer.run()

    # analyzer = DetectionAnalysisManager.from_voc(root_dir="/data/voc/VOCdevkit", year=2012, report_title="VOC")
    # analyzer.run()

    # {
    #     "report_sections=[
    #         {
    #             "name="Image Features",
    #             "features=[
    #                 "SummaryStats",
    #                 "ImagesResolution",
    #                 "ImageColorDistribution",
    #                 "ImagesAverageBrightness",
    #                 {"ImageDuplicates={"train_image_dir="/data/coco/images/train2017/", valid_image_dir="/data/coco/images/val2017/"}},
    #             ],
    #         },
    #         {
    #             "name="Object Detection Features",
    #             "features=[
    #                 {"DetectionSampleVisualization={"n_rows=3, n_cols=4, stack_splits_vertically=True}},
    #                 {"DetectionClassHeatmap={"n_rows=6, n_cols=2, heatmap_shape=[200, 200]}},
    #                 {"DetectionBoundingBoxArea={"topk=30, prioritization_mode="train_val_diff"}},
    #                 "DetectionBoundingBoxPerImageCount",
    #                 "DetectionBoundingBoxSize",
    #                 {"DetectionClassFrequency={"topk=30, prioritization_mode="train_val_diff"}},
    #                 {"DetectionClassesPerImageCount={"topk=30, prioritization_mode="train_val_diff"}},
    #                 {"DetectionBoundingBoxIoU={"num_bins=10, class_agnostic=True}},
    #             ],
    #         },
    #     ]
    # }

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

    # Running on all the Roboflow100 datasets
    for dataset_name in os.listdir("/data/rf100"):
        dataset_path = os.path.join("/data/rf100", dataset_name)
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
