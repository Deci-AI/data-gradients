import os
from data_gradients.managers.detection_manager import DetectionAnalysisManager


if __name__ == "__main__":

    analyzer = DetectionAnalysisManager.from_coco(root_dir="/data/coco", year=2017, report_title="COCO")
    analyzer.run()

    analyzer = DetectionAnalysisManager.from_voc(root_dir="/data/voc/VOCdevkit", year=2012, report_title="VOC")
    analyzer.run()

    # Running on all the Roboflow100 datasets
    for dataset_name in os.listdir("/data/rf100"):
        analyzer = DetectionAnalysisManager.from_coco_format(
            root_dir=os.path.join("/data/rf100", dataset_name),
            train_images_subdir="train",
            train_annotation_file_path="train/_annotations.coco.json",
            val_images_subdir="valid",
            val_annotation_file_path="valid/_annotations.coco.json",
            report_title=f"RF100 {dataset_name}",
        )
        analyzer.run()
