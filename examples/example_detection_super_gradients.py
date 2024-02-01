"""This is an example will require you to install the super-gradients package.

It shows how DataGradients can be used on top of Datasets provided by SuperGradients.

Required step:
```
pip install super-gradients
```
"""

# Note: This example will require you to install the super-gradients package

from data_gradients.managers.detection_manager import DetectionAnalysisManager
from data_gradients.datasets.detection.coco_detection_dataset import COCODetectionDataset
from data_gradients.utils.data_classes.image_channels import BGRChannels

if __name__ == "__main__":
    train_set = COCODetectionDataset(root_dir="/data/coco", split="train", year="2017")
    val_set = COCODetectionDataset(root_dir="/data/coco", split="val", year="2017")

    # %%

    manager = DetectionAnalysisManager(
        report_title="Detection Coco 2000 10 iou thresh",
        train_data=train_set,
        val_data=val_set,
        class_names=["background"] + train_set.class_names,
        is_label_first=True,
        bbox_format="xywh",
        image_channels=BGRChannels("BGR"),
        batches_early_stop=2000,
    )

    # %%

    manager.run()
