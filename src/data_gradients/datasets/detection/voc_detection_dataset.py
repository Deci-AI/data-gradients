import os
from typing import Union

from data_gradients.datasets.detection.xml_paired_image_label_detection_dataset import XMLPairedImageLabelDetectionDataset

PASCAL_VOC_2012_CLASSES_LIST = [
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
]


class VOCDetectionDataset(XMLPairedImageLabelDetectionDataset):
    def __init__(self, root_dir: str, year: Union[int, str], image_set: str = "train", verbose: bool = False):
        super().__init__(
            root_dir=root_dir,
            images_dir=os.path.join(root_dir, f"VOC{year}", "JPEGImages"),
            labels_dir=os.path.join(root_dir, f"VOC{year}", "Annotations"),
            config_path=os.path.join(root_dir, f"VOC{year}", "ImageSets", "Main", f"{image_set}.txt"),
            class_names=PASCAL_VOC_2012_CLASSES_LIST,
            verbose=verbose,
        )
