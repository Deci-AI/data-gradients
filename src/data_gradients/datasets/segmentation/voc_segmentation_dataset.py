import os
from typing import Union

from data_gradients.datasets.segmentation.voc_format_segmentation_dataset import VOCFormatSegmentationDataset


VOC_COLORMAP = [
    [0, 0, 0],
    [128, 0, 0],
    [0, 128, 0],
    [128, 128, 0],
    [0, 0, 128],
    [128, 0, 128],
    [0, 128, 128],
    [128, 128, 128],
    [64, 0, 0],
    [192, 0, 0],
    [64, 128, 0],
    [192, 128, 0],
    [64, 0, 128],
    [192, 0, 128],
    [64, 128, 128],
    [192, 128, 128],
    [0, 64, 0],
    [128, 64, 0],
    [0, 192, 0],
    [128, 192, 0],
    [0, 64, 128],
]

VOC_CLASSE_NAMES = [
    "background",
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
    "potted plant",
    "sheep",
    "sofa",
    "train",
    "tv/monitor",
]


class VOCSegmentationDataset(VOCFormatSegmentationDataset):
    def __init__(self, root_dir: str, year: Union[int, str], split: str, verbose: bool = False):
        """
        :param root_dir:    Where the data is stored.
        :param year:        Year of the dataset. Usually 2007 or 2012.
        :param split:       Set of images to load. Usually `train` or `val`, but your dataset may include other sets such as `aeroplane_train.txt`, ...
                            Check out your ImageSets/Main folder to find the list
        :param verbose:     Whether to show extra information during loading.
        """
        super().__init__(
            root_dir=root_dir,
            images_subdir=os.path.join(f"VOC{year}", "JPEGImages"),
            labels_subdir=os.path.join(f"VOC{year}", "SegmentationClass"),
            config_path=os.path.join(f"VOC{year}", "ImageSets", "Segmentation", f"{split}.txt"),
            class_names=VOC_CLASSE_NAMES,
            color_map=VOC_COLORMAP,
            verbose=verbose,
        )
