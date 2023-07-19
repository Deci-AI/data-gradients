import os
from typing import Union

from data_gradients.datasets.download.voc import download_VOC
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
    """
    The VOCSegmentationDataset is specifically tailored for loading PASCAL VOC segmentation datasets.

    #### Expected folder structure
    Similar to the VOCFormatSegmentationDataset, this class also expects certain folder structures.
    The folder structure of the PASCAL VOC dataset is as follows:

    ```
        dataset_root/
            ├── VOC2007/
            │   ├── JPEGImages/
            │   ├── SegmentationClass/
            │   └── ImageSets/
            │       └── Segmentation/
            │           ├── train.txt
            │           └── val.txt
            └── VOC2012/
                ├── JPEGImages/
                ├── SegmentationClass/
                └── ImageSets/
                    └── Segmentation/
                        ├── train.txt
                        └── val.txt
    ```
    Each label image should be a color image where the color of each pixel corresponds to the class of that pixel.

    #### Instantiation
    ```
    from data_gradients.datasets.segmentation import VOCSegmentationDataset

    train_set = VOCSegmentationDataset(
        root_dir="<path/to/dataset_root>",
        year=2007,
        split="train",
        verbose=True
    )
    val_set = VOCSegmentationDataset(
        root_dir="<path/to/dataset_root>",
        year=2007,
        split="val",
        verbose=True
    )
    ```
    """

    CLASS_NAMES = VOC_CLASSE_NAMES

    def __init__(self, root_dir: str, year: Union[int, str], split: str, download: bool = True, verbose: bool = False):
        """
        :param root_dir:    Root directory where the VOC dataset is stored.
        :param year:        Year of the VOC dataset (2007 or 2012).
        :param split:       Set of images to load. This usually corresponds to the data split (train or val).
                            Your dataset may also include other sets such as those specific to a class (e.g., `aeroplane_train.txt`).
                            These sets can be found in the `ImageSets/Main` folder.
        :param download:    If True, download the VOC dataset.
        :param verbose:     If True, print out additional information during the data loading process.
        """
        root_dir = os.path.abspath(root_dir)
        print(root_dir)
        if download:
            download_VOC(year=year, download_root=root_dir)

        super().__init__(
            root_dir=root_dir,
            images_subdir=os.path.join("VOCdevkit", f"VOC{year}", "JPEGImages"),
            labels_subdir=os.path.join("VOCdevkit", f"VOC{year}", "SegmentationClass"),
            config_path=os.path.join("VOCdevkit", f"VOC{year}", "ImageSets", "Segmentation", f"{split}.txt"),
            class_names=VOC_CLASSE_NAMES,
            color_map=VOC_COLORMAP,
            verbose=verbose,
        )
