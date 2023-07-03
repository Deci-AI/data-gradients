import os
from typing import Union

from data_gradients.datasets.detection.voc_format_detection_dataset import VOCFormatDetectionDataset

PASCAL_VOC_CLASS_NAMES = (
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
)


class VOCDetectionDataset(VOCFormatDetectionDataset):
    """VOC Detection Dataset is a sub-class of the VOCFormatDetectionDataset,
    but where the folders are structured exactly similarly to the original PascalVOC.

    #### Expected folder structure
    Any structure including at least one sub-directory for images and one for xml labels. They can be the same.

    Example 1: Separate directories for images and labels
    ```
    dataset_root/
        ├── VOC2007/
        │   ├── JPEGImages/
        │   │   ├── 1.jpg
        │   │   ├── 2.jpg
        │   │   └── ...
        │   ├── Annotations/
        │   │   ├── 1.xml
        │   │   ├── 2.xml
        │   │   └── ...
        │   └── ImageSets/
        │       └── Main
        │           ├── train.txt
        │           ├── val.txt
        │           ├── train_val.txt
        │           └── ...
        └── VOC2012/
            └── ...
    ```


    **Note**: The label file need to be stored in XML format, but the file extension can be different.

    #### Expected label files structure
    The label files must be structured in XML format, like in the following example:

    ``` xml
    <annotation>
        <object>
            <name>chair</name>
            <bndbox>
                <xmin>1</xmin>
                <ymin>213</ymin>
                <xmax>263</xmax>
                <ymax>375</ymax>
            </bndbox>
        </object>
        <object>
            <name>sofa</name>
            <bndbox>
                <xmin>104</xmin>
                <ymin>151</ymin>
                <xmax>334</xmax>
                <ymax>287</ymax>
            </bndbox>
        </object>
    </annotation>
    ```


    #### Instantiation
    Let's take an example where we only have VOC2012
    ```
    dataset_root/
        └── VOC2012/
            ├── JPEGImages/
            │   ├── 1.jpg
            │   ├── 2.jpg
            │   └── ...
            ├── Annotations/
            │   ├── 1.xml
            │   ├── 2.xml
            │   └── ...
            └── ImageSets/
                └── Main
                    ├── train.txt
                    └── val.txt
    ```

    ```python
    from data_gradients.datasets.detection import VOCDetectionDataset

    train_set = VOCDetectionDataset(root_dir="<path/to/dataset_root>", year=2012, split="train")
    val_set = VOCDetectionDataset(root_dir="<path/to/dataset_root>", year=2012, split="val")
    ```
    """

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
            labels_subdir=os.path.join(f"VOC{year}", "Annotations"),
            config_path=os.path.join(f"VOC{year}", "ImageSets", "Main", f"{split}.txt"),
            class_names=PASCAL_VOC_CLASS_NAMES,
            verbose=verbose,
        )
