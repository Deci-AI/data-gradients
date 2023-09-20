import numpy as np
import logging
from typing import Sequence, Optional
from xml.etree import ElementTree

from data_gradients.datasets.base_dataset import BaseImageLabelDirectoryDataset
from data_gradients.datasets.FolderProcessor import DEFAULT_IMG_EXTENSIONS


logger = logging.getLogger(__name__)


class VOCFormatDetectionDataset(BaseImageLabelDirectoryDataset):
    """The VOC format Detection Dataset supports datasets where labels are stored in XML following according to VOC standard.

    #### Expected folder structure
    Any structure including at least one sub-directory for images and one for xml labels. They can be the same.

    Example 1: Separate directories for images and labels
    ```
        dataset_root/
            ├── images/
            │   ├── train/
            │   │   ├── 1.jpg
            │   │   ├── 2.jpg
            │   │   └── ...
            │   ├── test/
            │   │   ├── ...
            │   └── validation/
            │       ├── ...
            └── labels/
                ├── train/
                │   ├── 1.xml
                │   ├── 2.xml
                │   └── ...
                ├── test/
                │   ├── ...
                └── validation/
                    ├── ...
    ```

    Example 2: Same directory for images and labels
    ```
        dataset_root/
            ├── train.txt
            ├── validation.txt
            ├── train/
            │   ├── 1.jpg
            │   ├── 1.xml
            │   ├── 2.jpg
            │   ├── 2.xml
            │   └── ...
            └── validation/
                ├── ...
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

    The (optional) config file should include the list image ids to include.
    ```
    1
    5
    6
    ...
    34122
    ```
    The associated images/labels will then be loaded from the images_subdir and labels_subdir.
    If config_path is not provided, all images will be used.

    #### Instantiation
    ```
    dataset_root/
        ├── train.txt
        ├── validation.txt
        ├── images/
        │   ├── train/
        │   │   ├── 1.jpg
        │   │   ├── 2.jpg
        │   │   └── ...
        │   ├── test/
        │   │   ├── ...
        │   └── validation/
        │       ├── ...
        └── labels/
            ├── train/
            │   ├── 1.txt
            │   ├── 2.txt
            │   └── ...
            ├── test/
            │   ├── ...
            └── validation/
                ├── ...
    ```


    ```python
    from data_gradients.datasets.detection import VOCFormatDetectionDataset

    train_set = VOCFormatDetectionDataset(
        root_dir="<path/to/dataset_root>", images_subdir="images/train", labels_subdir="labels/train", config_path="train.txt"
    )
    val_set = VOCFormatDetectionDataset(
        root_dir="<path/to/dataset_root>", images_subdir="images/validation", labels_subdir="labels/validation", config_path="validation.txt"
    )
    ```
    """

    def __init__(
        self,
        root_dir: str,
        images_subdir: str,
        labels_subdir: str,
        class_names: Sequence[str],
        config_path: Optional[str] = None,
        verbose: bool = False,
        image_extensions: Sequence[str] = DEFAULT_IMG_EXTENSIONS,
        label_extensions: Sequence[str] = ("xml",),
    ):
        """
        :param root_dir:            Where the data is stored.
        :param images_subdir:       Local path to directory that includes all the images. Path relative to `root_dir`. Can be the same as `labels_subdir`.
        :param labels_subdir:       Local path to directory that includes all the labels. Path relative to `root_dir`. Can be the same as `images_subdir`.
        :param class_names:         List of class names. This is required to be able to parse the class names into class ids.
        :param config_path:         Path to an optional config file. This config file should contain the list of file ids to include.
                                    If None, all the available images and tagets will be loaded.
        :param verbose:             Whether to show extra information during loading.
        :param image_extensions:    List of image file extensions to load from.
        :param label_extensions:    List of label file extensions to load from.
        """
        super().__init__(
            root_dir=root_dir,
            images_subdir=images_subdir,
            labels_subdir=labels_subdir,
            config_path=config_path,
            verbose=verbose,
            image_extensions=image_extensions,
            label_extensions=label_extensions,
        )
        self.class_names = class_names

    def load_labels(self, path: str) -> np.ndarray:

        with open(path, encoding="utf-8") as f:
            xml_parser = ElementTree.parse(f).getroot()

        labels = []
        for obj in xml_parser.iter("object"):
            class_name = obj.find("name").text
            xml_box = obj.find("bndbox")

            if class_name in self.class_names and obj.find("difficult").text != "1":  # TODO: understand if we want difficult!=1 or not
                class_id = self.class_names.index(class_name)
                xmin = xml_box.find("xmin").text
                ymin = xml_box.find("ymin").text
                xmax = xml_box.find("xmax").text
                ymax = xml_box.find("ymax").text
                labels.append([class_id, xmin, ymin, xmax, ymax])

        return np.array(labels, dtype=float) if labels else np.zeros((0, 5), dtype=float)
