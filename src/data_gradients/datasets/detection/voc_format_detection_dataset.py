import os
import numpy as np
import logging
from typing import List, Tuple, Sequence, Optional
from xml.etree import ElementTree

from data_gradients.datasets.FolderProcessor import ImageLabelFilesIterator, ImageLabelConfigIterator, DEFAULT_IMG_EXTENSIONS
from data_gradients.datasets.utils import load_image, ImageChannelFormat


logger = logging.getLogger(__name__)


class VOCFormatDetectionDataset:
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


    #### Instantiation
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

    train_set = VOCFormatDetectionDataset(root_dir="<path/to/dataset_root>", images_dir="images/train", labels_dir="labels/train")
    val_set = VOCFormatDetectionDataset(root_dir="<path/to/dataset_root>", images_dir="images/validation", labels_dir="labels/validation")
    ```
    """

    def __init__(
        self,
        root_dir: str,
        images_dir: str,
        labels_dir: str,
        class_names: List[str],
        config_path: Optional[str],
        verbose: bool = False,
        image_extensions: Sequence[str] = DEFAULT_IMG_EXTENSIONS,
        label_extensions: Sequence[str] = ("xml",),
    ):
        """
        :param root_dir:        Where the data is stored.
        :param images_dir:      Local path to directory that includes all the images. Path relative to `root_dir`. Can be the same as `labels_dir`.
        :param labels_dir:      Local path to directory that includes all the labels. Path relative to `root_dir`. Can be the same as `images_dir`.
        :param class_names:     List of class names. This is required to be able to parse the class names into class ids.
        :param verbose:         Whether to show extra information during loading.
        :param image_extensions: List of image file extensions to load from.
        :param label_extensions: List of label file extensions to load from.
        """
        self.class_names = class_names
        if config_path is None:
            self.image_label_tuples = ImageLabelFilesIterator(
                images_dir=os.path.join(root_dir, images_dir),
                labels_dir=os.path.join(root_dir, labels_dir),
                image_extensions=image_extensions,
                label_extensions=label_extensions,
                verbose=verbose,
            )
        else:
            self.image_label_tuples = ImageLabelConfigIterator(
                images_dir=os.path.join(root_dir, images_dir),
                labels_dir=os.path.join(root_dir, labels_dir),
                config_path=config_path,
                image_extensions=image_extensions,
                label_extensions=label_extensions,
                verbose=verbose,
            )

    def load_image(self, index: int) -> np.ndarray:
        img_file, _ = self.image_label_tuples[index]
        return load_image(path=img_file, channel_format=ImageChannelFormat.RGB)

    def load_annotation(self, index: int) -> np.ndarray:
        _, label_path = self.image_label_tuples[index]

        with open(label_path) as f:
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

    def __getitem__(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
        image = self.load_image(index)
        annotation = self.load_annotation(index)
        return image, annotation
