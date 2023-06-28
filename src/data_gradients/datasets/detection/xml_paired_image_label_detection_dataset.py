import os
import numpy as np
import logging
from typing import List, Tuple, Sequence
from xml.etree import ElementTree

from data_gradients.datasets.FolderProcessor import ImageLabelFilesIterator, DEFAULT_IMG_EXTENSIONS
from data_gradients.datasets.utils import load_image, ImageChannelFormat


logger = logging.getLogger(__name__)


class XMLPairedImageLabelDetectionDataset:
    def __init__(
        self,
        root_dir: str,
        images_dir: str,
        labels_dir: str,
        class_names: List[str],
        verbose: bool = False,
        image_extension: Sequence[str] = DEFAULT_IMG_EXTENSIONS,
        label_extension: Sequence[str] = ("xml",),
    ):
        """
        :param root_dir:        Where the data is stored.
        :param images_dir:      Local path to directory that includes all the images. Path relative to `root_dir`. Can be the same as `labels_dir`.
        :param labels_dir:      Local path to directory that includes all the labels. Path relative to `root_dir`. Can be the same as `images_dir`.
        :param class_names:     List of class names. This is required to be able to parse the class names into class ids.
        :param verbose:         Whether to show extra information during loading.
        :param image_extension: List of image file extensions to load from.
        :param label_extension: List of label file extensions to load from.
        """
        self.image_label_tuples = ImageLabelFilesIterator(
            images_dir=os.path.join(root_dir, images_dir),
            labels_dir=os.path.join(root_dir, labels_dir),
            image_extension=image_extension,
            label_extension=label_extension,
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

        return np.array(labels) if labels else np.zeros((0, 5))

    def __getitem__(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
        image = self.load_image(index)
        annotation = self.load_annotation(index)
        return image, annotation
