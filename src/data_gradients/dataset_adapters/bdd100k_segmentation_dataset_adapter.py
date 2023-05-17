import os
from typing import List

import numpy as np
from PIL import Image

from data_gradients.dataset_adapters.adapter_interface import SegmentationDatasetAdapter, SegmentationSample


class BDD100KSegmentationDatasetAdapter(SegmentationDatasetAdapter):
    """
    SegmentationDatasetAdapter implementation of the BDD100K dataset.
    The BDD100K data and annotations can be obtained at https://bdd-data.berkeley.edu/.

    Usage:

    >>>    from data_gradients.dataset_adapters import BDD100KSegmentationDatasetAdapter
    >>>    train_ds_adapter = BDD100KSegmentationDatasetAdapter(data_dir="path/to/train/dataset")
    >>>    val_ds_adapter = BDD100KSegmentationDatasetAdapter(data_dir="path/to/val/dataset")
    >>>
    >>>    data = {
    >>>        "train": train_ds_adapter,
    >>>        "val": val_ds_adapter
    >>>    }
    >>>
    >>>    mgr = SegmentationAnalysisManager(...)
    >>>    mgr.run(data)

    """

    def __init__(self, data_dir: str):
        super().__init__()
        self.num_classes = 19
        self.ignore_labels = [255]
        self.class_names = {
            0: "road",
            1: "sidewalk",
            2: "building",
            3: "wall",
            4: "fence",
            5: "pole",
            6: "traffic light",
            7: "traffic sign",
            8: "vegetation",
            9: "terrain",
            10: "sky",
            11: "person",
            12: "rider",
            13: "car",
            14: "truck",
            15: "bus",
            16: "train",
            17: "motorcycle",
            18: "bicycle",
        }

        self.known_labels = np.array(list(self.class_names.keys()) + self.ignore_labels)
        files_list = sorted(os.listdir(data_dir))
        self.samples_fn = []
        for f in files_list:
            if f[-3:] == "jpg":
                self.samples_fn.append(
                    [
                        os.path.join(data_dir, f),
                        os.path.join(data_dir, f[0:-3] + "png"),
                    ]
                )

    def get_num_classes(self) -> int:
        return self.num_classes

    def get_class_names(self) -> List[str]:
        return list(self.class_names.values())

    def get_ignored_classes(self) -> List[int]:
        return self.ignore_labels

    def get_iterator(self):
        for image_path, mask_path in self.samples_fn:
            image = Image.open(image_path).convert("RGB")
            mask = Image.open(mask_path)

            image = np.array(image)
            mask = np.array(mask)

            if not np.isin(mask, self.known_labels).all():
                unexpected_labels = set(np.unique(mask)) - set(self.known_labels)
                raise ValueError(
                    f"Unknown labels found in the mask. Unexpected labels: {unexpected_labels}"
                )
            yield SegmentationSample(image=image, mask=mask, sample_id=image_path)

    def __len__(self):
        return len(self.samples_fn)

    def __getitem__(self, item):
        image_path, mask_path = self.samples_fn[item]

        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path)

        image = np.array(image)
        mask = np.array(mask)

        if not np.isin(mask, self.known_labels).all():
            unexpected_labels = set(np.unique(mask)) - set(self.known_labels)
            raise ValueError(
                f"Unknown labels found in the mask. Unexpected labels: {unexpected_labels}"
            )
        return SegmentationSample(image=image, mask=mask, sample_id=image_path)
