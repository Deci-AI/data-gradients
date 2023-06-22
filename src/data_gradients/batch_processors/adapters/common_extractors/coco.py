import PIL.Image
import numpy as np
import torch
from pydantic import BaseModel
from pydantic.error_wrappers import ValidationError
from typing import List, Union, Tuple, Any, Sequence, Dict

coco_image_type = Union[torch.Tensor, PIL.Image.Image, np.ndarray]
coco_annotation_type = Dict[str, Union[float, int, bool, List[float]]]


class CocoAnnotation(BaseModel):
    area: float
    iscrowd: int
    image_id: int
    bbox: List[float]
    category_id: int
    id: int


class CocoAnnotations(BaseModel):
    __root__: List[CocoAnnotation]


data = [
    {
        "area": 120057.13925,
        "iscrowd": 0,
        "image_id": 9,
        "bbox": [1.08, 187.69, 611.59, 285.84],
        "category_id": 51,
        "id": 1038967,
    },
    {
        "area": 44434.751099999994,
        "iscrowd": 0,
        "image_id": 9,
        "bbox": [311.73, 4.31, 319.28, 228.68],
        "category_id": 51,
        "id": 1039564,
    },
    {
        "area": 49577.94434999999,
        "iscrowd": 0,
        "image_id": 9,
        "bbox": [249.6, 229.27, 316.24, 245.08],
        "category_id": 56,
        "id": 1058555,
    },
]


def check_coco_dataset_format(data: Any) -> bool:
    """Check if input data corresponds to Coco standard annotation format.

    :param data: Raw data of your data iterable
    :return: True, if data follows Coco standard annotation format.
    """
    if not isinstance(data, Sequence) or len(data) <= 1:
        return False
    image, annotations = data[:2]

    if not isinstance(image, (torch.Tensor, PIL.Image.Image, np.ndarray)):
        return False

    try:
        _ = CocoAnnotations.parse_obj(annotations)
    except ValidationError:
        return False

    return True


def coco_detection_labels_extractor(coco_sample: Tuple[coco_image_type, List[coco_annotation_type]]) -> torch.Tensor:
    _image, annotations = coco_sample[:2]
    labels = []
    for annotation in annotations:
        if annotation.get("iscrowd", 0) == 0:
            class_id = annotation["category_id"]
            bbox = annotation["bbox"]
            labels.append((class_id, *bbox))
    return torch.Tensor(labels)


def coco_detection_labels_extractor_with_crowd(coco_sample: Tuple[coco_image_type, List[coco_annotation_type]]) -> torch.Tensor:
    _image, annotations = coco_sample[:2]
    labels = []
    for annotation in annotations:
        class_id = annotation["category_id"]
        bbox = annotation["bbox"]
        labels.append((class_id, *bbox))
    return torch.Tensor(labels)
