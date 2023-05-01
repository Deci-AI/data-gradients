from typing import Tuple

import torch
from torch import Tensor

from data_gradients.batch_processors.formatters.base import BatchFormatter
from data_gradients.batch_processors.formatters.utils import ensure_images_shape, ensure_channel_first, drop_nan


class DetectionBatchFormatter(BatchFormatter):
    """Detection formatter class"""

    def __init__(self, n_image_channels: int):
        """
        :param n_image_channels:    Number of image channels (3 for RGB, 1 for Gray Scale, ...)
        """
        self.n_image_channels = n_image_channels

    def format(self, images: Tensor, labels: Tensor) -> Tuple[Tensor, Tensor]:
        """Validate batch images and labels format, and ensure that they are in the relevant format for segmentation.

        :param images: Batch of images, in (BS, ...) format
        :param labels: Batch of labels, in (BS, N, [Labels, C, Y, X, Y]) format
        :return:
            - images: Batch of images already formatted into (BS, C, W, H)
            - labels: Batch of labels already formatted into (BS, N, [Labels, C, Y, X, Y])
        """
        labels = drop_nan(labels)

        images = ensure_channel_first(images, n_image_channels=self.n_image_channels)
        images = ensure_images_shape(images, n_image_channels=self.n_image_channels)

        return images, labels


def convert_bbox_format(bbox_tensor):
    """
    A function that converts a tensor of bounding boxes to the 'label_xyxy' format.

    Parameters:
    bbox_tensor (torch.Tensor): A tensor representing the bounding boxes with labels in the format (label, x1, y1, x2, y2).

    Returns:
    torch.Tensor: A tensor representing the bounding boxes with labels in the 'label_xyxy' format.
    """
    # Define a dictionary that maps the input format to the conversion function
    conversion_functions = {
        "xyxy": lambda x: x,
        "xywh": lambda x: (x[0], x[1], x[0] + bbox[2], bbox[1] + bbox[3]),
        "cxcywh": lambda x: (x[0] - x[2] / 2, bbox[1] - bbox[3] / 2, bbox[0] + bbox[2] / 2, bbox[1] + bbox[3] / 2),
    }

    # Ask the user for the input format of the bounding boxes
    bbox_format = input("What is the format of the bounding boxes (xyxy, xywh, cxcywh)? ")
    input_format = input(f"Is it label_{bbox_format} or {bbox_format}_label?")

    # Check if the input format is valid
    if input_format not in conversion_functions:
        raise ValueError("Invalid input format.")

    # Convert the bounding boxes to the 'label_xyxy' format
    output_tensor = []
    for bbox in bbox_tensor:
        bbox_values = bbox[1:]
        bbox_xyxy = conversion_functions[input_format](tuple(bbox_values))
        output_tensor.append(("" + bbox[0],) + bbox_xyxy)

    # Convert the output tensor to a torch tensor
    output_tensor = torch.tensor(output_tensor)

    return output_tensor


def cxcywh_to_xyxy(bboxes):
    """
    Transforms bboxes from CX-CY-W-H format to XYXY format
    :param bboxes: BBoxes of shape (..., 4) in CX-CY-W-H format
    :return: BBoxes of shape (..., 4) in XYXY format
    """
    cx, cy, w, h = bboxes[..., 0], bboxes[..., 1], bboxes[..., 2], bboxes[..., 3]
    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = x1 + w
    y2 = y1 + h

    return torch.stack([x1, y1, x2, y2], dim=-1)


def xywh_to_xyxy(bboxes):
    """
    Transforms bboxes from XYWH format to XYXY format
    :param bboxes: BBoxes of shape (..., 4) in XYWH format
    :return: BBoxes of shape (..., 4) in XYXY format
    """
    x1, y1, w, h = bboxes[..., 0], bboxes[..., 1], bboxes[..., 2], bboxes[..., 3]
    x2 = x1 + w
    y2 = y1 + h

    return torch.stack([x1, y1, x2, y2], dim=-1)
