from typing import Tuple

import numpy as np
import torch


def cxcywh_to_xyxy(bboxes: torch.Tensor) -> torch.Tensor:
    """Transform bboxes from CX-CY-W-H format to XYXY format.

    :param bboxes:  BBoxes of shape (..., 4) in CX-CY-W-H format
    :return:        BBoxes of shape (..., 4) in XYXY format
    """
    cx, cy, w, h = bboxes[..., 0], bboxes[..., 1], bboxes[..., 2], bboxes[..., 3]
    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = x1 + w
    y2 = y1 + h

    return torch.stack([x1, y1, x2, y2], dim=-1)


def xywh_to_xyxy(bboxes: torch.Tensor) -> torch.Tensor:
    """Transform bboxes from XYWH format to XYXY format.

    :param bboxes:  BBoxes of shape (..., 4) in XYWH format
    :return:        BBoxes of shape (..., 4) in XYXY format
    """
    x1, y1, w, h = bboxes[..., 0], bboxes[..., 1], bboxes[..., 2], bboxes[..., 3]
    x2 = x1 + w
    y2 = y1 + h

    return torch.stack([x1, y1, x2, y2], dim=-1)


XYXY_CONVERTERS = {
    "xyxy": {"function": lambda x: x, "description": "xyxy: x-left, y-top, x-right, y-bottom"},
    "xywh": {"function": xywh_to_xyxy, "description": "xywh: x-left, y-top, width, height"},
    "cxcywh": {"function": cxcywh_to_xyxy, "description": "cxcywh: x-center, y-center, width, height"},
}


class XYXYConvertError(Exception):
    ...


class XYXYConverter:
    def __init__(self, format_name: str):
        if format_name not in XYXY_CONVERTERS:
            raise ValueError(f"`{format_name}` is not a supported bounding box format. It should be one of {list(XYXY_CONVERTERS.keys())}")
        self.converter = XYXY_CONVERTERS[format_name]["function"]

    def __call__(self, bboxes: torch.Tensor) -> torch.Tensor:
        try:
            return self.converter(bboxes)
        except Exception as e:
            raise XYXYConvertError(f"{e}:\n \t => Error happened when converting tensors to xyxy format.") from e

    @staticmethod
    def get_available_options():
        return {info["description"]: key for key, info in XYXY_CONVERTERS.items()}


def scale_bboxes(old_shape: Tuple[float, float], new_shape: Tuple[float, float], bboxes_xyxy: np.ndarray):
    """Scale bounding boxes to a new shape.
    :param old_shape:   Old shape of the image, (H, W) format
    :param new_shape:   New shape of the image, (H, W) format
    :param bboxes_xyxy: Bounding boxes in xyxy format
    """

    scales = np.array(
        [
            [
                new_shape[1] / old_shape[1],  # X1
                new_shape[0] / old_shape[0],  # Y1
                new_shape[1] / old_shape[1],  # X2
                new_shape[0] / old_shape[0],  # Y2
            ],
        ]
    )

    # apply scaling to the bounding box coordinates
    bboxes_xyxy_scaled = bboxes_xyxy * scales

    return bboxes_xyxy_scaled
