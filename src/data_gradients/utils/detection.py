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
