from typing import Tuple, Dict, Any, Optional, Callable

import torch
from torch import Tensor

from data_gradients.batch_processors.utils import check_all_integers
from data_gradients.batch_processors.formatters.base import BatchFormatter
from data_gradients.batch_processors.formatters.utils import ensure_images_shape, ensure_channel_first, drop_nan


class DetectionBatchFormatter(BatchFormatter):
    """Detection formatter class"""

    def __init__(self, n_image_channels: int, xyxy_converter: Optional[Callable[[Tensor], Tensor]] = None, label_first: Optional[bool] = None):
        """
        :param n_image_channels:    Number of image channels (3 for RGB, 1 for Gray Scale, ...)
        :param xyxy_converter:      Function to convert the bboxes to the `xyxy` format.
        :param label_first:         Whether the annotated_bboxes states with labels, or with the bboxes. (typically label_xyxy vs xyxy_label)
        """
        self.n_image_channels = n_image_channels
        self.xyxy_converter = xyxy_converter
        self.label_first = label_first

    def format(self, images: Tensor, labels: Tensor) -> Tuple[Tensor, Tensor]:
        """Validate batch images and labels format, and ensure that they are in the relevant format for detection.

        :param images: Batch of images, in (BS, ...) format
        :param labels: Batch of labels, in (BS, N, 5) format
        :return:
            - images: Batch of images already formatted into (BS, C, H, W)
            - labels: Batch of labels already formatted into (BS, N, 5 [label_xyxy])
        """
        labels = drop_nan(labels)

        images = ensure_channel_first(images, n_image_channels=self.n_image_channels)
        images = ensure_images_shape(images, n_image_channels=self.n_image_channels)
        labels = ensure_labels_shape(annotated_bboxes=labels)

        if self.label_first is None or self.xyxy_converter is None:
            show_annotated_bboxes(annotated_bboxes=labels)

        if self.label_first is None:
            self.label_first = ask_user_is_label_first()

        if self.xyxy_converter is None:
            self.xyxy_converter = ask_user_xyxy_converter()

        labels = convert_to_label_xyxy(annotated_bboxes=labels, image_shape=images.shape[-2:], xyxy_converter=self.xyxy_converter, label_first=self.label_first)
        return images, labels


def show_annotated_bboxes(annotated_bboxes: Tensor) -> None:
    """Show an example of the annotated bounding boxes."""
    print()
    print("This is how your bbox annotations look like:")
    print(annotated_bboxes[0, :3, :])


def ensure_labels_shape(annotated_bboxes: Tensor) -> Tensor:
    """Make sure that the labels have the correct shape, i.e. (BS, N, 5)."""
    if annotated_bboxes.ndim != 3:
        raise RuntimeError(
            f"The batch labels should have 3 dimensions (Batch_size x N_labels_per_image x 5 (label + 4 bbox coordinates)), but got {annotated_bboxes.ndim}"
        )

    if annotated_bboxes.shape[-1] != 5:
        raise RuntimeError(f"Labels last dimension should be 5 (label + 4 bbox coordinates), but got {annotated_bboxes.shape[-1]}")

    return annotated_bboxes


def convert_to_label_xyxy(annotated_bboxes: Tensor, image_shape: Tuple[int, int], xyxy_converter: Callable[[Tensor], Tensor], label_first: bool):
    """Convert a tensor of annotated bounding boxes to the 'label_xyxy' format.

    :param annotated_bboxes:    Annotated bounding boxes, in (BS, N, 5) format. Could be any format.
    :param image_shape:         Shape of the image, in (H, W) format.
    :param xyxy_converter:      Function to convert the bboxes to the `xyxy` format.
    :param label_first:         Whether the annotated_bboxes states with labels, or with the bboxes. (typically label_xyxy vs xyxy_label)
    :return:                    Annotated bounding boxes, in (BS, N, 5 [label_xyxy]) format.
    """
    annotated_bboxes = annotated_bboxes[:]

    if label_first:
        labels, bboxes = annotated_bboxes[..., :1], annotated_bboxes[..., 1:]
    else:
        labels, bboxes = annotated_bboxes[..., -1:], annotated_bboxes[..., :-1]

    if not check_all_integers(labels):
        raise RuntimeError(f"Labels should all be integers, but got {labels}")

    xyxy_bboxes = xyxy_converter(bboxes)

    if xyxy_bboxes.max().item() < 1:
        h, w = image_shape
        bboxes[..., 0::2] *= w
        bboxes[..., 1::2] *= h

    return torch.cat([labels, xyxy_bboxes], dim=-1)


def ask_user_xyxy_converter() -> Callable[[Tensor], Tensor]:
    xyxy_converter_descriptions = {
        lambda x: x: "xyxy: x- left, y-top, x-right, y-bottom",
        xywh_to_xyxy: "xywh: x-left, y-top, width, height",
        cxcywh_to_xyxy: "cxcywh: x-center, y-center, width, height",
    }
    xyxy_converter = ask_user(main_question="What is the format of the bounding boxes?", options_described=xyxy_converter_descriptions)
    return xyxy_converter


def ask_user_is_label_first() -> bool:
    is_label_first_descriptions = {
        True: "Start with label, followed with bboxes ([label, x1, y1, x2, y2] for instance)",
        False: "Start with bboxes, followed by labels ([x1, y1, x2, y2, label] for instance)",
    }
    is_label_first = ask_user(main_question='Are your annotations "label first" or "label last"?', options_described=is_label_first_descriptions)
    return is_label_first


def ask_user(main_question: str, options_described: Dict[Any, str]) -> Any:
    """Prompt the user to choose an option from a list of options.

    :param main_question:       The main question or instruction for the user.
    :param options_described:   Dictionary containing the options as keys and their descriptions as values.
    :return:                    The chosen option (key from the options_described dictionary).
    """
    options, options_descriptions = list(options_described.keys()), list(options_described.values())
    numbers_to_chose_from = [str(i) for i in range(len(options))]

    options_formatted = "\n".join([f"\t {number} | {option_description}" for number, option_description in zip(numbers_to_chose_from, options_descriptions)])

    user_answer = None
    while user_answer not in numbers_to_chose_from:
        print()
        if user_answer is not None:
            print(f'"{user_answer}" is not a valid option. Please chose a number between 0-{len(numbers_to_chose_from)}.')
        user_answer = input(f"{main_question}\n{options_formatted}\n (Write down the number) >>> ")

    return options[int(user_answer)]


def cxcywh_to_xyxy(bboxes: Tensor) -> Tensor:
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


def xywh_to_xyxy(bboxes: Tensor) -> Tensor:
    """Transform bboxes from XYWH format to XYXY format.

    :param bboxes:  BBoxes of shape (..., 4) in XYWH format
    :return:        BBoxes of shape (..., 4) in XYXY format
    """
    x1, y1, w, h = bboxes[..., 0], bboxes[..., 1], bboxes[..., 2], bboxes[..., 3]
    x2 = x1 + w
    y2 = y1 + h

    return torch.stack([x1, y1, x2, y2], dim=-1)
