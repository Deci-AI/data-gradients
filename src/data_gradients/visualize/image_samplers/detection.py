from typing import List, Tuple
import matplotlib.pyplot as plt
import cv2

import numpy as np
import torch

from data_gradients.utils import DetectionBatchData
from data_gradients.visualize.image_samplers.base import ImageSampleManager


class DetectionImageSampleManager(ImageSampleManager):
    def update(self, data: DetectionBatchData) -> None:
        for image, labels, bboxes in zip(data.images, data.labels, data.bboxes):
            if len(self.samples) < self.n_samples:
                self.samples.append(draw_bboxes(image=image, labels=labels, bboxes=bboxes))


def draw_bboxes(image: torch.Tensor, labels: torch.Tensor, bboxes: torch.Tensor) -> torch.Tensor:
    """Draw annotated bboxes on an image.

    :param image:       Input image tensor.
    :param labels:      Labels in shape [N].
    :param bboxes:      Labels in shape [N, 4].
    :return:            Image with annotated bboxes.
    """
    image = image.cpu().numpy().copy().transpose(1, 2, 0)
    labels = labels.cpu().numpy()
    bboxes = bboxes.cpu().numpy()

    colors = generate_color_mapping(int(labels.max()) + 1)

    for label, bbox in zip(labels, bboxes):

        if (bbox == 0).all():
            pass

        image = draw_bbox(
            image=image,
            title=f"id: {int(label)}",
            color=colors[int(label)],
            box_thickness=2,
            x1=int(bbox[0]),
            y1=int(bbox[1]),
            x2=int(bbox[2]),
            y2=int(bbox[3]),
        )
    return torch.from_numpy(image.transpose((2, 0, 1)))


def draw_bbox(
    image: np.ndarray,
    title: str,
    color: Tuple[int, int, int],
    box_thickness: int,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
) -> np.ndarray:
    """Draw a bounding box on an image.

    :param image:           Image on which to draw the bounding box.
    :param color:           RGB values of the color of the bounding box.
    :param title:           Title to display inside the bounding box.
    :param box_thickness:   Thickness of the bounding box border.
    :param x1:              x-coordinate of the top-left corner of the bounding box.
    :param y1:              y-coordinate of the top-left corner of the bounding box.
    :param x2:              x-coordinate of the bottom-right corner of the bounding box.
    :param y2:              y-coordinate of the bottom-right corner of the bounding box.
    :return: Image with bbox
    """

    overlay = image.copy()
    overlay = cv2.rectangle(overlay, (x1, y1), (x2, y2), color, box_thickness)

    # Adapt font size to image shape.
    # This is required because small images require small font size, but this makes the title look bad,
    # so when possible we increase the font size to a more appropriate value.
    font_size = 0.25 + 0.07 * min(overlay.shape[:2]) / 100
    font_size = max(font_size, 0.5)  # Set min font_size to 0.5
    font_size = min(font_size, 0.8)  # Set max font_size to 0.8

    overlay = draw_text_box(image=overlay, text=title, x=x1, y=y1, font=2, font_size=font_size, background_color=color, thickness=1)

    return cv2.addWeighted(overlay, 0.75, image, 0.25, 0)


def draw_text_box(
    image: np.ndarray,
    text: str,
    x: int,
    y: int,
    font: int,
    font_size: float,
    background_color: Tuple[int, int, int],
    thickness: int = 1,
) -> np.ndarray:
    """Draw a text inside a box

    :param image:               The image on which to draw the text box.
    :param text:                The text to display in the text box.
    :param x:                   The x-coordinate of the top-left corner of the text box.
    :param y:                   The y-coordinate of the top-left corner of the text box.
    :param font:                The font to use for the text.
    :param font_size:           The size of the font to use.
    :param background_color:    The color of the text box and text as a tuple of three integers representing RGB values.
    :param thickness:           The thickness of the text.
    :return: Image with the text inside the box.
    """
    text_color = best_text_color(background_color)
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_size, thickness)
    text_left_offset = 7

    image = cv2.rectangle(image, (x, y), (x + text_width + text_left_offset, y - text_height - int(15 * font_size)), background_color, -1)
    image = cv2.putText(image, text, (x + text_left_offset, y - int(10 * font_size)), font, font_size, text_color, thickness, lineType=cv2.LINE_AA)
    return image


def best_text_color(background_color: Tuple[int, int, int]) -> Tuple[int, int, int]:
    """Determine the best color for text to be visible on a given background color.

    :param background_color: RGB values of the background color.
    :return: RGB values of the best text color for the given background color.
    """

    # If the brightness is greater than 0.5, use black text; otherwise, use white text.
    if compute_brightness(background_color) > 0.5:
        return (0, 0, 0)  # Black
    else:
        return (255, 255, 255)  # White


def compute_brightness(color: Tuple[int, int, int]) -> float:
    """Computes the brightness of a given color in RGB format. From https://alienryderflex.com/hsp.html

    :param color: A tuple of three integers representing the RGB values of the color.
    :return: The brightness of the color.
    """
    return (0.299 * color[0] + 0.587 * color[1] + 0.114 * color[0]) / 255


def generate_color_mapping(num_classes: int) -> List[Tuple[int, ...]]:
    """Generate a unique BGR color for each class

    :param num_classes: The number of classes in the dataset.
    :return:            List of RGB colors for each class.
    """
    cmap = plt.cm.get_cmap("gist_rainbow", num_classes)
    colors = [cmap(i, bytes=True)[:3][::-1] for i in range(num_classes)]
    return [tuple(int(v) for v in c) for c in colors]
