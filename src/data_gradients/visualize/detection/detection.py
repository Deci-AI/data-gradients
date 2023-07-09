from typing import List, Tuple, Set
import cv2

import numpy as np
from data_gradients.visualize.detection.detection_legend import draw_legend_on_canvas
from data_gradients.visualize.detection.utils import best_text_color, generate_color_mapping
from data_gradients.visualize.utils import resize_and_align_bottom_center


def draw_bboxes(image: np.ndarray, bboxes_xyxy: np.ndarray, bboxes_ids: np.ndarray, class_names: List[str]) -> np.ndarray:
    """Draw annotated bboxes on an image.

    :param image:       Input image tensor.
    :param bboxes_xyxy: BBoxes, in [N, 4].
    :param bboxes_ids:  Class ids [N].
    :param class_names: List of class names. (unique, not per bbox)
    :return:            Image with annotated bboxes.
    """
    if len(bboxes_ids) == 0:
        return image
    colors = generate_color_mapping(len(class_names) + 1)

    # Initialize an empty list to store the classes that appear in the image
    classes_in_image_with_color: Set[Tuple[str, Tuple]] = set()

    for (x1, y1, x2, y2), class_id in zip(bboxes_xyxy, bboxes_ids):
        class_name: str = class_names[class_id]
        color = colors[class_names.index(class_name)]

        # If the class is not already in the list, add it
        classes_in_image_with_color.add((class_name, color))

        image = draw_bbox(
            image=image,
            color=color,
            box_thickness=2,
            x1=x1,
            y1=y1,
            x2=x2,
            y2=y2,
        )

    image = resize_and_align_bottom_center(image, target_shape=(600, 600))

    canvas = draw_legend_on_canvas(image=image, class_color_tuples=classes_in_image_with_color)
    image = np.concatenate((image, canvas), axis=0)

    return image


def draw_bbox(
    image: np.ndarray,
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
    :param box_thickness:   Thickness of the bounding box border.
    :param x1:              x-coordinate of the top-left corner of the bounding box.
    :param y1:              y-coordinate of the top-left corner of the bounding box.
    :param x2:              x-coordinate of the bottom-right corner of the bounding box.
    :param y2:              y-coordinate of the bottom-right corner of the bounding box.
    :return: Image with bbox
    """
    overlay = image.copy()
    overlay = cv2.rectangle(overlay, (x1, y1), (x2, y2), color, box_thickness)
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
