from typing import List, Tuple, Set
import matplotlib.pyplot as plt
import cv2

import numpy as np


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
    classes_in_image_with_color = set()

    for (x1, y1, x2, y2), class_id in zip(bboxes_xyxy, bboxes_ids):
        class_name = class_names[class_id]
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

    image = resize_keep_aspect_ratio(image, target_shape=(400, 400))
    # Draw the class names on the side of the image
    image = draw_class_names(image=image, classes_in_image_with_color=classes_in_image_with_color)

    return image


def resize_keep_aspect_ratio(image: np.ndarray, target_shape: tuple):
    """
    Resizes a single image to fit within a specified width while maintaining the aspect ratio.
    If the resulting height is below 400 pixels, the image won't be padded. The width is padded
    with center alignment, and the top is padded with white color to maintain the aspect ratio.

    Args:
        image (np.ndarray): The input image.
        target_shape (tuple): A tuple (width, height) specifying the desired dimensions.

    Returns:
        np.ndarray: The resized and padded image.

    """
    image_height, image_width = image.shape[:2]
    target_width, target_height = target_shape

    scale_factor = min(target_width / image_width, target_height / image_height)
    new_width = int(image_width * scale_factor)
    new_height = int(image_height * scale_factor)
    resized_image = cv2.resize(image, (new_width, new_height))

    canvas = np.full((target_height, target_width, 3), 255, dtype=np.uint8)
    x = int((target_width - new_width) / 2)
    y = int(target_height - new_height)
    canvas[y : y + new_height, x : x + new_width] = resized_image

    return canvas


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


def draw_class_names(image: np.ndarray, classes_in_image_with_color: Set[Tuple[str, tuple]]) -> np.ndarray:
    """Draw class names below the image.

    :param image: Image on which to draw the class names.
    :param classes_in_image_with_color: Set of tuples containing class names and their corresponding colors.
    :return: Image with class names.
    """

    # Create a blank canvas for the class names
    canvas_height_ratio = 0.1  # Ratio of canvas height to image height
    canvas = np.ones((int(image.shape[0] * canvas_height_ratio), image.shape[1], 3), dtype=np.uint8) * 255

    # Define margin and calculate font size and line type based on number of classes
    margin = 20  # Margin from border
    font_size = 1
    line_type = max(int(font_size * 2), 1)  # Line type, scaled with font size

    # Prepare for adding class names
    line_y = margin + int(canvas.shape[0] * 0.1)
    current_x = margin

    for class_name, class_color in sorted(classes_in_image_with_color, key=lambda x: x[0]):

        # Calculate width of the new class name
        text_width, text_height = cv2.getTextSize(class_name, cv2.FONT_HERSHEY_SIMPLEX, font_size, line_type)[0]

        # Check if the new class name will fit on the current line
        if current_x + text_width + 2 * margin > canvas.shape[1]:
            # Start a new line
            line_y += int(canvas.shape[0] * 0.1)
            current_x = margin

        # Create a semi-transparent background for the class name
        # background_color = [int(c * alpha + 255 * (1 - alpha)) for c in class_color]  # Semi-transparent class color
        background_top_left = (current_x - margin // 2, line_y - text_height - margin // 2)
        background_bottom_right = (current_x + text_width + margin // 2, line_y + margin // 2)
        canvas = cv2.rectangle(canvas, background_top_left, background_bottom_right, class_color, -1)

        # Add class_name
        canvas = cv2.putText(
            canvas, class_name, (current_x, line_y), cv2.FONT_HERSHEY_SIMPLEX, font_size, best_text_color(class_color), line_type, lineType=cv2.LINE_AA
        )

        # Update current position on the x axis
        current_x += text_width + margin

    # Concatenate the image and the canvas
    image = np.concatenate((image, canvas), axis=0)
    return image


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
