from typing import Tuple

import numpy as np
from PIL import Image, ImageDraw


def scale_bboxes(old_size: Tuple[float, float], new_size: Tuple[float, float], bboxes_xyxy: np.ndarray):
    # compute scale factors
    scales = np.array(
        [
            [
                new_size[1] / old_size[1],  # X1
                new_size[0] / old_size[0],  # Y1
                new_size[1] / old_size[1],  # X2
                new_size[0] / old_size[0],  # Y2
            ],
        ]
    )

    # apply scaling to the bounding box coordinates
    bboxes_xyxy_scaled = bboxes_xyxy * scales

    return bboxes_xyxy_scaled


def draw_bbox_on_image(image_size, bbox, fill_color="black"):
    # create new image
    img = Image.new("RGB", image_size, "white")
    draw = ImageDraw.Draw(img)

    # draw rectangle
    draw.rectangle(bbox, outline=fill_color, fill=fill_color)

    return img
