from typing import Tuple

import numpy as np


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
