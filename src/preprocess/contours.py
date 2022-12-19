from typing import List, Tuple, Dict

import cv2.cv2 as cv2
import numpy as np
import torch


def get_contours(label: torch.Tensor) -> np.array:
    """
    Find contours in each class-channel individually, using opencv findContours method
    :param label: Tensor [N, W, H] where N is number of valid classes
    :return: List with the shape [N, Nc, P, 1, 2] where N is number of valid classes, Nc are number of contours
    per class, P are number of points for each contour and (1, 2) are set of points.
    """
    # Tensor to numpy (for opencv usage)
    label = label.numpy()
    # Type to INT8 as for Index array
    label = label.astype(np.uint8)

    all_onehot_contour = []
    # For each class
    for class_channel in range(label.shape[0]):
        # Get tensor [class, W, H]
        onehot = label[class_channel, ...]
        # Find contours and return shape of [N, P, 1, 2] where N is number of contours and P list of points
        onehot_contour, _ = cv2.findContours(onehot, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # Check if contour is OK
        valid_onehot_contours = get_valid_contours(onehot_contour)
        if len(valid_onehot_contours):
            # Attach to the Channel dim to get a [C, N, P, 1, 2] Tensor
            all_onehot_contour.append(valid_onehot_contours)

    # Return tensor with [N, Nc, P, 1, 2] where N is # of classes, Nc # of contours per class and
    return all_onehot_contour


def get_valid_contours(contours: Tuple) -> List:
    """
    Contours sometimes are buggy, as for 2-points-contour, a stragith line, etc.
    We'll remove the by the valid - criteria (temporary) - minimal size of (3 ^ 2) pixels
    :param contours: Any list of contours
    :return: Valid list of contours
    """
    minimal_contour_size = 9
    valid_contours = [c for c in contours if get_contour_area(c) > minimal_contour_size]
    return valid_contours


def get_num_contours(contours: List[np.array]) -> int:
    return len(contours)


def get_contour_area(contour: np.array) -> float:
    """
    Get area of contours [pixels]
    :param contour: List of points
    :return: number of pixels inside contour (integer)
    """
    area = cv2.contourArea(contour)
    return float(area)


def get_contour_center_of_mass(contour: np.array) -> Tuple[int, int]:
    """
    Find contours center of mass by its moments
    :param contour: List of points
    :return: X, Y pixels of contour's center of mass
    """
    moments = cv2.moments(contour)
    cx = int(moments['m10'] / float(moments['m00']))
    cy = int(moments['m01'] / float(moments['m00']))
    return cx, cy


def get_contour_perimeter(contour: np.array) -> float:
    return cv2.arcLength(contour, closed=True)


def get_contour_is_convex(contour: np.array) -> bool:
    return cv2.isContourConvex(contour)


def get_rotated_bounding_rect(contour: np.array) -> np.array:
    """
    Get the minimum area bounding rectangle of the contour given
    :param contour: List of points
    :return: [[Center X, Center Y], [Width, Height], [Box angle]]
    """
    rect = cv2.minAreaRect(contour)  # rect = ((cx, cy), (w, h), rotated angle)
    # return rect_to_box(rect)
    return rect


def rect_to_box(rect):
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    return box


def get_aspect_ratio_of_bounding_rect(contour: np.array) -> float:
    rect = cv2.minAreaRect(contour)  # rect = ((cx, cy), (w, h), rotated angle)
    return rect[1][0] / rect[1][1]


def get_extreme_points(contour: np.array) -> Dict:
    extreme_points = dict()
    extreme_points["leftmost"] = tuple(contour[contour[:, :, 0].argmin()][0])
    extreme_points["rightmost"] = tuple(contour[contour[:, :, 0].argmax()][0])
    extreme_points["topmost"] = tuple(contour[contour[:, :, 1].argmin()][0])
    extreme_points["bottommost"] = tuple(contour[contour[:, :, 1].argmax()][0])
    return extreme_points
