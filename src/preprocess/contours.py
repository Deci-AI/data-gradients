from typing import List, Tuple, Dict

import cv2.cv2 as cv2
import numpy as np
import torch
# find contours
# CHAIN_APPROX_SIMPLE reduces number of un-needed points (i.e., straight line)
# CHAIN_APPROX_NONE keep all points
# RETR_LIST simply retrievs all contours
# RETR_EXTERNAL get all outer contours only
# RERT_TREE gets all contours + "family" details


def get_contours(label: torch.Tensor, debug_image=None) -> np.array:
    """

    :param label: Tensor [C, W, H] where C is number of classes
    :param debug_image: Optional, debug purposes
    :return:
    """
    if debug_image is not None:
        debug_image = debug_image.numpy()
        debug_image = debug_image.transpose(1, 2, 0)
        debug_image = cv2.cvtColor(debug_image, cv2.COLOR_RGB2BGR)
        cv2.imshow("orig_image", debug_image)
        # orig_image_shape = (512, 512)

    label = label.numpy().astype(np.uint8)

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

    if debug_image is not None:
        img_contours = np.zeros((512, 512, 3))
        for i, contours in enumerate(all_onehot_contour):
            color = (0, 0, 255) if i == 0 else ((0, 255, 0) if i == 1 else ((255, 0, 0) if i == 2 else (127, 0, 255)))
            img_contours = cv2.drawContours(img_contours, contours, -1, color, 1)
            single_img_contours = np.zeros((512, 512, 3))
            single_img_contours = cv2.drawContours(single_img_contours, contours, -1, color, 1)
            for c in contours:
                box = get_rotated_bounding_rect(c)
                single_img_contours = cv2.drawContours(single_img_contours, [box], -1, (0, 255, 0), 1)
                single_img_contours = cv2.circle(single_img_contours, get_contour_center_of_mass(c), 5, (255, 255, 255), 2)
            print(f"Class is: {np.delete((np.unique(label[i, ... ])), 0)} got {len(contours)} contours")
            cv2.imshow("single", single_img_contours)
            cv2.waitKey(0)

        cv2.imshow("img_contours", img_contours)
        q = cv2.waitKey(0)
        if q == ord('q'):
            exit(0)

    # Return tensor with [C, N, P, 1, 2] where C is # of classes, N # of contours per class and
    # (P, 1, 2) is points vector
    return all_onehot_contour


def get_valid_contours(contours: Tuple) -> List:
    # TODO: Who is valid? minimal contour size for now
    minimal_contour_size = 8
    valid_contours = [c for c in contours if get_contour_area(c) > minimal_contour_size]
    return valid_contours


def get_contour_class(contour: np.array, label: np.array) -> int:
    centers, _, _ = cv2.minAreaRect(contour)
    centers = np.array(centers).astype(np.uint16)
    return int(label[0][centers[1]][centers[0]] * 255)


def get_num_contours(contours: List[np.array]) -> int:
    return len(contours)


def get_contour_area(contour: np.array) -> int:
    area = cv2.contourArea(contour)
    return int(area)


def get_contour_center_of_mass(contour: np.array) -> Tuple[int, int]:
    moments = cv2.moments(contour)
    area = float(moments['m00'])
    if area <= 0:
        return -1, -1
    cx = int(moments['m10'] / area)
    cy = int(moments['m01'] / area)
    return cx, cy


def get_contour_perimeter(contour: np.array) -> float:
    return cv2.arcLength(contour, closed=True)


def get_contour_is_convex(contour: np.array) -> bool:
    return cv2.isContourConvex(contour)


def get_rotated_bounding_rect(contour: np.array) -> np.array:
    rect = cv2.minAreaRect(contour)  # rect = ((cx, cy), (w, h), rotated angle)
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
