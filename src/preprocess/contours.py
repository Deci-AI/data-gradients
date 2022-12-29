from typing import List, Tuple, Dict

import cv2
import numpy as np
import torch

def debug_convexity_things(labels, images):
    for label, image in zip(labels, images):
        drawing = image.numpy()
        drawing = drawing.transpose(1, 2, 0)
        drawing = cv2.cvtColor(drawing, cv2.COLOR_RGB2BGR)
        cv2.imshow("original", drawing)

        label = label.numpy().astype(np.uint8)
        for class_channel in range(label.shape[0]):
            onehot = label[class_channel, ...]
            convex_hull = []
            bbox = []
            onehot_contours, hierarchy = cv2.findContours(onehot, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for i in range(len(onehot_contours)):
                convex_hull.append(get_convex_hull(onehot_contours[i]))
                bbox.append(rect_to_box(get_rotated_bounding_rect(onehot_contours[i])))

            for i in range(len(onehot_contours)):
                img_contours = np.zeros_like(drawing)
                cv2.drawContours(img_contours, onehot_contours, i, (0, 255, 0), 1, 8, hierarchy)
                cv2.putText(img_contours, "Contour", (20, 20), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
                cv2.drawContours(img_contours, convex_hull, i, (0, 0, 255), 1, 8)
                cv2.putText(img_contours, "Convex hull", (20, 40), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
                cv2.drawContours(img_contours, bbox, i, (255, 0, 0), 1, 8)
                cv2.putText(img_contours, "Bbox", (20, 60), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
                contour_area = round(get_contour_area(onehot_contours[i]))
                hull_area = round(get_contour_area(convex_hull[i]))
                if not hull_area > 0:
                    print(convex_hull[i], onehot_contours[i], contour_area, hull_area)
                    continue
                ratio = float(round(contour_area / hull_area, 3))
                cv2.putText(img_contours, f"Contour area - {contour_area}", (20, 80), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)
                cv2.putText(img_contours, f"HULL area - {hull_area}", (20, 100), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)
                cv2.putText(img_contours, f'Ratio - {ratio}', (20, 120), cv2.FONT_HERSHEY_PLAIN, 1,
                            (255, 255, 255), 1)

                hull = cv2.convexHull(onehot_contours[i], returnPoints=False)
                defects = cv2.convexityDefects(onehot_contours[i], hull)
                for defect in defects:
                    defect = defect[0]
                    startpoint = onehot_contours[i][defect[0]][0]
                    endpoint = onehot_contours[i][defect[1]][0]
                    farthestpoint = onehot_contours[i][defect[2]][0]
                    distance = defect[3]
                    if distance > 5000:
                        # cv2.line(img_contours, endpoint, startpoint, (255, 0, 255), 1)
                        print(endpoint, startpoint)
                        mid = [int(abs((endpoint[0] + startpoint[0])/2)), int(abs((endpoint[1] + startpoint[1]))/2)]
                        print(mid)
                        cv2.circle(img_contours, startpoint, 2, (255, 0, 255), 2)
                        cv2.circle(img_contours, endpoint, 2, (255, 0, 255), 2)
                        cv2.circle(img_contours, mid, 2, (255, 0, 255), 2)
                        cv2.line(img_contours, mid, farthestpoint, (0, 255, 255), 1)
                        cv2.putText(img_contours, f"{str(distance)}", mid, cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 255))
                    cv2.imshow(f"contouyr", img_contours)
                    cv2.waitKey(0)
def get_box_area(box):
    print(box)
    x1, y1 = box[0]
    x2, y2 = box[1]
    print(abs(y2-y1), abs(x2-x1))
    area = abs(y2-y1) * abs(x2-x1)
    return area

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
    if float(moments['m00']) < 10:
        return -1, -1
    cx = int(moments['m10'] / float(moments['m00']))
    cy = int(moments['m01'] / float(moments['m00']))
    return cx, cy


def get_contour_perimeter(contour: np.array) -> float:
    perimeter = cv2.arcLength(contour, closed=True)
    return perimeter


def get_contour_is_convex(contour: np.array) -> bool:
    return cv2.isContourConvex(contour)


def get_convex_hull(contour: np.array) -> np.array:
    convex_hull = cv2.convexHull(contour, hull=False)
    return convex_hull


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
