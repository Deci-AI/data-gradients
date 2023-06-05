from typing import Tuple, Optional, Callable, List

import torch
from torch import Tensor

from data_gradients.batch_processors.utils import check_all_integers
from data_gradients.batch_processors.formatters.base import BatchFormatter
from data_gradients.batch_processors.formatters.utils import ensure_images_shape, ensure_channel_first, drop_nan
from data_gradients.utils.utils import ask_user


class UnsupportedDetectionBatchFormatError(Exception):
    def __init__(self, batch_format: tuple):
        grouped_batch_format = "(Batch_size x padding_size x 5) with 5: (class_id + 4 bbox coordinates))"
        flat_batch_format = "(N, 6) with 6: (image_id + class_id + 4 bbox coordinates)"
        super().__init__(
            f"Supported format for detection is not supported. Supported formats are:\n- {grouped_batch_format}\n- {flat_batch_format}\n Got: {batch_format}"
        )


class DetectionBatchFormatter(BatchFormatter):
    """Detection formatter class"""

    def __init__(
        self,
        class_names: List[str],
        class_names_to_use: List[str],
        n_image_channels: int,
        xyxy_converter: Optional[Callable[[Tensor], Tensor]] = None,
        label_first: Optional[bool] = None,
    ):
        """
        :param class_names:         List of all class names in the dataset. The index should represent the class_id.
        :param class_names_to_use:  List of class names that we should use for analysis.
        :param n_image_channels:    Number of image channels (3 for RGB, 1 for Gray Scale, ...)
        :param xyxy_converter:      Function to convert the bboxes to the `xyxy` format.
        :param label_first:         Whether the annotated_bboxes states with labels, or with the bboxes. (typically label_xyxy vs xyxy_label)
        """
        class_names_to_use = set(class_names_to_use)
        self.class_ids_to_use = [class_id for class_id, class_name in enumerate(class_names) if class_name in class_names_to_use]

        self.n_image_channels = n_image_channels
        self.xyxy_converter = xyxy_converter
        self.label_first = label_first

    def format(self, images: Tensor, labels: Tensor) -> Tuple[Tensor, List[Tensor]]:
        """Validate batch images and labels format, and ensure that they are in the relevant format for detection.

        :param images: Batch of images, in (BS, ...) format
        :param labels: Batch of labels, in (BS, N, 5) format
        :return:
            - images: Batch of images already formatted into (BS, C, H, W)
            - labels: List of bounding boxes, each of shape (N_i, 5 [label_xyxy]) with N_i being the number of bounding boxes with class_id in class_ids
        """
        labels = drop_nan(labels)

        images = ensure_channel_first(images, n_image_channels=self.n_image_channels)
        images = ensure_images_shape(images, n_image_channels=self.n_image_channels)
        labels = self.ensure_labels_shape(annotated_bboxes=labels)

        if self.label_first is None or self.xyxy_converter is None:
            self.show_annotated_bboxes(annotated_bboxes=labels)

        if self.label_first is None:
            self.label_first = self.ask_user_is_label_first()

        if self.xyxy_converter is None:
            self.xyxy_converter = self.ask_user_xyxy_converter()

        if 0 <= images.min() and images.max() <= 1:
            images *= 255
            images = images.to(torch.uint8)

        labels = self.convert_to_label_xyxy(
            annotated_bboxes=labels,
            image_shape=images.shape[-2:],
            xyxy_converter=self.xyxy_converter,
            label_first=self.label_first,
        )

        labels = self.filter_non_relevant_annotations(bboxes=labels, class_ids_to_use=self.class_ids_to_use)

        return images, labels

    @staticmethod
    def ensure_labels_shape(annotated_bboxes: Tensor) -> Tensor:
        """Make sure that the labels have the correct shape, i.e. (BS, N, 5)."""
        if annotated_bboxes.ndim == 2:
            if annotated_bboxes.shape[-1] != 6:
                raise UnsupportedDetectionBatchFormatError(batch_format=annotated_bboxes.shape)
            else:
                return group_detection_batch(annotated_bboxes)
        elif annotated_bboxes.ndim != 3 or annotated_bboxes.shape[-1] != 5:
            raise UnsupportedDetectionBatchFormatError(batch_format=annotated_bboxes.shape)
        else:
            return annotated_bboxes

    @staticmethod
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

        if xyxy_bboxes.max().item() < 2:
            h, w = image_shape
            bboxes[..., 0::2] *= w
            bboxes[..., 1::2] *= h

        return torch.cat([labels, xyxy_bboxes], dim=-1)

    @staticmethod
    def show_annotated_bboxes(annotated_bboxes: Tensor) -> None:
        """Show an example of the annotated bounding boxes."""
        print("\n========================================================================")
        print("SAMPLE BOUNDING BOXES")
        print("========================================================================")
        print("Here's a sample of how your labels look like:")
        print("Each line corresponds to a bounding box, with the format you specified earlier.")
        print(annotated_bboxes[0, :3, :])
        print("")

    @staticmethod
    def ask_user_is_label_first() -> bool:
        is_label_first_descriptions = {
            "Label comes first (e.g. [class_id, x1, y1, x2, y2])": True,
            "Bounding box comes first (e.g. [x1, y1, x2, y2, class_id])": False,
        }
        selected_option = ask_user(
            main_question="Which comes first in your annotations, the class id or the bounding box?",
            options=list(is_label_first_descriptions.keys()),
        )
        return is_label_first_descriptions[selected_option]

    @staticmethod
    def ask_user_xyxy_converter() -> Callable[[Tensor], Tensor]:
        xyxy_converter_descriptions = {
            "xyxy: x- left, y-top, x-right, y-bottom": lambda x: x,
            "xywh: x-left, y-top, width, height": DetectionBatchFormatter.xywh_to_xyxy,
            "cxcywh: x-center, y-center, width, height": DetectionBatchFormatter.cxcywh_to_xyxy,
        }
        selected_option = ask_user(
            main_question="What is the format of the bounding boxes?",
            options=list(xyxy_converter_descriptions.keys()),
        )
        return xyxy_converter_descriptions[selected_option]

    @staticmethod
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

    @staticmethod
    def xywh_to_xyxy(bboxes: Tensor) -> Tensor:
        """Transform bboxes from XYWH format to XYXY format.

        :param bboxes:  BBoxes of shape (..., 4) in XYWH format
        :return:        BBoxes of shape (..., 4) in XYXY format
        """
        x1, y1, w, h = bboxes[..., 0], bboxes[..., 1], bboxes[..., 2], bboxes[..., 3]
        x2 = x1 + w
        y2 = y1 + h

        return torch.stack([x1, y1, x2, y2], dim=-1)

    @staticmethod
    def filter_non_relevant_annotations(bboxes: torch.Tensor, class_ids_to_use: List[int]) -> List[torch.Tensor]:
        """Filter the bounding box tensors to keep only the ones with relevant label; also removes padding.

        :param bboxes:              Bounding box tensors with shape [batch_size, padding_size, 5], where 5 represents (label, x, y, x, y).
        :param class_ids_to_use:    List of class ids to keep use.
        :return: List of filtered bounding box tensors, each of shape [n_bbox, 5],
                 where n_bbox is the number of bounding boxes with a label in the `valid_labels` list.
        """
        filtered_bbox_tensors = []
        class_ids_to_use_tensor = torch.tensor(class_ids_to_use)

        for bbox_tensor in bboxes:
            valid_indices = torch.nonzero(torch.isin(bbox_tensor[:, 0], class_ids_to_use_tensor)).squeeze()
            filtered_bbox = bbox_tensor[valid_indices]

            non_zero_indices = torch.any(filtered_bbox[:, 1:] != 0, dim=1)
            filtered_bbox = filtered_bbox[non_zero_indices]

            filtered_bbox_tensors.append(filtered_bbox)

        return filtered_bbox_tensors


def group_detection_batch(flat_batch: torch.Tensor) -> torch.Tensor:
    """Convert a flat batch of detections (N, 6) into a grouped batch of detections (B, P, 4)

    :param flat_batch: Flat batch of detections (N, 6) with 6: (image_id + class_id + 4 bbox coordinates)
    :return: Grouped batch of detections (B, P, 5) with:
                B: Batch size
                P: Padding size
                5: (class_id + 4 bbox coordinates)
    """
    batch_size = int(torch.max(flat_batch[:, 0])) + 1
    batch_targets = [[] for _ in range(batch_size)]

    for target in flat_batch:
        image_id, target = target[0].item(), target[1:]
        batch_targets[int(image_id)].append(target)

    max_n_labels_per_image = max(len(labels) for labels in batch_targets)

    output_array = torch.zeros(batch_size, max_n_labels_per_image, 5)

    for batch_index, targets in enumerate(batch_targets):
        for target_index, target in enumerate(targets):
            output_array[batch_index, target_index] = target

    return output_array
