from typing import List, Optional

import numpy as np
from super_gradients.training.processing.processing import default_vit_imagenet_processing_params

from data_gradients.common.registry.registry import register_feature_extractor
from data_gradients.feature_extractors.abstract_feature_extractor import Feature
from data_gradients.utils.data_classes import DetectionSample
from data_gradients.visualize.plot_options import HeatmapOptions
from data_gradients.feature_extractors.abstract_feature_extractor import AbstractFeatureExtractor
from super_gradients.training import models
from super_gradients.common.object_names import Models
import torch
from torchvision.ops import box_iou


@register_feature_extractor("DetectionClassSimilarity")
class DetectionClassSimilarity(AbstractFeatureExtractor):
    """
    Analyzes and visualizes the similarity of class instances across dataset splits using a pre-trained Vision Transformer model.

    Attributes:
        model (torch.nn.Module): The pre-trained Vision Transformer model.
        image_preprocessor (Callable): Function to preprocess images for model input.
        features (List[torch.Tensor]): Extracted feature vectors from detected objects.
        instances_class_ids (List[int]): Class IDs corresponding to each feature vector.
        all_classes_list (List[str]): All unique class names encountered in the dataset.
        iou_threshold (float): Threshold for Intersection over Union (IoU) to exclude overlapping bounding boxes.
    """

    def __init__(self, iou_threshold: Optional[float] = 0.1):
        """
        Initializes the feature extractor with a pre-trained Vision Transformer model and an IoU threshold for bounding box filtering.

        :param iou_threshold: Optional[float]. IoU threshold to exclude overlapping bounding boxes. If None, no filtering is applied.
        """
        self.model = models.get(Models.VIT_BASE, pretrained_weights="imagenet")
        self.model.eval()
        self.model.backbone_mode = True
        self.image_preprocessor = default_vit_imagenet_processing_params()["image_processor"]
        self.features = []
        self.instances_class_ids = []
        self.all_classes_list = None
        self.iou_threshold = iou_threshold

    def extract_features(self, cropped_images: List[np.ndarray]):
        """
        Extracts feature vectors from a list of cropped images using the pre-trained model.

        :param cropped_images: List[np.ndarray]. List of cropped images from detected objects.
        :return: List[torch.Tensor]. Extracted feature vectors for each image.
        """
        cropped_images = [c.astype(np.uint8)[:, :, ::-1] for c in cropped_images]
        preprocessed_cropped_images = [self.image_preprocessor.preprocess_image(cropped_image)[0] for cropped_image in cropped_images]
        preprocessed_cropped_images = torch.tensor(np.array(preprocessed_cropped_images))
        with torch.no_grad():
            extracted_features = self.model(preprocessed_cropped_images)
        return extracted_features

    def update(self, sample: DetectionSample):
        """
        Updates the feature extractor with data from a new sample, extracting features from detected objects.

        :param sample: DetectionSample. The new sample to process, containing image data and bounding boxes.
        """
        image = sample.image  # np.ndarray of shape [H, W, C] - The image as a numpy array with channels last
        image_height, image_width = image.shape[:2]
        cropped_images = []
        class_ids = []
        iou_matrix = None

        if self.all_classes_list is None:
            self.all_classes_list = sample.class_names
        if self.iou_threshold is not None:
            bboxes = torch.tensor(sample.bboxes_xyxy)
            iou_matrix = box_iou(bboxes, bboxes)

        for idx, (class_id, bbox_xyxy) in enumerate(zip(sample.class_ids, sample.bboxes_xyxy)):
            x1, y1, x2, y2 = map(int, bbox_xyxy)  # Convert to integer if necessary
            x1, x2, y1, y2 = self._clip_to_image_bounds(image_height, image_width, x1, x2, y1, y2)

            # Check if bbox coordinates are valid and area is at least 20 pixels
            if self._is_valid_coordinates(x1, x2, y1, y2) and not self._should_exclude_based_on_iou(iou_matrix, idx):
                cropped_image = image[y1:y2, x1:x2]  # Crop using numpy slicing
                cropped_images.append(cropped_image)
                class_ids.append(class_id)
        if len(cropped_images):
            extracted_features = self.extract_features(cropped_images)
            self.features.append(extracted_features)
            self.instances_class_ids.extend(class_ids)

    def _is_valid_coordinates(self, x1, x2, y1, y2) -> bool:
        """
        Determines if the bounding box coordinates are valid based on position and minimum area.

        :param x1: int. The left coordinate of the bounding box.
        :param x2: int. The right coordinate of the bounding box.
        :param y1: int. The top coordinate of the bounding box.
        :param y2: int. The bottom coordinate of the bounding box.
        :return: bool. True if the bounding box is valid, False otherwise.
        """
        return x1 < x2 and y1 < y2 and (x2 - x1) * (y2 - y1) >= 20

    def _clip_to_image_bounds(self, image_height, image_width, x1, x2, y1, y2) -> (int, int, int, int):
        """
        Clips bounding box coordinates to the image bounds.

        :param image_height: int. The height of the image.
        :param image_width: int. The width of the image.
        :param x1: int. The left coordinate of the bounding box.
        :param x2: int. The right coordinate of the bounding box.
        :param y1: int. The top coordinate of the bounding box.
        :param y2: int. The bottom coordinate of the bounding box.
        :return: Tuple[int, int, int, int]. The clipped bounding box coordinates.
        """
        x1 = max(0, min(x1, image_width - 1))
        y1 = max(0, min(y1, image_height - 1))
        x2 = max(0, min(x2, image_width - 1))
        y2 = max(0, min(y2, image_height - 1))
        return x1, x2, y1, y2

    def _should_exclude_based_on_iou(self, iou_matrix: torch.Tensor, idx: int) -> bool:
        """
        Determines if a bounding box should be excluded based on IoU with other bounding boxes.

        :param iou_matrix: torch.Tensor. The IoU matrix for all pairs of bounding boxes.
        :param idx: int. The index of the current bounding box in the IoU matrix.
        :return: bool. True if the bounding box should be excluded, False otherwise.
        """
        exclude = False
        if iou_matrix is not None:
            # Create a mask to exclude the current index (self-comparison)
            mask = torch.ones(iou_matrix.size(0), dtype=bool)
            mask[idx] = False

            # Check if there are no intersections at all
            if len(iou_matrix[idx, mask]) == 0:
                exclude = False
            else:
                # Check if the maximum IoU with any other box exceeds the threshold
                max_iou = iou_matrix[idx, mask].max().item()
                exclude = max_iou > self.iou_threshold
        return exclude

    def aggregate(self) -> Feature:
        """
        Aggregates extracted features to compute the class-to-class similarity and prepares the data for visualization.

        :return: Feature. The aggregated feature containing the similarity data and visualization details.
        """
        # Concatenate all features and class IDs
        all_features = torch.cat(self.features, dim=0)
        all_class_ids = torch.tensor(self.instances_class_ids, dtype=torch.int64)

        # Normalize the feature vectors
        norm_features = all_features / all_features.norm(dim=1, keepdim=True)

        # Initialize the similarity table
        num_classes = len(self.all_classes_list)
        similarity_table = torch.zeros((num_classes, num_classes))

        # Prepare the JSON object
        json_data = {}
        instances_count = np.zeros(len(self.all_classes_list))

        # Calculate the average similarity for each class pair
        for class_id_i in range(num_classes):
            for class_id_j in range(num_classes):
                indices_i = (all_class_ids == class_id_i).nonzero(as_tuple=True)[0]
                indices_j = (all_class_ids == class_id_j).nonzero(as_tuple=True)[0]
                class_pair_similarities = torch.mm(norm_features[indices_i], norm_features[indices_j].T)
                average_similarity = class_pair_similarities.mean().item() if class_pair_similarities.numel() > 0 else 0
                similarity_table[class_id_i, class_id_j] = average_similarity

                class_name_i = self.all_classes_list[class_id_i]
                class_name_j = self.all_classes_list[class_id_j]
                key = f"{class_name_i}-{class_name_j}"
                json_data[key] = {
                    "average_similarity": average_similarity,
                    f"{class_name_i}_instances": len(indices_i),
                    f"{class_name_j}_instances": len(indices_j),
                }
                instances_count[class_id_i] = len(indices_i)

        # Convert the similarity table to a numpy array
        similarity_table_np = similarity_table.numpy()

        num_clases_for_plot = min(num_classes, 15)

        data = {"All data": similarity_table_np[:num_clases_for_plot, :num_clases_for_plot]}

        # Prepare the plot options for the heatmap
        heatmap_options = HeatmapOptions(
            xticklabels=[self.all_classes_list[i] for i in range(num_clases_for_plot)],
            yticklabels=[self.all_classes_list[i] + f" ({int(instances_count[i])})" for i in range(num_clases_for_plot)],
            x_label_name="Class",
            y_label_name="Class, Instance Count",
            cbar=True,
            fmt=".2f",
            cmap="viridis",
            annot=True,
            square=True,
            figsize=(10, 10),
            tight_layout=True,
            x_ticks_rotation=90,
        )

        feature = Feature(
            data=data,
            plot_options=heatmap_options,
            json=json_data,
            title="Class-to-Class Similarity",
            description="A table showing the average cosine similarity between features of each class pair.",
        )

        return feature
