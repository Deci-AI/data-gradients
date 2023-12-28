import unittest

import numpy as np
import pandas as pd

from data_gradients.feature_extractors.object_detection.bounding_boxes_area import DetectionBoundingBoxArea
from data_gradients.utils.data_classes.data_samples import DetectionSample
from data_gradients.utils.data_classes.image_channels import ImageChannels


class TestComputeHistogram(unittest.TestCase):
    def test_compute_histogram(self):
        test_df = pd.DataFrame(
            {
                "bbox_area_sqrt": [1, 2, 3, 3, 3, 2, 3],
                "split": ["train", "train", "train", "train", "val", "val", "val"],
                "class_name": ["A", "B", "A", "A", "B", "A", "C"],
            }
        )

        result = DetectionBoundingBoxArea._compute_histogram(test_df, name="bbox_area_sqrt")

        expected_result = {
            "train": {"name": "bbox_area_sqrt", "bin_width": 1, "max_value": 3, "histograms": {"A": [0, 1, 0, 2], "B": [0, 0, 1, 0]}},
            "val": {"name": "bbox_area_sqrt", "bin_width": 1, "max_value": 3, "histograms": {"A": [0, 0, 1, 0], "B": [0, 0, 0, 1], "C": [0, 0, 0, 1]}},
        }

        self.assertEqual(result, expected_result)

    def test_single_data_point(self):
        test_df = pd.DataFrame({"bbox_area_sqrt": [1], "split": ["train"], "class_name": ["A"]})
        result = DetectionBoundingBoxArea._compute_histogram(test_df, name="bbox_area_sqrt")

        expected_result = {"train": {"name": "bbox_area_sqrt", "bin_width": 1, "max_value": 1, "histograms": {"A": [0, 1]}}}

        self.assertEqual(result, expected_result)

    def test_minimum_maximum_values(self):
        test_df = pd.DataFrame({"bbox_area_sqrt": [1, 100], "split": ["val", "val"], "class_name": ["A", "A"]})
        result = DetectionBoundingBoxArea._compute_histogram(test_df, name="bbox_area_sqrt")

        expected_result = {"val": {"name": "bbox_area_sqrt", "bin_width": 1, "max_value": 100, "histograms": {"A": [0] + [1] + [0] * 98 + [1]}}}

        self.assertEqual(result, expected_result)

    def test_histogram_json_output_area_sqrt(self):
        train_sample = DetectionSample(
            sample_id="sample_1",
            split="train",
            image=np.zeros((100, 100, 3)),
            image_channels=ImageChannels.from_str("RGB"),
            bboxes_xyxy=np.array([[2, 2, 4, 4], [3, 3, 6, 6], [1, 1, 5, 5], [1, 1, 4, 4], [5, 5, 6, 6], [7, 7, 9, 9]]),
            class_ids=np.array([0, 1, 2, 2, 3, 4]),
            class_names={0: "A", 1: "B", 2: "C", 3: "D", 4: "E"},
        )

        val_sample = DetectionSample(
            sample_id="sample_2",
            split="val",
            image=np.zeros((100, 100, 3)),
            image_channels=ImageChannels.from_str("RGB"),
            bboxes_xyxy=np.array(
                [
                    [1, 1, 3, 3],
                    [2, 2, 5, 5],
                    [5, 5, 6, 6],
                ]
            ),
            class_ids=np.array([0, 1, 1]),
            class_names={0: "A", 1: "B"},
        )

        extractor = DetectionBoundingBoxArea()
        extractor.update(train_sample)
        extractor.update(val_sample)
        feature = extractor.aggregate()

        histogram_dict_train = feature.json["train"]["histogram_per_class_area"]
        histogram_dict_val = feature.json["val"]["histogram_per_class_area"]

        expected_result_train = {
            "name": "bbox_area_sqrt",
            "bin_width": 1,
            "max_value": 4,
            "histograms": {"A": [0, 0, 1, 0, 0], "B": [0, 0, 0, 1, 0], "C": [0, 0, 0, 1, 1], "D": [0, 1, 0, 0, 0], "E": [0, 0, 1, 0, 0]},
        }

        expected_result_val = {"name": "bbox_area_sqrt", "bin_width": 1, "max_value": 4, "histograms": {"A": [0, 0, 1, 0, 0], "B": [0, 1, 0, 1, 0]}}

        self.assertEqual(histogram_dict_train, expected_result_train)
        self.assertEqual(histogram_dict_val, expected_result_val)

    def test_histogram_json_output_area_perimeter(self):
        train_sample = DetectionSample(
            sample_id="sample_1",
            split="train",
            image=np.zeros((100, 100, 3)),
            image_channels=ImageChannels.from_str("RGB"),
            bboxes_xyxy=np.array([[2, 2, 4, 4], [30, 30, 60, 60], [20, 20, 30, 80], [1, 20, 40, 40], [50, 5, 90, 6], [17, 27, 79, 39]]),
            class_ids=np.array([0, 1, 2, 2, 3, 4]),
            class_names={0: "A", 1: "B", 2: "C", 3: "D", 4: "E"},
        )

        val_sample = DetectionSample(
            sample_id="sample_2",
            split="val",
            image=np.zeros((100, 100, 3)),
            image_channels=ImageChannels.from_str("RGB"),
            bboxes_xyxy=np.array(
                [
                    [1, 1, 3, 3],
                    [20, 25, 50, 50],
                    [15, 50, 60, 80],
                ]
            ),
            class_ids=np.array([0, 1, 1]),
            class_names={0: "A", 1: "B"},
        )

        extractor = DetectionBoundingBoxArea()
        extractor.update(train_sample)
        extractor.update(val_sample)
        feature = extractor.aggregate()

        histogram_dict_train = feature.json["train"]["histogram_per_class_area_perimeter"]
        histogram_dict_val = feature.json["val"]["histogram_per_class_area_perimeter"]

        expected_result_train = {
            "name": "bbox_area_perimeter",
            "bin_width": 1,
            "max_value": 9,
            "histograms": {
                "A": [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                "B": [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                "C": [0, 0, 0, 0, 1, 0, 1, 0, 0, 0],
                "D": [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                "E": [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            },
        }

        expected_result_val = {
            "name": "bbox_area_perimeter",
            "bin_width": 1,
            "max_value": 9,
            "histograms": {"A": [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], "B": [0, 0, 0, 0, 0, 0, 1, 0, 0, 1]},
        }

        self.assertEqual(histogram_dict_train, expected_result_train)
        self.assertEqual(histogram_dict_val, expected_result_val)


if __name__ == "__main__":
    unittest.main()
