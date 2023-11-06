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

        result = DetectionBoundingBoxArea._compute_histogram(test_df, transform_name="sqrt")

        expected_result = {
            "train": {"transform": "sqrt", "bin_width": 1, "max_value": 3, "histograms": {"A": [0, 1, 0, 2], "B": [0, 0, 1, 0]}},
            "val": {"transform": "sqrt", "bin_width": 1, "max_value": 3, "histograms": {"A": [0, 0, 1, 0], "B": [0, 0, 0, 1], "C": [0, 0, 0, 1]}},
        }

        self.assertEqual(result, expected_result)

    def test_single_data_point(self):
        test_df = pd.DataFrame({"bbox_area_sqrt": [1], "split": ["train"], "class_name": ["A"]})
        result = DetectionBoundingBoxArea._compute_histogram(test_df, transform_name="sqrt")

        expected_result = {"train": {"transform": "sqrt", "bin_width": 1, "max_value": 1, "histograms": {"A": [0, 1]}}}

        self.assertEqual(result, expected_result)

    def test_minimum_maximum_values(self):
        test_df = pd.DataFrame({"bbox_area_sqrt": [1, 100], "split": ["val", "val"], "class_name": ["A", "A"]})
        result = DetectionBoundingBoxArea._compute_histogram(test_df, transform_name="sqrt")

        expected_result = {"val": {"transform": "sqrt", "bin_width": 1, "max_value": 100, "histograms": {"A": [0] + [1] + [0] * 98 + [1]}}}

        self.assertEqual(result, expected_result)

    def test_histogram_json_output(self):
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

        histogram_dict_train = feature.json["train"]["histogram_per_class"]
        histogram_dict_val = feature.json["val"]["histogram_per_class"]

        expected_result_train = {
            "transform": "sqrt",
            "bin_width": 1,
            "max_value": 4,
            "histograms": {"A": [0, 0, 1, 0, 0], "B": [0, 0, 0, 1, 0], "C": [0, 0, 0, 1, 1], "D": [0, 1, 0, 0, 0], "E": [0, 0, 1, 0, 0]},
        }

        expected_result_val = {"transform": "sqrt", "bin_width": 1, "max_value": 4, "histograms": {"A": [0, 0, 1, 0, 0], "B": [0, 1, 0, 1, 0]}}

        self.assertEqual(histogram_dict_train, expected_result_train)
        self.assertEqual(histogram_dict_val, expected_result_val)


if __name__ == "__main__":
    unittest.main()
