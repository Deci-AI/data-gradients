import unittest

import pandas as pd

from data_gradients.feature_extractors.object_detection.bounding_boxes_area import DetectionBoundingBoxArea


class TestComputeHistogram(unittest.TestCase):
    def test_compute_histogram(self):
        test_df = pd.DataFrame({
            'bbox_area_sqrt': [1, 2, 3, 3, 3, 2, 3],
            'split': ['train', 'train', 'train', 'train', 'val', 'val', 'val'],
            'class_name': ['A', 'B', 'A', 'A', 'B', 'A', 'C']
        })

        result = DetectionBoundingBoxArea._compute_histogram(test_df, transform_name='sqrt', min_bin_val=1)

        expected_result = {
            'train': {
                'transform': 'sqrt',
                'bin_width': 1,
                'min_value': 1,
                'max_value': 4,
                'histograms': {
                    'A': [1, 0, 2],
                    'B': [0, 1, 0]
                }
            },
            'val': {
                'transform': 'sqrt',
                'bin_width': 1,
                'min_value': 1,
                'max_value': 4,
                'histograms': {
                    'A': [0, 1, 0],
                    'B': [0, 0, 1],
                    'C': [0, 0, 1]
                }
            }
        }

        self.assertEqual(result, expected_result)

    def test_single_data_point(self):
        test_df = pd.DataFrame({'bbox_area_sqrt': [1], 'split': ['train'], 'class_name': ['A']})
        result = DetectionBoundingBoxArea._compute_histogram(test_df, transform_name='sqrt', min_bin_val=1)

        expected_result = {
            'train': {
                'transform': 'sqrt',
                'bin_width': 1,
                'min_value': 1,
                'max_value': 2,
                'histograms': {
                    'A': [1]
                }
            }
        }

        self.assertEqual(result, expected_result)

    def test_minimum_maximum_values(self):
        test_df = pd.DataFrame({
            'bbox_area_sqrt': [1, 100],
            'split': ['val', 'val'],
            'class_name': ['A', 'A']
        })
        result = DetectionBoundingBoxArea._compute_histogram(test_df, transform_name='sqrt', min_bin_val=1)

        expected_result = {
            'val': {
                'transform': 'sqrt',
                'bin_width': 1,
                'min_value': 1,
                'max_value': 101,
                'histograms': {
                    'A': [1] + [0] * 98 + [1]
                }
            }
        }

        self.assertEqual(result, expected_result)

    def test_min_bin_val(self):
        test_df = pd.DataFrame({
            'bbox_area_sqrt': [3, 3, 3],
            'split': ['val', 'val', 'val'],
            'class_name': ['A', 'A', 'A']
        })
        result = DetectionBoundingBoxArea._compute_histogram(test_df, transform_name='sqrt', min_bin_val=2)

        expected_result = {
            'val': {
                'transform': 'sqrt',
                'bin_width': 1,
                'min_value': 2,
                'max_value': 4,
                'histograms': {
                    'A': [0, 3]
                }
            }
        }

        self.assertEqual(result, expected_result)


if __name__ == '__main__':
    unittest.main()
