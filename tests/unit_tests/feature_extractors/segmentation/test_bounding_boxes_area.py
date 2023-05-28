import unittest
import numpy as np

from data_gradients.utils.data_classes.data_samples import SegmentationSample, ImageChannelFormat
from data_gradients.utils.data_classes.contour import Contour
from data_gradients.feature_extractors.segmentationV2.bounding_boxes_area import BoundingBoxAreaFeatureExtractor
from data_gradients.visualize.seaborn_renderer import SeabornRenderer


class ComponentsSizeDistributionV2Test(unittest.TestCase):
    def setUp(self) -> None:
        train_sample = SegmentationSample(
            sample_id="sample_1",
            split="train",
            image=np.zeros((100, 100, 3)),
            image_format=ImageChannelFormat.RGB,
            mask=np.zeros((3, 100, 100)),
            contours=[
                [
                    Contour(
                        points=[(10, 10), (20, 20), (30, 30)],
                        area=50,
                        w=10,
                        h=10,
                        center=(15, 15),
                        perimeter=40,
                        class_id=0,
                        bbox_area=50,
                    ),
                    Contour(
                        points=[(20, 20), (30, 30), (40, 40)],
                        area=80,
                        w=20,
                        h=20,
                        center=(30, 30),
                        perimeter=60,
                        class_id=0,
                        bbox_area=80,
                    ),
                ],
                [
                    Contour(
                        points=[(50, 50), (60, 60), (70, 70)],
                        area=30,
                        w=10,
                        h=10,
                        center=(60, 60),
                        perimeter=30,
                        class_id=1,
                        bbox_area=30,
                    ),
                    Contour(
                        points=[(70, 70), (80, 80), (90, 90)],
                        area=60,
                        w=20,
                        h=20,
                        center=(80, 80),
                        perimeter=50,
                        class_id=1,
                        bbox_area=60,
                    ),
                    Contour(
                        points=[(90, 90), (95, 95), (100, 100)],
                        area=20,
                        w=10,
                        h=10,
                        center=(95, 95),
                        perimeter=40,
                        class_id=1,
                        bbox_area=20,
                    ),
                ],
                [
                    Contour(
                        points=[(30, 30), (40, 40), (50, 50)],
                        area=40,
                        w=10,
                        h=10,
                        center=(40, 40),
                        perimeter=30,
                        class_id=2,
                        bbox_area=40,
                    ),
                    Contour(
                        points=[(60, 60), (70, 70), (80, 80)],
                        area=70,
                        w=20,
                        h=20,
                        center=(70, 70),
                        perimeter=50,
                        class_id=2,
                        bbox_area=70,
                    ),
                ],
            ],
        )

        valid_sample = SegmentationSample(
            sample_id="sample_2",
            split="valid",
            image=np.zeros((100, 100, 3)),
            image_format=ImageChannelFormat.RGB,
            mask=np.zeros((3, 100, 100)),
            contours=[
                [
                    Contour(
                        points=[(10, 10), (20, 20), (30, 30)],
                        area=50,
                        w=10,
                        h=10,
                        center=(15, 15),
                        perimeter=40,
                        class_id=0,
                        bbox_area=50,
                    ),
                    Contour(
                        points=[(20, 20), (30, 30), (40, 40)],
                        area=80,
                        w=20,
                        h=20,
                        center=(30, 30),
                        perimeter=60,
                        class_id=0,
                        bbox_area=80,
                    ),
                ],
                [
                    Contour(
                        points=[(50, 50), (60, 60), (70, 70)],
                        area=30,
                        w=10,
                        h=10,
                        center=(60, 60),
                        perimeter=30,
                        class_id=1,
                        bbox_area=30,
                    ),
                ],
            ],
        )
        self.extractor = BoundingBoxAreaFeatureExtractor()
        self.extractor.update(train_sample)
        self.extractor.update(valid_sample)

    def test_update_and_aggregate(self):
        # Create a sample SegmentationSample object for testing
        feature = self.extractor.aggregate()

        expected_data = [
            {"split": "train", "class_name": "0", "bbox_area": 0.5},
            {"split": "train", "class_name": "0", "bbox_area": 0.8},
            {"split": "train", "class_name": "1", "bbox_area": 0.3},
            {"split": "train", "class_name": "1", "bbox_area": 0.6},
            {"split": "train", "class_name": "1", "bbox_area": 0.2},
            {"split": "train", "class_name": "2", "bbox_area": 0.4},
            {"split": "train", "class_name": "2", "bbox_area": 0.7},
            {"split": "valid", "class_name": "0", "bbox_area": 0.5},
            {"split": "valid", "class_name": "0", "bbox_area": 0.8},
            {"split": "valid", "class_name": "1", "bbox_area": 0.3},
        ]
        self.assertEqual(feature.data.to_dict(orient="records"), expected_data)

    def test_plot(self):
        feature = self.extractor.aggregate()
        sns = SeabornRenderer()
        f = sns.render(feature.data, feature.plot_options)
        f.show()


if __name__ == "__main__":
    unittest.main()
