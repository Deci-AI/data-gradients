import unittest
import numpy as np

from data_gradients.utils.data_classes.data_samples import SegmentationSample, ImageChannelFormat
from data_gradients.utils.data_classes.contour import Contour
from data_gradients.feature_extractors.segmentationV2.bounding_boxes_area import BoundingBoxAreaFeatureExtractor
from data_gradients.feature_extractors.segmentationV2.bounding_boxes_resolution import BoundingBoxResolution
from data_gradients.feature_extractors.segmentationV2.classes_distribution import ClassesDistribution
from data_gradients.visualize.seaborn_renderer import SeabornRenderer


class SegmentationBBoxV2Test(unittest.TestCase):
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
                        area=500,
                        w=10,
                        h=10,
                        center=(15, 15),
                        perimeter=40,
                        class_id=0,
                        bbox_area=500,
                    ),
                    Contour(
                        points=[(20, 20), (30, 30), (40, 40)],
                        area=800,
                        w=20,
                        h=20,
                        center=(30, 30),
                        perimeter=60,
                        class_id=0,
                        bbox_area=800,
                    ),
                ],
                [
                    Contour(
                        points=[(50, 50), (60, 60), (70, 70)],
                        area=300,
                        w=10,
                        h=10,
                        center=(60, 60),
                        perimeter=30,
                        class_id=1,
                        bbox_area=300,
                    ),
                    Contour(
                        points=[(70, 70), (80, 80), (90, 90)],
                        area=600,
                        w=20,
                        h=20,
                        center=(80, 80),
                        perimeter=50,
                        class_id=1,
                        bbox_area=600,
                    ),
                    Contour(
                        points=[(90, 90), (95, 95), (100, 100)],
                        area=200,
                        w=10,
                        h=10,
                        center=(95, 95),
                        perimeter=40,
                        class_id=1,
                        bbox_area=200,
                    ),
                ],
                [
                    Contour(
                        points=[(30, 30), (40, 40), (50, 50)],
                        area=400,
                        w=10,
                        h=10,
                        center=(40, 40),
                        perimeter=30,
                        class_id=2,
                        bbox_area=400,
                    ),
                    Contour(
                        points=[(60, 60), (70, 70), (80, 80)],
                        area=700,
                        w=20,
                        h=20,
                        center=(70, 70),
                        perimeter=50,
                        class_id=2,
                        bbox_area=700,
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
                        area=500,
                        w=10,
                        h=10,
                        center=(15, 15),
                        perimeter=40,
                        class_id=0,
                        bbox_area=500,
                    ),
                    Contour(
                        points=[(20, 20), (30, 30), (40, 40)],
                        area=800,
                        w=20,
                        h=20,
                        center=(30, 30),
                        perimeter=60,
                        class_id=0,
                        bbox_area=800,
                    ),
                ],
                [
                    Contour(
                        points=[(50, 50), (60, 60), (70, 70)],
                        area=300,
                        w=10,
                        h=10,
                        center=(60, 60),
                        perimeter=30,
                        class_id=1,
                        bbox_area=300,
                    ),
                ],
            ],
        )

        self.resolution_extractor = BoundingBoxResolution()
        self.resolution_extractor.update(train_sample)
        self.resolution_extractor.update(valid_sample)

        self.class_distribution_extractor = ClassesDistribution()
        self.class_distribution_extractor.update(train_sample)
        self.class_distribution_extractor.update(valid_sample)

        self.area_extractor = BoundingBoxAreaFeatureExtractor()
        self.area_extractor.update(train_sample)
        self.area_extractor.update(valid_sample)

    def test_update_and_aggregate(self):
        # Create a sample SegmentationSample object for testing
        feature = self.area_extractor.aggregate()

        expected_data = [
            {"split": "train", "class_name": "0", "bbox_area": 5},
            {"split": "train", "class_name": "0", "bbox_area": 8},
            {"split": "train", "class_name": "1", "bbox_area": 3},
            {"split": "train", "class_name": "1", "bbox_area": 6},
            {"split": "train", "class_name": "1", "bbox_area": 2},
            {"split": "train", "class_name": "2", "bbox_area": 4},
            {"split": "train", "class_name": "2", "bbox_area": 7},
            {"split": "valid", "class_name": "0", "bbox_area": 5},
            {"split": "valid", "class_name": "0", "bbox_area": 8},
            {"split": "valid", "class_name": "1", "bbox_area": 3},
        ]
        for x, y in zip(feature.data.to_dict(orient="records"), expected_data):
            self.assertAlmostEqual(x["bbox_area"], y["bbox_area"], delta=0.01)

    def test_plot(self):
        feature = self.area_extractor.aggregate()
        sns = SeabornRenderer()
        f = sns.render(feature.data, feature.plot_options)
        f.show()

    def test_resolution_plot(self):
        feature = self.resolution_extractor.aggregate()
        sns = SeabornRenderer()
        f = sns.render(feature.data, feature.plot_options)
        f.show()

    def test_class_distribution_plot(self):
        feature = self.class_distribution_extractor.aggregate()
        sns = SeabornRenderer()
        f = sns.render(feature.data, feature.plot_options)
        f.show()


if __name__ == "__main__":
    unittest.main()
