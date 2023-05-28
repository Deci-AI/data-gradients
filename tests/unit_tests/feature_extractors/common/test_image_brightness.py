import unittest
import numpy as np

from data_gradients.utils.data_classes.data_samples import ImageSample, ImageChannelFormat
from data_gradients.feature_extractors.commonV2.image_brightness import ImageBrightness
from data_gradients.visualize.seaborn_renderer import SeabornRenderer


class ComponentsSizeDistributionV2Test(unittest.TestCase):
    def setUp(self) -> None:
        self.extractor = ImageBrightness()

        train_image = np.zeros((3, 100, 100), dtype=np.uint8)
        train_image[0, :20, :] = 100
        train_image[1, :80, :] = 200
        train_sample = ImageSample(
            sample_id="sample_1",
            split="train",
            image=train_image,
            image_format=ImageChannelFormat.RGB,
        )
        self.extractor.update(train_sample)

        train_image = np.zeros((3, 100, 100), dtype=np.uint8)
        train_image[0, :50, :] = 50
        train_image[1, :80, :] = 200
        train_sample = ImageSample(
            sample_id="sample_2",
            split="train",
            image=train_image,
            image_format=ImageChannelFormat.RGB,
        )
        self.extractor.update(train_sample)

        train_image = np.zeros((3, 100, 100), dtype=np.uint8)
        train_image[1, :80, :] = 200
        train_sample = ImageSample(
            sample_id="sample_3",
            split="train",
            image=train_image,
            image_format=ImageChannelFormat.RGB,
        )
        self.extractor.update(train_sample)

        valid_image = np.zeros((3, 100, 100), dtype=np.uint8)
        valid_image[0, :20, :] = 150
        valid_image[1, :80, :] = 250
        valid_sample = ImageSample(
            sample_id="sample_4",
            split="valid",
            image=valid_image,
            image_format=ImageChannelFormat.RGB,
        )
        self.extractor.update(valid_sample)

        valid_image = np.zeros((3, 100, 100), dtype=np.uint8)
        valid_image[0, :50, :] = 150
        valid_image[1, :80, :] = 50
        valid_sample = ImageSample(
            sample_id="sample_5",
            split="valid",
            image=valid_image,
            image_format=ImageChannelFormat.RGB,
        )
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
