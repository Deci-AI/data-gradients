import unittest
import numpy as np

from data_gradients.utils.data_classes.data_samples import ImageSample, ImageChannelFormat
from data_gradients.feature_extractors.common.image_resolution import ImagesResolution
from data_gradients.visualize.seaborn_renderer import SeabornRenderer


class ImageResolutionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.extractor = ImagesResolution()

        train_sample = ImageSample(
            sample_id="sample_0",
            split="train",
            image=np.zeros((50, 100, 3), dtype=np.uint8),
            image_format=ImageChannelFormat.RGB,
        )
        self.extractor.update(train_sample)

        train_sample = ImageSample(
            sample_id="sample_1",
            split="train",
            image=np.zeros((150, 250, 3), dtype=np.uint8),
            image_format=ImageChannelFormat.RGB,
        )
        self.extractor.update(train_sample)

        train_sample = ImageSample(
            sample_id="sample_2",
            split="train",
            image=np.zeros((150, 200, 3), dtype=np.uint8),
            image_format=ImageChannelFormat.RGB,
        )
        self.extractor.update(train_sample)

        train_sample = ImageSample(
            sample_id="sample_3",
            split="train",
            image=np.zeros((150, 300, 3), dtype=np.uint8),
            image_format=ImageChannelFormat.RGB,
        )
        self.extractor.update(train_sample)

        train_sample = ImageSample(
            sample_id="sample_3",
            split="train",
            image=np.zeros((200, 200, 3), dtype=np.uint8),
            image_format=ImageChannelFormat.RGB,
        )
        self.extractor.update(train_sample)

        valid_sample = ImageSample(
            sample_id="sample_4",
            split="valid",
            image=np.zeros((100, 100, 3), dtype=np.uint8),
            image_format=ImageChannelFormat.RGB,
        )
        self.extractor.update(valid_sample)

        valid_sample = ImageSample(
            sample_id="sample_4",
            split="valid",
            image=np.zeros((150, 150, 3), dtype=np.uint8),
            image_format=ImageChannelFormat.RGB,
        )
        self.extractor.update(valid_sample)

        valid_sample = ImageSample(
            sample_id="sample_4",
            split="valid",
            image=np.zeros((200, 250, 3), dtype=np.uint8),
            image_format=ImageChannelFormat.RGB,
        )
        self.extractor.update(valid_sample)

    def test_update_and_aggregate(self):
        # Create a sample SegmentationSample object for testing
        output_json = self.extractor.aggregate().json

        expected_json = {
            "width": {"count": 8.0, "mean": 193.75, "std": 72.88689868556625, "min": 100.0, "25%": 137.5, "50%": 200.0, "75%": 250.0, "max": 300.0},
            "height": {"count": 8.0, "mean": 143.75, "std": 49.55156044825574, "min": 50.0, "25%": 137.5, "50%": 150.0, "75%": 162.5, "max": 200.0},
        }

        for col in ("width", "height"):
            for key in output_json[col].keys_to_reach_object():
                self.assertEqual(round(output_json[col][key], 4), round(expected_json[col][key], 4))

    def test_plot(self):
        feature = self.extractor.aggregate()
        sns = SeabornRenderer()
        f = sns.render(feature.data, feature.plot_options)
        f.show()


if __name__ == "__main__":
    unittest.main()
