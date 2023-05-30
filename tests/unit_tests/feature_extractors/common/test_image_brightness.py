import unittest
import numpy as np

from data_gradients.utils.data_classes.data_samples import ImageSample, ImageChannelFormat
from data_gradients.feature_extractors.commonV2.image_average_brightness import ImagesAverageBrightness
from data_gradients.feature_extractors.commonV2.image_color_distribution import ImageColorDistribution
from data_gradients.visualize.seaborn_renderer import SeabornRenderer


class ImageBrightnessTest(unittest.TestCase):
    def setUp(self) -> None:
        self.average_brightness = ImagesAverageBrightness()
        self.color_distribution = ImageColorDistribution()

        train_image = np.zeros((100, 100, 3), dtype=np.uint8)
        train_sample = ImageSample(
            sample_id="sample_1",
            split="train",
            image=train_image,
            image_format=ImageChannelFormat.RGB,
        )
        self.average_brightness.update(train_sample)
        self.color_distribution.update(train_sample)

        train_image = np.zeros((100, 100, 3), dtype=np.uint8)
        train_image[0, :50, :] = 50
        train_image[1, :80, :] = 200
        train_sample = ImageSample(
            sample_id="sample_2",
            split="train",
            image=train_image,
            image_format=ImageChannelFormat.RGB,
        )
        self.average_brightness.update(train_sample)
        self.color_distribution.update(train_sample)

        train_image = np.zeros((100, 100, 3), dtype=np.uint8)
        train_image[1, :80, :] = 200
        train_sample = ImageSample(
            sample_id="sample_3",
            split="train",
            image=train_image,
            image_format=ImageChannelFormat.RGB,
        )
        self.average_brightness.update(train_sample)
        self.color_distribution.update(train_sample)

        train_image = np.zeros((100, 100, 3), dtype=np.uint8)
        train_image[1, :80, :] = 200
        train_sample = ImageSample(
            sample_id="sample_3",
            split="train",
            image=train_image,
            image_format=ImageChannelFormat.RGB,
        )
        self.average_brightness.update(train_sample)
        self.color_distribution.update(train_sample)

        valid_image = np.zeros((100, 100, 3), dtype=np.uint8)
        valid_image[0, :20, :] = 150
        valid_image[1, :80, :] = 250
        valid_sample = ImageSample(
            sample_id="sample_4",
            split="valid",
            image=valid_image,
            image_format=ImageChannelFormat.RGB,
        )
        self.average_brightness.update(valid_sample)
        self.color_distribution.update(valid_sample)

        valid_image = np.zeros((100, 100, 3), dtype=np.uint8)
        valid_image[0, :20, :] = 150
        valid_image[1, :80, :] = 250
        valid_sample = ImageSample(
            sample_id="sample_4",
            split="valid",
            image=valid_image,
            image_format=ImageChannelFormat.RGB,
        )
        self.average_brightness.update(valid_sample)
        self.color_distribution.update(valid_sample)

        valid_image = np.zeros((100, 100, 3), dtype=np.uint8)
        valid_image[0, :50, :] = 150
        valid_image[1, :80, :] = 50
        valid_sample = ImageSample(
            sample_id="sample_5",
            split="valid",
            image=valid_image,
            image_format=ImageChannelFormat.RGB,
        )
        self.average_brightness.update(valid_sample)
        self.color_distribution.update(valid_sample)

    def test_update_and_aggregate(self):
        # Create a sample SegmentationSample object for testing
        output_json = self.average_brightness.aggregate().json

        expected_json = {
            "count": 7.0,
            "mean": 93.3667,
            "std": 9.501169,
            "min": 85.333333,
            "25%": 85.333333,
            "50%": 94.166667,
            "75%": 95.866667,
            "max": 111.66667,
        }

        for key in output_json.keys():
            self.assertEqual(round(output_json[key], 4), round(expected_json[key], 4))

    def test_plot(self):
        feature = self.average_brightness.aggregate()
        sns = SeabornRenderer()
        figs = sns.render(feature.data, feature.plot_options)
        for f in figs:
            f.show()

    def test_color_distribution_plot(self):
        feature = self.color_distribution.aggregate()
        sns = SeabornRenderer()
        figs = sns.render(feature.data, feature.plot_options)
        for f in figs:
            f.show()


if __name__ == "__main__":
    unittest.main()
