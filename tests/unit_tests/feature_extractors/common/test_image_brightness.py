import unittest
import numpy as np

from data_gradients.utils.data_classes.data_samples import ImageSample, Image
from data_gradients.feature_extractors.common.image_average_brightness import ImagesAverageBrightness
from data_gradients.feature_extractors.common.image_color_distribution import ImageColorDistribution
from data_gradients.visualize.seaborn_renderer import SeabornRenderer
from data_gradients.utils.data_classes.image_channels import ImageChannels
from data_gradients.dataset_adapters.formatters.utils import Uint8ImageFormat


class ImageBrightnessTest(unittest.TestCase):
    def setUp(self) -> None:
        self.average_brightness = ImagesAverageBrightness()
        self.color_distribution = ImageColorDistribution()

        train_image = np.zeros((100, 100, 3), dtype=np.uint8)
        train_sample = ImageSample(
            sample_id="sample_1",
            split="train",
            image=Image(data=train_image, format=Uint8ImageFormat(), channels=ImageChannels.from_str("RGB")),
        )
        self.average_brightness.update(train_sample)
        self.color_distribution.update(train_sample)

        train_image = np.zeros((100, 100, 3), dtype=np.uint8)
        train_image[0, :50, :] = 50
        train_image[1, :80, :] = 200
        train_sample = ImageSample(
            sample_id="sample_2",
            split="train",
            image=Image(data=train_image, format=Uint8ImageFormat(), channels=ImageChannels.from_str("RGB")),
        )
        self.average_brightness.update(train_sample)
        self.color_distribution.update(train_sample)

        train_image = np.zeros((100, 100, 3), dtype=np.uint8)
        train_image[1, :80, :] = 200
        train_sample = ImageSample(
            sample_id="sample_3",
            split="train",
            image=Image(data=train_image, format=Uint8ImageFormat(), channels=ImageChannels.from_str("RGB")),
        )
        self.average_brightness.update(train_sample)
        self.color_distribution.update(train_sample)

        train_image = np.zeros((100, 100, 3), dtype=np.uint8)
        train_image[1, :80, :] = 200
        train_sample = ImageSample(
            sample_id="sample_3",
            split="train",
            image=Image(data=train_image, format=Uint8ImageFormat(), channels=ImageChannels.from_str("RGB")),
        )
        self.average_brightness.update(train_sample)
        self.color_distribution.update(train_sample)

        valid_image = np.zeros((100, 100, 3), dtype=np.uint8)
        valid_image[0, :20, :] = 150
        valid_image[1, :80, :] = 250
        valid_sample = ImageSample(
            sample_id="sample_4",
            split="val",
            image=Image(data=valid_image, format=Uint8ImageFormat(), channels=ImageChannels.from_str("RGB")),
        )
        self.average_brightness.update(valid_sample)
        self.color_distribution.update(valid_sample)

        valid_image = np.zeros((100, 100, 3), dtype=np.uint8)
        valid_image[0, :20, :] = 150
        valid_image[1, :80, :] = 250
        valid_sample = ImageSample(
            sample_id="sample_4",
            split="val",
            image=Image(data=valid_image, format=Uint8ImageFormat(), channels=ImageChannels.from_str("RGB")),
        )
        self.average_brightness.update(valid_sample)
        self.color_distribution.update(valid_sample)

        valid_image = np.zeros((100, 100, 3), dtype=np.uint8)
        valid_image[0, :50, :] = 150
        valid_image[1, :80, :] = 50
        valid_sample = ImageSample(
            sample_id="sample_5",
            split="val",
            image=Image(data=valid_image, format=Uint8ImageFormat(), channels=ImageChannels.from_str("RGB")),
        )
        self.average_brightness.update(valid_sample)
        self.color_distribution.update(valid_sample)

    def test_update_and_aggregate(self):
        # Create a sample SegmentationSample object for testing
        output_json = self.average_brightness.aggregate().json

        expected_json = {
            "train": {
                "count": 4.0,
                "mean": 87.54166666666666,
                "std": 4.416666666666671,
                "min": 85.33333333333333,
                "25%": 85.33333333333333,
                "50%": 85.33333333333333,
                "75%": 87.54166666666666,
                "max": 94.16666666666667,
            },
            "val": {
                "count": 3.0,
                "mean": 101.13333333333333,
                "std": 9.122134253196094,
                "min": 95.86666666666666,
                "25%": 95.86666666666666,
                "50%": 95.86666666666666,
                "75%": 103.76666666666667,
                "max": 111.66666666666667,
            },
        }

        for split in output_json.keys():
            for key in output_json[split].keys():
                self.assertEqual(round(output_json[split][key], 4), round(expected_json[split][key], 4))

    def test_plot(self):
        feature = self.average_brightness.aggregate()
        sns = SeabornRenderer()
        f = sns.render(feature.data, feature.plot_options)
        f.show()

    def test_color_distribution_plot(self):
        feature = self.color_distribution.aggregate()
        sns = SeabornRenderer()
        f = sns.render(feature.data, feature.plot_options)
        f.show()


if __name__ == "__main__":
    unittest.main()
