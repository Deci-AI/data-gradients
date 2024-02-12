import unittest

import numpy as np

from data_gradients.feature_extractors import ImagesAverageBrightness
from data_gradients.utils.data_classes.data_samples import ImageSample, Image
from data_gradients.utils.data_classes.image_channels import ImageChannels
from data_gradients.dataset_adapters.formatters.utils import FloatImageFormat


class AverageBrightnessTest(unittest.TestCase):
    """ """

    def setUp(self):
        self.split = "train"

    def test_black_image(self):
        sample = ImageSample(
            image=Image(data=np.zeros((100, 100, 3), dtype=np.uint8), format=FloatImageFormat(), channels=ImageChannels.from_str("RGB")),
            split=self.split,
            sample_id="Random",
        )
        feature_extractor = ImagesAverageBrightness()
        feature_extractor.update(sample)
        feature_extractor.aggregate()

    def test_white_image(self):
        sample = ImageSample(
            image=Image(data=np.ones((100, 100, 3), dtype=np.uint8), format=FloatImageFormat(), channels=ImageChannels.from_str("RGB")),
            split=self.split,
            sample_id="Random",
        )
        feature_extractor = ImagesAverageBrightness()
        feature_extractor.update(sample)
        feature_extractor.aggregate()

    def test_noise_image(self):
        image = np.ones((100, 100, 1), dtype=np.uint8)
        image[:50, :100] = 0

        sample = ImageSample(
            image=Image(data=image, format=FloatImageFormat(), channels=ImageChannels.from_str("G")),
            split=self.split,
            sample_id="Random",
        )

        target_value = np.mean(image)
        self.assertAlmostEqual(target_value, 0.5)
        feature_extractor = ImagesAverageBrightness()

        # Move it to be outside its bucket (bucket for 1 value will have this value inside it
        target_value += 0.02
        feature_extractor.update(sample)
        feature_extractor.aggregate()


if __name__ == "__main__":
    unittest.main()
