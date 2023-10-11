import unittest

import numpy as np

from data_gradients.feature_extractors import ImagesAverageBrightness
from data_gradients.utils.data_classes import ImageSample


class AverageBrightnessTest(unittest.TestCase):
    """ """

    def setUp(self):
        self.split = "train"

    def test_black_image(self):
        sample = ImageSample(image=np.zeros((100, 100, 3), dtype=np.uint8), split=self.split, sample_id="Random", image_channels=str.RGB)
        feature_extractor = ImagesAverageBrightness()
        feature_extractor.update(sample)
        feature_extractor.aggregate()

    def test_white_image(self):
        sample = ImageSample(image=np.ones((100, 100, 3), dtype=np.uint8), split=self.split, sample_id="Random", image_channels=str.RGB)
        feature_extractor = ImagesAverageBrightness()
        feature_extractor.update(sample)
        feature_extractor.aggregate()

    def test_noise_image(self):
        image = np.ones((100, 100, 1), dtype=np.uint8)
        image[:50, :100] = 0
        sample = ImageSample(image=image, split=self.split, sample_id="Random", image_channels=str.GRAYSCALE)

        target_value = np.mean(image)
        self.assertAlmostEqual(target_value, 0.5)
        feature_extractor = ImagesAverageBrightness()

        # Move it to be outside its bucket (bucket for 1 value will have this value inside it
        target_value += 0.02
        feature_extractor.update(sample)
        feature_extractor.aggregate()


if __name__ == "__main__":
    unittest.main()
