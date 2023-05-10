import unittest

import cv2
import numpy as np
import torch

from data_gradients.feature_extractors import AverageBrightness
from data_gradients.utils import BatchData


class AverageBrightnessTest(unittest.TestCase):
    """ """

    def setUp(self):
        self.feature_extractor = AverageBrightness()
        self.split = "train"

    def test_black_image(self):
        batch = BatchData(images=torch.zeros((1, 3, 100, 100)), labels=[], split=self.split)
        target_value = 0
        self.feature_extractor.update(batch)
        results = self.feature_extractor._aggregate(self.split)
        self._check_value_in_right_bin(results.bin_values, results.bin_names, target_value)

    def test_white_image(self):
        target_value = 1
        batch = BatchData(images=torch.ones((1, 3, 100, 100)), labels=[], split=self.split)
        self.feature_extractor.update(batch)
        results = self.feature_extractor._aggregate(self.split)
        self._check_value_in_right_bin(results.bin_values, results.bin_names, target_value)

    def test_noise_image(self):
        images = torch.ones((1, 3, 100, 100))
        images[0][:, :50, :100] = 0
        batch = BatchData(images=images, labels=[], split=self.split)

        image = batch.images[0]
        np_image = image.numpy().transpose(1, 2, 0)
        lightness, _, _ = cv2.split(cv2.cvtColor(np_image, cv2.COLOR_BGR2LAB))
        n_lightness = lightness / np.max(lightness)
        target_value = np.mean(n_lightness)

        if target_value != 0.5:
            self.fail(f"Target value of half ones and half zeros image should be 0.5! Got {target_value}")

        # Move it to be outside its bucket (bucket for 1 value will have this value inside it
        target_value += 0.02
        self.feature_extractor.update(batch)
        results = self.feature_extractor._aggregate(self.split)
        self._check_value_in_right_bin(results.bin_values, results.bin_names, target_value)

    def _check_value_in_right_bin(self, values, bins, target_value):
        bin_index = -1
        for bin_ in bins:
            a_bin = float(bin_.split("<")[0])
            b_bin = float(bin_.split("<")[1])
            if a_bin <= target_value < b_bin:
                bin_index = bins.index(bin_)
                break
        if bin_index < 0:
            self.fail("Did not find a bin containing the right value")
        value_index = np.argmax(values)
        if value_index != bin_index:
            # self.fail(f'Value is not in the right bin! Got value index {value_index} ({values})'
            #           f' and bins index {bin_index} ({bins}) with target {target_value}')
            self.fail(f"{bin_index}, {bins[bin_index]}, {target_value}")


if __name__ == "__main__":
    unittest.main()
