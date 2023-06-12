import unittest

import cv2
import numpy as np

from data_gradients.utils.image_processing import resize_in_chunks


class TestAssets(unittest.TestCase):
    def test_below_512(self):
        image = np.random.random((300, 300, 200))

        resized_in_chunks = resize_in_chunks(img=image, size=(100, 100), interpolation=cv2.INTER_LINEAR)
        resized = cv2.resize(src=image, dsize=(100, 100), interpolation=cv2.INTER_LINEAR)
        self.assertTrue(np.array_equal(resized_in_chunks, resized))

    def test_above_512(self):
        image = np.random.random((300, 300, 600))

        resized_in_chunks = resize_in_chunks(img=image, size=(100, 100), interpolation=cv2.INTER_LINEAR)
        with self.assertRaises(Exception):
            _ = cv2.resize(src=image, dsize=(100, 100), interpolation=cv2.INTER_LINEAR)

        self.assertEqual(tuple(resized_in_chunks.shape), (100, 100, 600))


if __name__ == "__main__":
    unittest.main()
