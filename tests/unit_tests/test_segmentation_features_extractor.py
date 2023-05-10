import unittest

import cv2
import numpy as np
import pandas as pd

from data_gradients.feature_extractors.segmentation.segmentation_features_extractor import SemanticSegmentationFeaturesExtractor


class SemanticSegmentationFeaturesExtractorTest(unittest.TestCase):
    def test_simple_image(self):
        image = np.zeros((10, 10), dtype=np.uint8)
        extractor = SemanticSegmentationFeaturesExtractor(ignore_labels=0)
        dict_df = extractor(image)
        df = pd.DataFrame.from_dict(dict_df)
        self.assertEquals(len(df), 0)

    def test_image_with_simple_shapes(self):
        image = np.zeros((1024, 1024), dtype=np.uint8)
        image[700:, 700:] = 255
        cv2.circle(image, (100, 100), 50, 1, -1)
        cv2.rectangle(image, (500, 500), (600, 600), 2, -1)

        extractor = SemanticSegmentationFeaturesExtractor(ignore_labels=[0, 255])
        dict_df = extractor(image)
        df = pd.DataFrame.from_dict(dict_df)
        self.assertEquals(len(df), 2)
        print(df)

    def test_image_with_simple_shapes_no_ignore(self):
        image = np.zeros((1024, 1024), dtype=np.uint8)
        image[700:, 700:] = 255
        cv2.circle(image, (100, 100), 50, 1, -1)
        cv2.rectangle(image, (500, 500), (600, 600), 2, -1)

        extractor = SemanticSegmentationFeaturesExtractor(ignore_labels=None)
        dict_df = extractor(image)
        df = pd.DataFrame.from_dict(dict_df)
        self.assertEquals(len(df), 4)
        print(df)
