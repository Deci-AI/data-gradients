from typing import Mapping, Any, Optional

import cv2
import numpy as np

from data_gradients.feature_extractors.features import ImageFeatures


class ImageFeaturesExtractor:
    """
    Extracts features from a input image .
    """

    def __init__(self):
        """
        """

    def __call__(self, image: np.ndarray, shared_keys: Optional[Mapping[str, Any]] = None) -> Mapping[str, Any]:
        """
        Extracts features from a single image.

        :param image: An input segmentation mask of [H,W,C] or [H,W] shape.
        :param shared_keys: A dictionary of shared keys that will be added to the each row of the output.
                            For instance this may include image id or dataset split property that is shared
                            for every instance.

        :return: A dictionary of features
        """
        if not isinstance(image, np.ndarray):
            raise ValueError("image must be a numpy array. Got: {}".format(type(image)))

        image = image.reshape((image.shape[0], image.shape[1], -1))

        if image.shape[2] == 3:
            image_grayscale = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        elif image.shape[2] > 1:
            image_grayscale = image.mean(axis=2)
        else:
            image_grayscale = image

        if image.shape[2] <= 4:
            mean, std = cv2.meanStdDev(image)
            mean = mean[:image.shape[2]].reshape(-1)
            std = std[:image.shape[2]].reshape(-1)
        else:
            mean = image.mean(axis=(0, 1), keepdims=False)
            std = image.std(axis=(0, 1), keepdims=False)

        features = {
            ImageFeatures.ImageWidth: [image.shape[1]],
            ImageFeatures.ImageHeight: [image.shape[0]],
            ImageFeatures.ImageArea: [image.shape[0] * image.shape[1]],
            ImageFeatures.ImageAspectRatio: [image.shape[1] / image.shape[0]],
            ImageFeatures.ImageMean: [mean],
            ImageFeatures.ImageStd: [std],
            ImageFeatures.ImageNumChannels: [image.shape[2]],
            ImageFeatures.ImageMinBrightness: [image_grayscale.min()],
            ImageFeatures.ImageAvgBrightness: [image_grayscale.mean()],
            ImageFeatures.ImageMaxBrightness: [image_grayscale.max()],
        }

        if shared_keys is not None:
            for key, value in shared_keys.items():
                features[key] = [value]

        return features
