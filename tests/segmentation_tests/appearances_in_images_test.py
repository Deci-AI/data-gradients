import unittest

import torch

from src.feature_extractors.segmentation.appearances_in_images import AppearancesInImages
from src.preprocess import SegmentationPreprocessor
from tests.segmentation_tests.example_dataset import train_loader, num_classes, ignore_labels


class AppearancesInImagesTest(unittest.TestCase):
    def setUp(self) -> None:
        self._feature_extractor = AppearancesInImages(num_classes=num_classes,
                                                      ignore_labels=ignore_labels)
        self._preprocessor = SegmentationPreprocessor(num_classes=num_classes,
                                                      ignore_labels=ignore_labels)

        self._train_dataloader = train_loader

    def test(self):
        images, labels = next(iter(self._train_dataloader))

        segmentation_batch_data = self._preprocessor.preprocess(images, labels)

        self._feature_extractor.execute(segmentation_batch_data)

        all_unique_labels = torch.Tensor()
        for label in labels:
            unique = torch.unique(label)
            all_unique_labels = torch.concat((all_unique_labels, unique[unique > 0]), dim=0)

        hist = [torch.count_nonzero(all_unique_labels == i).item() for i in range(1, num_classes -1)]

        self.assertEqual(hist, list(self._feature_extractor._hist.values()))


if __name__ == '__main__':
    unittest.main()
