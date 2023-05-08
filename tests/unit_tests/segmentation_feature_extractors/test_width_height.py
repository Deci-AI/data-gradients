import unittest

from data_gradients.feature_extractors import WidthHeight
from dummy_data import get_dummy_segmentation_batch_data


class TestComponentsSizeDistribution(unittest.TestCase):
    def setUp(self):
        self.num_classes = 3
        self.batch_size = 2
        batch = get_dummy_segmentation_batch_data(batch_size=self.batch_size, num_classes=self.num_classes, height=256, width=256)

        self.extractor = WidthHeight()
        self.extractor.update(batch)

    def test_update(self):
        width, height = self.extractor._width, self.extractor._height

        assert len(width["train"]) == len(height["train"]) == self.batch_size * self.num_classes
        assert width["train"] == [0.1953125, 0.79296875, 0.390625, 0.1953125, 0.76171875, 0.390625]
        assert height["train"] == [0.5859375, 0.48828125, 0.390625, 0.5859375, 0.48828125, 0.390625]

    def test_aggregate(self):
        results = self.extractor._aggregate("train")
        assert list(results.x), [0.1953125, 0.79296875, 0.390625, 0.1953125, 0.76171875, 0.390625]
        assert list(results.y), [0.5859375, 0.48828125, 0.390625, 0.5859375, 0.48828125, 0.390625]


if __name__ == "__main__":
    unittest.main()
