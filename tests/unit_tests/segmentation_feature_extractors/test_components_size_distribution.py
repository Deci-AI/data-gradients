import unittest

from data_gradients.feature_extractors import ComponentsSizeDistribution
from dummy_data import get_dummy_segmentation_batch_data


class TestComponentsSizeDistribution(unittest.TestCase):
    def setUp(self):
        self.num_classes = 3
        self.batch_size = 2
        batch = get_dummy_segmentation_batch_data(batch_size=self.batch_size, num_classes=self.num_classes, height=256, width=256)

        self.extractor = ComponentsSizeDistribution(num_classes=self.num_classes, ignore_labels=[])
        self.extractor.update(batch)

    def test_update(self):
        assert all(len(per_class_data) == self.batch_size for per_class_data in self.extractor._hist["train"].values())
        assert self.extractor._hist["train"][0] == [11.444091796875, 11.444091796875]
        assert self.extractor._hist["train"][1] == [38.71917724609375, 37.19329833984375]
        assert self.extractor._hist["train"][2] == [14.95361328125, 14.95361328125]

    def test_aggregate(self):
        result = self.extractor._aggregate("train")
        assert list(result.bin_values), [11.444, 37.498, 14.954]
        assert list(result.bin_names), list(range(self.num_classes))


if __name__ == "__main__":
    unittest.main()
