import unittest

from data_gradients.managers.segmentation_manager import SegmentationAnalysisManager
from data_gradients.datasets.segmentation.coco_segmentation_dataset import CocoSegmentationDataset


class CocoSegmentationDatasetTest(unittest.TestCase):
    def setUp(self):
        self.train_set = CocoSegmentationDataset(root_dir="../../../../example_dataset/tinycoco", split="train", year="2017")
        self.val_set = CocoSegmentationDataset(root_dir="../../../../example_dataset/tinycoco", split="val", year="2017")

    def test_coco_dataset(self):
        da = SegmentationAnalysisManager(
            report_title="Testing Data-Gradients NEW",
            train_data=self.train_set,
            val_data=self.val_set,
            class_names=self.train_set.class_names,
            batches_early_stop=10,
            use_cache=False,
        )

        da.run()


if __name__ == "__main__":
    unittest.main()
