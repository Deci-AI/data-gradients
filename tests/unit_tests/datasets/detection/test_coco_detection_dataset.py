import unittest
from torch.utils.data import DataLoader
from data_gradients.managers.detection_manager import DetectionAnalysisManager

from data_gradients.datasets.detection.coco_detection_dataset import CocoDetectionDataset


class CocoSegmentationDatasetTest(unittest.TestCase):
    def setUp(self):
        self.train_set = CocoDetectionDataset(root_dir="../../../../example_dataset/tinycoco", split="train", year="2017")
        self.val_set = CocoDetectionDataset(root_dir="../../../../example_dataset/tinycoco", split="val", year="2017")

    def test_coco_dataset(self):
        da = DetectionAnalysisManager(
            report_title="Testing Data-Gradients NEW",
            train_data=self.train_set,
            val_data=self.val_set,
            class_names=self.train_set.class_names,
            batches_early_stop=10,
            use_cache=False,
        )

        da.run()

    def test_coco_dataset_batch(self):

        da = DetectionAnalysisManager(
            report_title="Testing Data-Gradients NEW",
            train_data=DataLoader(self.train_set, batch_size=1),
            val_data=DataLoader(self.val_set, batch_size=1),
            class_names=self.train_set.class_names,
            batches_early_stop=10,
            use_cache=False,
        )

        da.run()


if __name__ == "__main__":
    unittest.main()
