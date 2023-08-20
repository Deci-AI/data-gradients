import unittest
from torch.utils.data import DataLoader
from data_gradients.managers.segmentation_manager import SegmentationAnalysisManager
from data_gradients.datasets.segmentation.voc_segmentation_dataset import VOCSegmentationDataset


class VOCSegmentationDatasetTest(unittest.TestCase):
    def setUp(self):
        root_dir = "..."  # TO FILL
        self.train_set_2007 = VOCSegmentationDataset(
            root_dir=root_dir,
            year="2007",
            split="train",
            verbose=False,
        )
        self.val_set_2007 = VOCSegmentationDataset(
            root_dir=root_dir,
            year="2007",
            split="val",
            verbose=False,
        )

    def test_coco_dataset(self):
        da = SegmentationAnalysisManager(
            report_title="Testing Data-Gradients NEW",
            train_iterator=self.train_set_2007,
            val_iterator=self.val_set_2007,
            class_names=VOCSegmentationDataset.CLASS_NAMES,
            batches_early_stop=10,
            use_cache=False,
        )

        da.run()

    def test_coco_dataset_batch(self):

        da = SegmentationAnalysisManager(
            report_title="Testing Data-Gradients NEW",
            train_iterator=DataLoader(self.train_set_2007, batch_size=1),
            val_iterator=DataLoader(self.val_set_2007, batch_size=1),
            class_names=VOCSegmentationDataset.CLASS_NAMES,
            batches_early_stop=10,
            use_cache=False,
        )

        da.run()


if __name__ == "__main__":
    unittest.main()
