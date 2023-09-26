import unittest

from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor

from data_gradients.datasets.bdd_dataset import BDDDataset
from data_gradients.managers.segmentation_manager import SegmentationAnalysisManager


class BDDDSegmentationDatasetTest(unittest.TestCase):
    def setUp(self):
        from pathlib import Path

        bdd_example_data_dir = str(Path(__file__).parent.parent.parent.parent.parent / "example_dataset" / "bdd_example")
        self.train_set = BDDDataset(
            data_folder=bdd_example_data_dir,
            split="train",
            transform=Compose([ToTensor()]),
            target_transform=Compose([ToTensor()]),
        )
        self.val_set = BDDDataset(
            data_folder=bdd_example_data_dir,
            split="val",
            transform=Compose([ToTensor()]),
            target_transform=Compose([ToTensor()]),
        )

    def test_coco_dataset(self):
        da = SegmentationAnalysisManager(
            report_title="TEST_BDD_DATASET_SEGMENTATION",
            train_data=self.train_set,
            val_data=self.val_set,
            class_names=BDDDataset.CLASS_NAMES,
            class_names_to_use=BDDDataset.CLASS_NAMES[:-1],
            use_cache=False,
            batches_early_stop=5,
            is_batch=False,
        )
        da.run()

    def test_coco_dataset_batch(self):
        da = SegmentationAnalysisManager(
            report_title="TEST_BDD_DATALOADER_SEGMENTATION",
            train_data=DataLoader(self.train_set, batch_size=1),
            val_data=DataLoader(self.val_set, batch_size=1),
            class_names=BDDDataset.CLASS_NAMES,
            class_names_to_use=BDDDataset.CLASS_NAMES[:-1],
            use_cache=False,
            batches_early_stop=5,
            is_batch=True,
        )
        da.run()


if __name__ == "__main__":
    unittest.main()
