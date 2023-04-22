import os
import unittest
import shutil

from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor

from data_gradients.example_dataset.bdd_dataset import BDDDataset
from data_gradients.managers.segmentation_manager import SegmentationAnalysisManager


class SegmentationManagerTest(unittest.TestCase):
    def setUp(self):
        # Create torch DataSet
        train_dataset = BDDDataset(
            data_folder="../data/bdd_example",
            split="train",
            transform=Compose([ToTensor()]),
            target_transform=Compose([ToTensor()]),
        )
        val_dataset = BDDDataset(
            data_folder="../data/bdd_example",
            split="val",
            transform=Compose([ToTensor()]),
            target_transform=Compose([ToTensor()]),
        )

        # Create torch DataLoader
        self.train_loader = DataLoader(train_dataset, batch_size=8)
        self.val_loader = DataLoader(val_dataset, batch_size=8)

        self.num_classes = BDDDataset.NUM_CLASSES
        self.ignore_labels = BDDDataset.IGNORE_LABELS
        self.class_id_to_name = BDDDataset.CLASS_ID_TO_NAMES

    def test_black_image(self):
        if os.path.exists("logs"):
            shutil.rmtree("logs")

        da = SegmentationAnalysisManager(
            train_data=self.train_loader,
            val_data=self.val_loader,
            num_classes=self.num_classes,
            # Optionals
            ignore_labels=self.ignore_labels,
            id_to_name=self.class_id_to_name,
            samples_to_visualize=3,
            images_extractor=None,
            labels_extractor=None,
            threshold_soft_labels=0.5,
            batches_early_stop=75,
            short_run=False,
        )

        da.run()
        self.assertTrue(os.path.exists("logs"))
        shutil.rmtree("logs")
