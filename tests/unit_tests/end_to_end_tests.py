import random
import unittest

import numpy as np
import torch
from torch.utils.data import DataLoader

from data_gradients.managers.classification_manager import ClassificationAnalysisManager


class EndToEndTest(unittest.TestCase):
    """ """

    def test_classification_task(self):
        class_names = ["class_1", "class_2", "class_3", "class_4"]

        train_samples = []
        for i in range(100):
            dummy_image = torch.randn((3, random.randint(100, 500), random.randint(100, 500)), dtype=torch.float32)
            train_samples += [(dummy_image, 0)]

        for i in range(100):
            dummy_image = torch.randn((3, random.randint(300, 600), random.randint(200, 300)), dtype=torch.float32)
            train_samples += [(dummy_image, 1)]

        for i in range(100):
            dummy_image = torch.randn((3, random.randint(100, 200), random.randint(700, 800)), dtype=torch.float32)
            train_samples += [(dummy_image, 2)]

        valid_samples = []
        for i in range(100):
            dummy_image = torch.randn((3, random.randint(200, 250), random.randint(200, 250)), dtype=torch.float32)
            valid_samples += [(dummy_image, 3)]

        for i in range(100):
            dummy_image = torch.randn((220, 230, 3), dtype=torch.float32)
            valid_samples += [(dummy_image, 0)]

        manager = ClassificationAnalysisManager(
            train_data=DataLoader(train_samples),
            val_data=DataLoader(valid_samples),
            report_title="End to End Classification Test",
            class_names=class_names,
            batches_early_stop=None,
            n_image_channels=3,
        )
        manager.run()


if __name__ == "__main__":
    unittest.main()
