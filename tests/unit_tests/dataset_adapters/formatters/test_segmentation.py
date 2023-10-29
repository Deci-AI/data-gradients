import unittest
import torch
from data_gradients.dataset_adapters.config.data_config import SegmentationDataConfig


from data_gradients.dataset_adapters.formatters.segmentation import SegmentationBatchFormatter
from data_gradients.utils.data_classes.image_channels import ImageChannels


class TestSegmentationBatchFormatter(unittest.TestCase):
    def setUp(self):
        self.data_config = SegmentationDataConfig(
            class_names=["background", "class1", "class2", "class3"],
            class_names_to_use=["class1", "class2", "class3"],
            image_channels=ImageChannels.from_str("RGB"),
        )
        self.formatter = SegmentationBatchFormatter(data_config=self.data_config, threshold_value=0.5)

    def test_basic_shapes(self):
        images = torch.rand(8, 3, 64, 64)  # Batch of images
        labels = torch.randint(0, 4, (8, 64, 64))  # Corresponding labels
        out_images, out_labels = self.formatter.format(images, labels)
        self.assertEqual(out_images.shape, (8, 3, 64, 64))
        self.assertEqual(out_labels.shape, (8, 64, 64))

    def test_single_sample(self):
        images = torch.rand(3, 64, 64)
        labels = torch.randint(0, 4, (64, 64))
        out_images, out_labels = self.formatter.format(images, labels)
        self.assertEqual(out_images.shape, (1, 3, 64, 64))
        self.assertEqual(out_labels.shape, (1, 64, 64))

    def test_image_formats(self):
        images = torch.rand(8, 64, 64, 3)
        labels = torch.randint(0, 4, (8, 64, 64))
        out_images, _ = self.formatter.format(images, labels)
        self.assertEqual(out_images.shape, (8, 3, 64, 64))

    def test_label_formats(self):
        images = torch.rand(8, 3, 64, 64)
        labels = torch.randn(8, 4, 64, 64)  # Float labels simulating one-hot encoded but not strictly [0,1]
        _, out_labels = self.formatter.format(images, labels)
        self.assertEqual(out_labels.shape, (8, 64, 64))

    def test_ensure_hard_labels(self):
        images = torch.rand(8, 3, 64, 64)
        labels = torch.randint(0, 4, (8, 64, 64)) / 255
        _, out_labels = self.formatter.format(images, labels)
        self.assertEqual(out_labels.shape, (8, 64, 64))
        self.assertTrue(torch.all(out_labels <= 4))

    def test_ignore_classes(self):
        images = torch.rand(8, 3, 64, 64)
        labels = torch.randint(0, 4, (8, 64, 64))
        _, out_labels = self.formatter.format(images, labels)
        self.assertTrue(torch.all(out_labels != 0))  # 0 (background) should be ignored

    def test_threshold_values(self):
        data_config = SegmentationDataConfig(
            class_names=["class1"],
            image_channels=ImageChannels.from_str("RGB"),
        )
        formatter = SegmentationBatchFormatter(data_config=data_config, threshold_value=0.05)

        images = torch.rand(8, 3, 64, 64)
        labels = torch.rand(8, 64, 64)
        _, out_labels = formatter.format(images, labels)

        self.assertTrue(torch.all((out_labels == 0) | (out_labels == 1)))  # Threshold should be applied to to exclusively 0 and 1 labels (categorical)
        self.assertGreater((out_labels == 1).sum(), (out_labels == 0).sum())  # Threshold is low, so most values should be 1

    def test_image_value_ranges(self):
        images = torch.rand(8, 3, 64, 64) * 2 - 1  # [-1, 1] range
        labels = torch.randint(0, 4, (8, 64, 64))
        out_images, _ = self.formatter.format(images, labels)
        self.assertTrue(torch.all(out_images >= 0) and torch.all(out_images <= 255))

    def test_label_conversion(self):
        images = torch.rand(8, 3, 64, 64)
        labels = torch.zeros(8, 4, 64, 64)
        labels[:, 2, :, :] = 1  # Simulate one-hot encoded with class 2 everywhere
        _, out_labels = self.formatter.format(images, labels)
        self.assertTrue(torch.all(out_labels == 2))


if __name__ == "__main__":
    unittest.main()
