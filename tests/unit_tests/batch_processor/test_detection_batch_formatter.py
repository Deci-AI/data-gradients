import torch
import unittest
from data_gradients.dataset_adapters.formatters.detection import DetectionBatchFormatter
from data_gradients.config.data import DetectionDataConfig


class DetectionBatchFormatterTest(unittest.TestCase):
    def setUp(self):
        self.empty_data_config = DetectionDataConfig(use_cache=False, is_label_first=True, xyxy_converter="xyxy")
        self.channel_last_image = torch.zeros(64, 32, 3, dtype=torch.uint8)
        self.channel_first_image = torch.zeros(3, 64, 32, dtype=torch.uint8)
        self.channel_last_images = torch.zeros(1, 64, 32, 3, dtype=torch.uint8)
        self.channel_first_images = torch.zeros(1, 3, 64, 32, dtype=torch.uint8)

    def test_format_sample_image(self):
        formatter = DetectionBatchFormatter(data_config=self.empty_data_config, class_names=["0", "1"], class_names_to_use=["0", "1"], n_image_channels=3)
        target_n5 = torch.Tensor(
            [
                [0, 10, 20, 15, 25],
                [0, 5, 10, 15, 25],
                [0, 5, 10, 15, 25],
                [1, 10, 20, 15, 25],
            ]
        )
        images, labels = formatter.format(self.channel_last_image, target_n5)
        self.assertTrue(torch.equal(images, self.channel_first_images))
        self.assertTrue(torch.equal(labels[0], target_n5))

        images, labels = formatter.format(self.channel_first_image, target_n5)
        self.assertTrue(torch.equal(images, self.channel_first_images))
        self.assertTrue(torch.equal(labels[0], target_n5))

    def test_format_batch_n5(self):
        formatter = DetectionBatchFormatter(data_config=self.empty_data_config, class_names=["0", "1"], class_names_to_use=["0", "1"], n_image_channels=3)
        target_sample_n5 = torch.Tensor(
            [
                [
                    [0, 10, 20, 15, 25],
                    [0, 5, 10, 15, 25],
                    [1, 5, 10, 15, 25],
                ],
                [
                    [0, 5, 10, 15, 25],
                    [1, 10, 20, 15, 25],
                    [0, 0, 0, 0, 0],
                ],
            ]
        )
        images, labels = formatter.format(self.channel_last_images, target_sample_n5)

        self.assertTrue(torch.equal(images, self.channel_first_images))
        expected_first_batch = torch.Tensor(
            [
                [0, 10, 20, 15, 25],
                [0, 5, 10, 15, 25],
                [1, 5, 10, 15, 25],
            ],
        )
        expected_second_batch = torch.Tensor(
            [
                [0, 5, 10, 15, 25],
                [1, 10, 20, 15, 25],
            ]
        )
        self.assertTrue(torch.equal(labels[0], expected_first_batch))
        self.assertTrue(torch.equal(labels[1], expected_second_batch))

    def test_format_batch_n6(self):
        formatter = DetectionBatchFormatter(data_config=self.empty_data_config, class_names=["0", "1"], class_names_to_use=["0", "1"], n_image_channels=3)
        target_n6 = torch.Tensor(
            [
                [0, 0, 10, 20, 15, 25],
                [0, 0, 5, 10, 15, 25],
                [0, 1, 5, 10, 15, 25],
                [1, 0, 5, 10, 15, 25],
                [1, 1, 10, 20, 15, 25],
            ]
        )
        images, labels = formatter.format(self.channel_last_images, target_n6)

        self.assertTrue(torch.equal(images, self.channel_first_images))
        expected_first_batch = torch.Tensor(
            [
                [0, 10, 20, 15, 25],
                [0, 5, 10, 15, 25],
                [1, 5, 10, 15, 25],
            ],
        )
        expected_second_batch = torch.Tensor(
            [
                [0, 5, 10, 15, 25],
                [1, 10, 20, 15, 25],
            ]
        )
        self.assertTrue(torch.equal(labels[0], expected_first_batch))
        self.assertTrue(torch.equal(labels[1], expected_second_batch))

    def test_format_empty_sample(self):
        formatter = DetectionBatchFormatter(data_config=self.empty_data_config, class_names=["0", "1"], class_names_to_use=["0", "1"], n_image_channels=3)
        expected_output_target = torch.zeros(0, 5)

        empty_tensor = torch.Tensor([])
        images, labels = formatter.format(self.channel_last_image, empty_tensor)
        self.assertTrue(torch.equal(images, self.channel_first_images))
        self.assertTrue(torch.equal(labels[0], expected_output_target))

        empty_zero_tensor = torch.zeros(0, 5)
        images, labels = formatter.format(self.channel_first_image, empty_zero_tensor)
        self.assertTrue(torch.equal(images, self.channel_first_images))
        self.assertTrue(torch.equal(labels[0], expected_output_target))

    def test_format_empty_batch(self):
        formatter = DetectionBatchFormatter(data_config=self.empty_data_config, class_names=["0", "1"], class_names_to_use=["0", "1"], n_image_channels=3)
        batch_size = 7
        expected_output_sample_target = torch.zeros(0, 5)

        empty_tensor = torch.zeros(batch_size, 0)
        images, labels = formatter.format(self.channel_last_images, empty_tensor)
        self.assertTrue(torch.equal(images, self.channel_first_images))
        self.assertEqual(len(labels), batch_size)
        for sample in labels:
            self.assertTrue(torch.equal(sample, expected_output_sample_target))

        empty_zero_tensor = torch.zeros(batch_size, 0, 5)
        images, labels = formatter.format(self.channel_first_images, empty_zero_tensor)
        self.assertTrue(torch.equal(images, self.channel_first_images))
        self.assertEqual(len(labels), batch_size)
        for sample in labels:
            self.assertTrue(torch.equal(sample, expected_output_sample_target))

    def test_group_detection_batch(self):
        flat_batch = torch.Tensor(
            [
                [0, 2, 10, 20, 15, 25],
                [0, 1, 5, 10, 15, 25],
                [0, 1, 5, 10, 15, 25],
                [1, 2, 10, 20, 15, 25],
                [3, 2, 10, 20, 15, 25],
                [5, 2, 10, 20, 15, 25],
                [6, 2, 10, 20, 15, 25],
            ]
        )

        expected_grouped_batch = torch.Tensor(
            [
                [[2, 10, 20, 15, 25], [1, 5, 10, 15, 25], [1, 5, 10, 15, 25]],
                [[2, 10, 20, 15, 25], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
                [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
                [[2, 10, 20, 15, 25], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
                [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
                [[2, 10, 20, 15, 25], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
                [[2, 10, 20, 15, 25], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
            ]
        )
        grouped_batch = DetectionBatchFormatter.group_detection_batch(flat_batch)
        self.assertTrue(torch.equal(grouped_batch, expected_grouped_batch))


if __name__ == "__main__":
    unittest.main()
