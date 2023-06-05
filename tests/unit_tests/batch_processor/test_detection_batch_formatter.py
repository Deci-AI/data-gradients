import torch
import unittest
from data_gradients.batch_processors.formatters.detection import DetectionBatchFormatter


class GroupDetectionBatchTest(unittest.TestCase):
    def test_update_and_aggregate(self):
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
