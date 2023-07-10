import torch
import unittest
from data_gradients.batch_processors.utils import to_one_hot


class ToOneHotTest(unittest.TestCase):
    def test_to_one_hot(self):
        # Test case 1
        labels = torch.tensor([[[0, 1], [2, 1]], [[1, 0], [2, 2]]])
        n_classes = 3

        one_hot_labels = to_one_hot(labels, n_classes)
        reconstructed_labels = torch.argmax(one_hot_labels, dim=1)

        self.assertTrue(torch.all(labels == reconstructed_labels))

        # Test case 2
        labels = torch.tensor([[[2, 0], [1, 2]], [[0, 1], [1, 0]]])
        n_classes = 3

        one_hot_labels = to_one_hot(labels, n_classes)
        reconstructed_labels = torch.argmax(one_hot_labels, dim=1)

        self.assertTrue(torch.all(labels == reconstructed_labels))

        # Test case 3
        labels = torch.tensor([[[0, 0], [0, 0]], [[1, 1], [1, 1]]])
        n_classes = 2

        one_hot_labels = to_one_hot(labels, n_classes)
        reconstructed_labels = torch.argmax(one_hot_labels, dim=1)

        self.assertTrue(torch.all(labels == reconstructed_labels))


if __name__ == "__main__":
    unittest.main()
