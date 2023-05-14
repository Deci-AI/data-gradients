import torch
import unittest
from data_gradients.batch_processors.formatters.detection import convert_bbox_to_label_x1_y1_x2_y2


class TestConvertBBoxToLabelX1Y1X2Y2(unittest.TestCase):
    def test_label_first_cxcywh(self):
        input_tensor = torch.tensor([[0, 50, 50, 100, 100], [1, 100, 100, 200, 200]])
        expected_output = torch.tensor([[0, 0, 0, 100, 100], [1, 0, 0, 200, 200]])
        output = convert_bbox_to_label_x1_y1_x2_y2(input_tensor)
        self.assertTrue(torch.allclose(output, expected_output))

    def test_label_last_cxcywh(self):
        input_tensor = torch.tensor([[50, 50, 100, 100, 0], [100, 100, 200, 200, 1]])
        expected_output = torch.tensor([[0, 0, 0, 100, 100], [1, 0, 0, 200, 200]])
        output = convert_bbox_to_label_x1_y1_x2_y2(input_tensor)
        self.assertTrue(torch.allclose(output, expected_output))

    def test_label_first_xyxy(self):
        input_tensor = torch.tensor([[0, 0, 0, 100, 100], [1, 0, 0, 200, 200]])
        expected_output = torch.tensor([[0, 0, 0, 100, 100], [1, 0, 0, 200, 200]])
        output = convert_bbox_to_label_x1_y1_x2_y2(input_tensor)
        self.assertTrue(torch.allclose(output, expected_output))

    def test_label_last_xyxy(self):
        input_tensor = torch.tensor([[0, 0, 100, 100, 0], [0, 0, 200, 200, 1]])
        expected_output = torch.tensor([[0, 0, 0, 100, 100], [1, 0, 0, 200, 200]])
        output = convert_bbox_to_label_x1_y1_x2_y2(input_tensor)
        self.assertTrue(torch.allclose(output, expected_output))

    # Add more test cases for other formats, if necessary


if __name__ == "__main__":
    unittest.main()
