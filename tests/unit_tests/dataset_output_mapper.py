import os.path
import unittest
import tempfile
import shutil

import numpy as np
from PIL import Image
import torch
from data_gradients.dataset_adapters.output_mapper.dataset_output_mapper import DatasetOutputMapper


class TestImageConverter(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.rbg_image_path = os.path.join(self.tmp_dir, "dummy_rgb_image.jpg")
        self.grayscale_image_path = os.path.join(self.tmp_dir, "dummy_grayscale_image.jpg")

        data = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        Image.fromarray(data, "RGB").save(self.rbg_image_path)
        Image.fromarray(data[:, :, 0], "L").save(self.grayscale_image_path)

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp_dir)

    def test_int_input(self):
        tensor = 3
        output = DatasetOutputMapper._to_torch(tensor)
        self.assertTrue(torch.equal(output, torch.tensor(tensor, dtype=torch.int64)))

    def test_pytorch_tensor_input(self):
        tensor = torch.randn((3, 100, 100))
        output = DatasetOutputMapper._to_torch(tensor)
        self.assertTrue(torch.equal(output, tensor))

    def test_numpy_ndarray_input(self):
        array = np.random.rand(100, 100, 3)
        output = DatasetOutputMapper._to_torch(array)
        self.assertTrue(torch.equal(output, torch.from_numpy(array)))

    def test_pil_image_input(self):
        image = Image.new("RGB", (100, 100))
        output = DatasetOutputMapper._to_torch(image)
        self.assertIsInstance(output, torch.Tensor)
        self.assertEqual(output.ndim, 3)

    def test_string_path_input(self):
        output = DatasetOutputMapper._to_torch(self.rbg_image_path)
        self.assertIsInstance(output, torch.Tensor)

        output = DatasetOutputMapper._to_torch(self.grayscale_image_path)
        self.assertIsInstance(output, torch.Tensor)

    def test_list_input(self):
        data = [np.random.rand(100, 100, 3), Image.new("RGB", (100, 100)), self.rbg_image_path, self.rbg_image_path]
        output = DatasetOutputMapper._to_torch(data)
        self.assertIsInstance(output, torch.Tensor)
        self.assertEqual(output.shape, (4, 100, 100, 3))

    def test_unsupported_type_input(self):
        with self.assertRaises(TypeError):
            DatasetOutputMapper._to_torch({"unsupported": "type"})

    def test_invalid_image_path(self):
        with self.assertRaises(Exception):
            DatasetOutputMapper._to_torch("invalid_path.jpg")

    def test_edge_cases(self):
        # Grayscale
        gray_image = np.random.rand(100, 100)
        output = DatasetOutputMapper._to_torch(gray_image)
        self.assertEqual(output.shape, (100, 100))

        # Empty list
        with self.assertRaises(Exception):  # Depending on your function's behavior for empty lists
            DatasetOutputMapper._to_torch([])


if __name__ == "__main__":
    unittest.main()
