import unittest
import numpy as np

# from data_gradients.dataset_adapters.config.data_config import ImageChannels, ImageFormat

from data_gradients.utils.data_classes.image_channels import RGBChannels, BGRChannels, GrayscaleChannels, LABChannels, image_channel_instance_factory


class TestImageChannelValidation(unittest.TestCase):

    # RGBChannels tests
    def test_valid_rgb(self):
        self.assertTrue(RGBChannels.validate_channels("RGB"))
        self.assertTrue(RGBChannels.validate_channels("OORGB"))

    def test_invalid_rgb(self):
        self.assertFalse(RGBChannels.validate_channels("RG"))
        self.assertFalse(RGBChannels.validate_channels("ORGBOOORGB"))

    # BGRChannels tests
    def test_valid_bgr(self):
        self.assertTrue(BGRChannels.validate_channels("BGR"))
        self.assertTrue(BGRChannels.validate_channels("OBGR"))

    def test_invalid_bgr(self):
        self.assertFalse(BGRChannels.validate_channels("BRG"))
        self.assertFalse(BGRChannels.validate_channels("OBGROOBGR"))

    # GrayscaleChannels tests
    def test_valid_grayscale(self):
        self.assertTrue(GrayscaleChannels.validate_channels("G"))
        self.assertTrue(GrayscaleChannels.validate_channels("OG"))

    def test_invalid_grayscale(self):
        self.assertFalse(GrayscaleChannels.validate_channels("LAB"))
        self.assertFalse(GrayscaleChannels.validate_channels("GG"))
        self.assertFalse(GrayscaleChannels.validate_channels("OGOOOGO"))

    # LABChannels tests
    def test_valid_lab(self):
        self.assertTrue(LABChannels.validate_channels("LAB"))
        self.assertTrue(LABChannels.validate_channels("OOLAB"))
        self.assertTrue(LABChannels.validate_channels("OOLABO"))

    def test_invalid_lab(self):
        self.assertFalse(LABChannels.validate_channels("ALB"))
        self.assertFalse(LABChannels.validate_channels("LBA"))
        self.assertFalse(LABChannels.validate_channels("OLABOOLAB"))

    def test_single_match(self):
        test_data = [
            # Standard cases
            ("RGB", RGBChannels),
            ("BGR", BGRChannels),
            ("G", GrayscaleChannels),
            ("LAB", LABChannels),
            # Including other irrelevant channels
            ("OORGB", RGBChannels),
            ("OBGR", BGRChannels),
            ("GOOO", GrayscaleChannels),
            ("OOLABO", LABChannels),
        ]

        for channels_str, ExpectedClass in test_data:
            instance = image_channel_instance_factory(channels_str)
            self.assertIsInstance(instance, ExpectedClass)

    def test_ambiguous_channels(self):
        # Ambiguous cases
        ambiguous_channels = ["OORGBBGR", "GOOG", "BGRLO", "LABRGB", "RGBOOOOL"]

        for channels_str in ambiguous_channels:
            with self.assertRaises(ValueError):
                image_channel_instance_factory(channels_str)

    def test_unsupported_channels(self):
        # Unsupported cases
        unsupported_channels = ["XYZ", "RGO", "LL", "LLOOOLO"]

        for channels_str in unsupported_channels:
            with self.assertRaises(ValueError) as context:
                print(channels_str)
                image_channel_instance_factory(channels_str)

            # Check if the error message indicates unsupported format
            self.assertIn("unsupported", str(context.exception).lower())

    # RGBChannels conversion tests
    def test_rgb_conversion(self):
        channels_instance = RGBChannels("OORGBO")
        image = np.zeros((100, 100, 6), dtype=np.uint8)
        image[:, :, 2:5] = [255, 0, 0]  # Set RGB to pure Red

        converted_image = channels_instance.convert_image_to_rgb(image)
        self.assertTrue(np.all(converted_image == [255, 0, 0]))

    # BGRChannels conversion tests
    def test_bgr_conversion(self):
        channels_instance = BGRChannels("OOBGRO")
        image = np.zeros((100, 100, 5), dtype=np.uint8)
        image[:, :, 2:5] = [0, 0, 255]  # Set BGR to pure Red in BGR

        converted_image = channels_instance.convert_image_to_rgb(image)
        self.assertTrue(np.all(converted_image == [255, 0, 0]))

    # GrayscaleChannels conversion tests
    def test_grayscale_conversion(self):
        channels_instance = GrayscaleChannels("OOGOOO")
        image = np.zeros((100, 100, 6), dtype=np.uint8)
        image[:, :, 2] = 128  # Grayscale value

        converted_image = channels_instance.convert_image_to_rgb(image)
        self.assertTrue(np.all(converted_image == [128, 128, 128]))

    # LABChannels conversion tests
    def test_lab_conversion(self):
        channels_instance = LABChannels("OOLABO")
        image = np.zeros((100, 100, 6), dtype=np.uint8)

        image[:, :, 2:5] = [136, 208, 195]  # Some LAB value (approximating [255, 0, 0])
        converted_image = channels_instance.convert_image_to_rgb(image)

        # LAB -> RGB and RGB -> LAB is not super stable, so we work with +- 5
        self.assertTrue(np.all(converted_image[:, :, 0] >= 250))  # R ~ 255
        self.assertTrue(np.all(converted_image[:, :, 1] <= 5))  # G ~ 0
        self.assertTrue(np.all(converted_image[:, :, 2] <= 5))  # B ~ 0


if __name__ == "__main__":
    unittest.main()
