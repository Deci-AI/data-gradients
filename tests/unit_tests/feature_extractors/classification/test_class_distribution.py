import unittest

import numpy as np

from data_gradients.feature_extractors.classification.class_distribution import ClassificationClassDistribution
from data_gradients.utils.data_classes.data_samples import ImageChannelFormat, ClassificationSample
from data_gradients.visualize.seaborn_renderer import SeabornRenderer


class ClassificationClassDistributionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.class_distribution = ClassificationClassDistribution()

        dummy_image = np.zeros((100, 100, 3), dtype=np.uint8)
        class_names = ["class_1", "class_2", "class_3", "class_4"]
        self.class_distribution.update(ClassificationSample(
            sample_id="sample_1",
            split="train",
            image=dummy_image,
            image_format=ImageChannelFormat.RGB,
            class_id=0,
            class_names=class_names,
        ))

        self.class_distribution.update(ClassificationSample(
            sample_id="sample_2",
            split="train",
            image=dummy_image,
            image_format=ImageChannelFormat.RGB,
            class_id=1,
            class_names=class_names,
        ))
        self.class_distribution.update(ClassificationSample(
            sample_id="sample_3",
            split="train",
            image=dummy_image,
            image_format=ImageChannelFormat.RGB,
            class_id=2,
            class_names=class_names,
        ))

        self.class_distribution.update(ClassificationSample(
            sample_id="sample_4",
            split="valid",
            image=dummy_image,
            image_format=ImageChannelFormat.RGB,
            class_id=3,
            class_names=class_names,
        ))
        self.class_distribution.update(ClassificationSample(
            sample_id="sample_5",
            split="valid",
            image=dummy_image,
            image_format=ImageChannelFormat.RGB,
            class_id=0,
            class_names=class_names,
        ))

    def test_update_and_aggregate(self):
        # Create a sample SegmentationSample object for testing
        output_json = self.class_distribution.aggregate().json

        expected_json = {
            "train": {
                "class_1": 1,
                "class_2": 1,
                "class_3": 1,
                "class_4": 0,
            },
            "valid": {
                "class_1": 1,
                "class_2": 0,
                "class_3": 0,
                "class_4": 1,
            },
        }

        self.assertEqual(output_json, expected_json)

    def test_plot(self):
        feature = self.class_distribution.aggregate()
        sns = SeabornRenderer()
        f = sns.render(feature.data, feature.plot_options)
        f.show()



if __name__ == "__main__":
    unittest.main()
