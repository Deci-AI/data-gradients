import unittest

import numpy as np

from data_gradients.feature_extractors.classification.class_frequency import ClassificationClassFrequency
from data_gradients.utils.data_classes.data_samples import ClassificationSample
from data_gradients.visualize.seaborn_renderer import SeabornRenderer
from data_gradients.utils.data_classes.image_channels import ImageChannels


class ClassificationClassFrequencyTest(unittest.TestCase):
    def setUp(self) -> None:
        self.class_distribution = ClassificationClassFrequency()

        dummy_image = np.zeros((100, 100, 3), dtype=np.uint8)
        class_names = ["class_1", "class_2", "class_3", "class_4"]
        self.class_distribution.update(
            ClassificationSample(
                sample_id="sample_1",
                split="train",
                image=dummy_image,
                image_channels=ImageChannels.from_str("RGB"),
                class_id=0,
                class_names=class_names,
            )
        )

        self.class_distribution.update(
            ClassificationSample(
                sample_id="sample_2",
                split="train",
                image=dummy_image,
                image_channels=ImageChannels.from_str("RGB"),
                class_id=1,
                class_names=class_names,
            )
        )
        self.class_distribution.update(
            ClassificationSample(
                sample_id="sample_3",
                split="train",
                image=dummy_image,
                image_channels=ImageChannels.from_str("RGB"),
                class_id=2,
                class_names=class_names,
            )
        )

        self.class_distribution.update(
            ClassificationSample(
                sample_id="sample_4",
                split="valid",
                image=dummy_image,
                image_channels=ImageChannels.from_str("RGB"),
                class_id=3,
                class_names=class_names,
            )
        )
        self.class_distribution.update(
            ClassificationSample(
                sample_id="sample_5",
                split="valid",
                image=dummy_image,
                image_channels=ImageChannels.from_str("RGB"),
                class_id=0,
                class_names=class_names,
            )
        )

    def test_plot(self):
        feature = self.class_distribution.aggregate()
        sns = SeabornRenderer()
        f = sns.render(feature.data, feature.plot_options)
        f.savefig(fname=self.class_distribution.__class__.__name__ + ".png")
        f.show()


if __name__ == "__main__":
    unittest.main()
