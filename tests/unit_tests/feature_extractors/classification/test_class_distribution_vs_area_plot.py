import random
import unittest
import uuid

import numpy as np

from data_gradients.feature_extractors.classification.class_distribution_vs_area_scatter import ClassificationClassDistributionVsAreaPlot
from data_gradients.utils.data_classes.data_samples import ClassificationSample, Image
from data_gradients.visualize.seaborn_renderer import SeabornRenderer
from data_gradients.utils.data_classes.image_channels import ImageChannels
from data_gradients.dataset_adapters.formatters.utils import Uint8ImageFormat


class ClassificationClassDistributionVsAreaPlotTest(unittest.TestCase):
    def setUp(self) -> None:
        self.class_distribution = ClassificationClassDistributionVsAreaPlot()
        class_names = {0: "class_1", 1: "class_2", 2: "class_3", 3: "class_4"}

        for i in range(100):
            dummy_image = np.zeros((random.randint(100, 500), random.randint(100, 500), 3), dtype=np.uint8)
            self.class_distribution.update(
                ClassificationSample(
                    sample_id=str(uuid.uuid4()),
                    split="train",
                    image=Image(data=dummy_image, format=Uint8ImageFormat(), channels=ImageChannels.from_str("RGB")),
                    class_id=0,
                    class_names=class_names,
                )
            )

        for i in range(100):
            dummy_image = np.zeros((random.randint(300, 600), random.randint(200, 300), 3), dtype=np.uint8)
            self.class_distribution.update(
                ClassificationSample(
                    sample_id=str(uuid.uuid4()),
                    split="train",
                    image=Image(data=dummy_image, format=Uint8ImageFormat(), channels=ImageChannels.from_str("RGB")),
                    class_id=1,
                    class_names=class_names,
                )
            )

        for i in range(100):
            dummy_image = np.zeros((random.randint(100, 200), random.randint(700, 800), 3), dtype=np.uint8)
            self.class_distribution.update(
                ClassificationSample(
                    sample_id=str(uuid.uuid4()),
                    split="train",
                    image=Image(data=dummy_image, format=Uint8ImageFormat(), channels=ImageChannels.from_str("RGB")),
                    class_id=2,
                    class_names=class_names,
                )
            )

        for i in range(100):
            dummy_image = np.zeros((random.randint(200, 250), random.randint(200, 250), 3), dtype=np.uint8)
            self.class_distribution.update(
                ClassificationSample(
                    sample_id=str(uuid.uuid4()),
                    split="valid",
                    image=Image(data=dummy_image, format=Uint8ImageFormat(), channels=ImageChannels.from_str("RGB")),
                    class_id=3,
                    class_names=class_names,
                )
            )

        for i in range(100):
            dummy_image = np.zeros((220, 230, 3), dtype=np.uint8)
            self.class_distribution.update(
                ClassificationSample(
                    sample_id=str(uuid.uuid4()),
                    split="valid",
                    image=Image(data=dummy_image, format=Uint8ImageFormat(), channels=ImageChannels.from_str("RGB")),
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
