import random
import unittest

import numpy as np

from data_gradients.feature_extractors.object_detection.bounding_boxes_iou import DetectionBoundingBoxIoU
from data_gradients.utils.data_classes.data_samples import DetectionSample
from data_gradients.visualize.seaborn_renderer import SeabornRenderer


class BoundingBoxesIoUTest(unittest.TestCase):
    def generate_random_dataset(self, num_classes, num_samples, average_number_of_bboxes_per_sample):
        samples = []
        class_names = np.array([f"class_{i}" for i in range(num_classes)])
        for sample_index in range(num_samples):
            num_boxes = random.randint(0, average_number_of_bboxes_per_sample)
            class_ids = np.random.randint(0, num_classes, size=num_boxes)
            bboxes_xyxy = np.random.randint(0, 320, size=(num_boxes, 4))
            bboxes_xyxy[:, 2:] += bboxes_xyxy[:, :2]

            train_sample = DetectionSample(
                sample_id=str(sample_index),
                split="train" if random.random() < 0.8 else "val",
                image=np.zeros((640, 640, 3)),
                image_channels=str.RGB,
                bboxes_xyxy=bboxes_xyxy,
                class_ids=class_ids,
                class_names=class_names,
            )
            samples.append(train_sample)
        return samples

    def test_plot(self):
        train_sample = DetectionSample(
            sample_id="sample_1",
            split="train",
            image=np.zeros((100, 100, 3)),
            image_channels=str.RGB,
            bboxes_xyxy=np.array(
                [
                    [10, 10, 20, 20],
                    [20, 20, 30, 30],
                    [50, 50, 60, 60],
                    [12, 13, 21, 22],
                    [19, 19, 32, 31],
                    [40, 55, 55, 65],
                ]
            ),
            class_ids=np.array([0, 1, 2, 2, 3, 4]),
            class_names=["class_1", "class_2", "class_3", "class_4", "class_5"],
        )

        valid_sample = DetectionSample(
            sample_id="sample_1",
            split="valid",
            image=np.zeros((100, 100, 3)),
            image_channels=str.RGB,
            bboxes_xyxy=np.array(
                [
                    [40, 40, 50, 50],
                    [50, 50, 60, 60],
                    [70, 70, 80, 80],
                    [42, 43, 51, 52],
                ]
            ),
            class_ids=np.array([0, 1, 2, 2]),
            class_names=["class_1", "class_2", "class_3", "class_4"],
        )

        extractor = DetectionBoundingBoxIoU(num_bins=20, class_agnostic=True)
        extractor.update(train_sample)
        extractor.update(valid_sample)
        feature = extractor.aggregate()
        sns = SeabornRenderer()
        f = sns.render(feature.data, feature.plot_options)
        f.savefig("test_plot.png")
        f.show()

    def test_plot_80_classes(self):
        samples = self.generate_random_dataset(num_classes=80, num_samples=1000, average_number_of_bboxes_per_sample=10)

        extractor = DetectionBoundingBoxIoU(num_bins=20, class_agnostic=True)
        for sample in samples:
            extractor.update(sample)

        feature = extractor.aggregate()
        sns = SeabornRenderer()
        f = sns.render(feature.data, feature.plot_options)
        f.savefig("test_plot_80_classes.png")
        f.show()

    def test_plot_16_classes(self):
        samples = self.generate_random_dataset(num_classes=16, num_samples=1000, average_number_of_bboxes_per_sample=10)

        extractor = DetectionBoundingBoxIoU(num_bins=10, class_agnostic=True)
        for sample in samples:
            extractor.update(sample)

        feature = extractor.aggregate()
        sns = SeabornRenderer()
        f = sns.render(feature.data, feature.plot_options)
        f.savefig("test_plot_16_classes.png")
        f.show()


if __name__ == "__main__":
    unittest.main()
