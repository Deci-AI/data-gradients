import unittest

from torch.utils.data import DataLoader

from data_gradients.managers.detection_manager import DetectionAnalysisManager
from data_gradients.datasets.detection import COCODetectionDataset


COCO_DETECTION_CLASSES_LIST = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]


class TinyCOCODetectionDatasetTest(unittest.TestCase):
    def setUp(self):
        from pathlib import Path

        mini_coco_data_dir = str(Path(__file__).parent.parent.parent.parent.parent / "example_dataset" / "tinycoco")
        self.train_set = COCODetectionDataset(root_dir=mini_coco_data_dir, split="train", year=2017)
        self.val_set = COCODetectionDataset(root_dir=mini_coco_data_dir, split="val", year=2017)

    def test_coco_dataset(self):
        da = DetectionAnalysisManager(
            report_title="TEST_COCO_DATASET_DETECTION",
            train_data=self.train_set,
            val_data=self.val_set,
            class_names=COCO_DETECTION_CLASSES_LIST,
            batches_early_stop=20,
            use_cache=False,
            is_label_first=True,
            bbox_format="xywh",
        )
        da.run()

    def test_coco_dataset_batch(self):
        da = DetectionAnalysisManager(
            report_title="TEST_COCO_DATALOADER_DETECTION",
            train_data=DataLoader(self.train_set, batch_size=1),
            val_data=DataLoader(self.val_set, batch_size=1),
            class_names=COCO_DETECTION_CLASSES_LIST,
            batches_early_stop=20,
            use_cache=False,
            is_label_first=True,
            bbox_format="xywh",
        )
        da.run()


if __name__ == "__main__":
    unittest.main()
