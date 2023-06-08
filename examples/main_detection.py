import numpy as np
from torch.utils.data import DataLoader
from torchvision.datasets import CocoDetection
from torchvision.transforms import transforms

from data_gradients.managers.detection_manager import DetectionAnalysisManager


if __name__ == "__main__":
    """
    Example script for running the Deci-Dataset-Analyzer tool.
    Arguments required for SegmentationAnalysisManager() are:
        train_data  -> An Iterable (i.e., torch data loader) containing train data
        val_data    -> An Iterable (i.e., torch data loader) containing valid data
        num_classes -> Number of valid classes
    Also if there are ignore labels, please pass them as a List[int].
    Default ignore label will be [0] as for background only.

    Example of use (CityScapes):
        train_dataset = CityScapesDataSet(root=dataset_root,
                                          image_set='train',
                                          transform=transforms_list)
        train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        number_of_classes = 19
        ignore_labels = [0, 1, 2, 3, 4, 5, 6, 9, 10, 14, 15, 16, 18, 29, 30, -1]

    """
    root = "../example_dataset/tinycoco"
    # Create torch DataSet
    train_dataset = CocoDetection(root=f"{root}/images/train2017", annFile=f"{root}/annotations/instances_train2017.json", transform=transforms.Compose([transforms.PILToTensor(),transforms.Pad(640)]))
    val_dataset = CocoDetection(root=f"{root}/images/val2017", annFile=f"{root}/annotations/instances_val2017.json", transform=transforms.Compose([transforms.PILToTensor(),transforms.Pad(640)]))


    def collate_fn(batch):
        images = []
        bboxes = []
        for index, (image, targets) in enumerate(batch):
            images.append(image)
            for target in targets:
                bboxes.append(np.array([index] + target["bbox"]))
        return images, bboxes

    # Create torch DataLoader
    train_loader = DataLoader(train_dataset, batch_size=8, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=8, collate_fn=collate_fn)

    class_names = [item["name"] for k, item in train_dataset.coco.cats.items()]

    da = DetectionAnalysisManager(
        report_title="Testing Data-Gradients",
        train_data=train_dataset,
        val_data=val_dataset,
        class_names=class_names,
        # Optionals
        images_extractor=None,
        labels_extractor=None,
        batches_early_stop=75,
    )

    da.run()
