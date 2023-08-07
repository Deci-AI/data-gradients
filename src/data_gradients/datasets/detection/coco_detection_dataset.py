from typing import Union

from data_gradients.datasets.detection.coco_format_detection_dataset import COCOFormatDetectionDataset


class COCODetectionDataset(COCOFormatDetectionDataset):
    """Coco Detection Dataset expects the exact same annotation files and dataset structure os the original Coco dataset.

    #### Expected folder structure
    The dataset folder structure should

    Example:
    ```
    dataset_root/
        ├── images/
        │   ├── train2017/
        │   ├── val2017/
        │   └── ...
        └── annotations/
            ├── instances_train2017.json
            ├── instances_val2017.json
            └── ...
    ```

    #### Instantiation
    To instantiate a dataset object for training data of the year 2017, use the following code:

    ```python
    from data_gradients.datasets.detection import COCODetectionDataset

    train_set = COCODetectionDataset(root_dir="<path/to/dataset_root>", split="train", year=2017)
    val_set = COCODetectionDataset(root_dir="<path/to/dataset_root>", split="val", year=2017)
    ```
    """

    def __init__(self, root_dir: str, split: str, year: Union[int, str] = 2017):
        """
        :param root_dir: Where the data is stored.
        :param split:    Which split of the data to use. `train` or `val`
        :param year:     Which year of the data to use. Default to `2017`
        """
        super().__init__(
            root_dir=root_dir,
            images_subdir=f"images/{split}{year}",
            annotation_file_path=f"annotations/instances_{split}{year}.json",
        )
