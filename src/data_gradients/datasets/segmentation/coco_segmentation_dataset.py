from typing import Union

from data_gradients.datasets.segmentation.coco_format_segmentation_dataset import COCOFormatSegmentationDataset


class COCOSegmentationDataset(COCOFormatSegmentationDataset):
    """The COCOSegmentationDataset class is a convenience subclass of the COCOFormatSegmentationDataset that simplifies
    the instantiation for the widely-used COCO Segmentation Dataset.

    This class assumes the default COCO dataset structure and naming conventions. The data should be stored in a specific
    structure where each split of data (train, val) and year of the dataset is kept in a different directory.

    #### Expected folder structure

    ```
    dataset_root/
        ├── images/
        │   ├── train2017/
        │   │   ├── 1.jpg
        │   │   ├── 2.jpg
        │   │   └── ...
        │   └── val2017/
        │       ├── 15481.jpg
        │       ├── 15482.jpg
        │       └── ...
        └── annotations/
            ├── instances_train2017.json
            └── instances_val2017.json
    ```

    #### Instantiation

    ```python
    from data_gradients.datasets.segmentation import COCOSegmentationDataset
    train_set = COCOSegmentationDataset(root_dir="<path/to/dataset_root>", split="train", year=2017)
    val_set = COCOSegmentationDataset(root_dir="<path/to/dataset_root>", split="val", year=2017)
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
