from typing import Union

from data_gradients.datasets.detection.coco_format_detection_dataset import CocoFormatDetectionDataset


class CocoDetectionDataset(CocoFormatDetectionDataset):
    def __init__(self, root_dir: str, split: str, year: Union[int, str] = 2017):
        """
        :param root_dir: Where the data is stored.
        :param split:    Which split of the data to use. `train` or `val`
        :param year:     Which year of the data to use. Default to `2017`
        """
        super().__init__(
            root_dir=root_dir,
            images_dir=f"images/{split}{year}",
            annotation_file_path=f"annotations/instances_{split}{year}.json",
        )
