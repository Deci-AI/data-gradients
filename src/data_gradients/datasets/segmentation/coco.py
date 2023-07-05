from typing import Union

from data_gradients.datasets.segmentation.coco_format import CocoFormatSegmentationDataset


class CocoSegmentationDataset(CocoFormatSegmentationDataset):
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
