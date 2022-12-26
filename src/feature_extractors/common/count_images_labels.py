from src.utils import SegBatchData
from src.feature_extractors.feature_extractor_abstract import FeatureExtractorAbstract
from src.logger.logger_utils import create_bar_plot, create_json_object


class NumberOfImagesLabels(FeatureExtractorAbstract):
    """
    Common task feature extractor -
    Count number of images, labels and background images and display as plot-bar.
    """
    def __init__(self):
        super().__init__()
        self._num_images = {'train': 0, 'val': 0}
        self._num_labels = {'train': 0, 'val': 0}
        self._num_bg_images = {'train': 0, 'val': 0}

    def execute(self, data: SegBatchData):
        self._num_images[data.split] += len(data.images)
        self._num_labels[data.split] += len(data.labels)
        for i, image_contours in enumerate(data.contours):
            if len(image_contours) < 1:
                self._num_bg_images[data.split] += 1

    def _process(self):
        for split in ['train', 'val']:

            values = [self._num_images[split], self._num_labels[split], self._num_bg_images[split]]
            keys = ["images", "labels", "background images"]

            create_bar_plot(ax=self.ax, labels=keys, data=values, y_label='Total #',
                            title='# Images & Labels', split=split, ticks_rotation=0,
                            color=self.colors[split], yticks=True)
            self.json_object.update({split: create_json_object(values, keys)})

