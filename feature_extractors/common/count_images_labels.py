from utils.data_classes import BatchData
from feature_extractors.feature_extractor_abstract import FeatureExtractorAbstract
from logger.logger_utils import create_bar_plot


class NumberOfImagesLabels(FeatureExtractorAbstract):
    def __init__(self):
        super().__init__()
        self._num_images: int = 0
        self._num_labels: int = 0
        self._num_objects: int = 0

    def execute(self, data: BatchData):
        self._num_images += len(data.images)
        self._num_labels += len(data.labels)

        for image_contours in data.contours:
            self._num_objects += sum([len(cls_contours) for cls_contours in image_contours])

    def process(self, ax, train):
        create_bar_plot(ax=ax, data=[self._num_images, self._num_labels, self._num_objects / 100],
                        labels=["images", "labels", "objects / 100"], y_label='Total #', title='# Images, Labels & Objects',
                        train=train, ticks_rotation=0, color=self.colors[int(train)])
