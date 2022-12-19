from src.utils import SegBatchData
from src.feature_extractors.feature_extractor_abstract import FeatureExtractorAbstract
from src.logger.logger_utils import create_bar_plot


class NumberOfImagesLabels(FeatureExtractorAbstract):
    def __init__(self):
        super().__init__()
        self._num_images: int = 0
        self._num_labels: int = 0
        self._num_bg_images: int = 0

    def execute(self, data: SegBatchData):
        self._num_images += len(data.images)
        self._num_labels += len(data.labels)
        for i, image_contours in enumerate(data.contours):
            if len(image_contours) < 1:
                self._num_bg_images += 1

    def process(self, ax, train):

        create_bar_plot(ax=ax, data=[self._num_images, self._num_labels, self._num_bg_images],
                        labels=["images", "labels", "background images"], y_label='Total #', title='# Images & Labels',
                        train=train, ticks_rotation=0, color=self.colors[int(train)],
                        yticks=True)
        return {'images': self._num_images,
                'labels': self._num_labels,
                'backgrounds': self._num_bg_images}
