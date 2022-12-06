from batch_data import BatchData
from feature_extractors import FeatureExtractorAbstract
from logger.logger_utils import create_bar_plot


class NumberOfImagesLabels(FeatureExtractorAbstract):
    def __init__(self, train_set):
        super().__init__(train_set)
        self._num_images: int = 0
        self._num_labels: int = 0

    def execute(self, data: BatchData):
        self._num_images += len(data.images)
        self._num_labels += len(data.labels)

    def process(self, ax):
        create_bar_plot(ax=ax, data=[self._num_images, self._num_labels], labels=["images", "labels"],
                        y_label='Total #', title='# Images & Labels', train=self.train_set,
                        ticks_rotation=0, color=self.colors[int(self.train_set)])
