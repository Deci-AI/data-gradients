import numpy as np

from src.utils import BatchData
from src.feature_extractors.segmentation.segmentation_abstract import SegmentationFeatureExtractorAbstract
from src.logger.logger_utils import create_bar_plot


class CountNumObjects(SegmentationFeatureExtractorAbstract):
    def __init__(self):
        super().__init__()
        self._number_of_objects_per_image = dict()
        self._total_objects: int = 0

    def execute(self, data: BatchData):
        for image_contours in data.contours:
            num_objects_in_image = sum([len(cls_contours) for cls_contours in image_contours])
            self._total_objects += num_objects_in_image
            if num_objects_in_image in self._number_of_objects_per_image:
                self._number_of_objects_per_image[num_objects_in_image] += 1
            else:
                self._number_of_objects_per_image.update({num_objects_in_image: 1})

    def process(self, ax, train):

        if len(self._number_of_objects_per_image) > 10:
            self._number_of_objects_per_image = self._into_buckets()

        # values = [((100 * value) / self._total_objects) for value in self._number_of_objects_per_image.values()]
        create_bar_plot(ax, self._number_of_objects_per_image.values(), self._number_of_objects_per_image.keys(),
                        x_label="# Objects in image", y_label="# Of images", title="# Objects per image",
                        train=train, color=self.colors[int(train)])

        ax.grid(visible=True, axis='y')

    def _into_buckets(self):
        bins = [*range(10), *range(10, max(list(self._number_of_objects_per_image.keys())), 5)]
        values = list(self._number_of_objects_per_image.values())
        indexes = np.digitize(values, bins)
        bins += [999]

        indexes_for_bins = np.array([bins[i] for i in indexes])

        hist = dict.fromkeys(bins)

        for b in bins:
            hist[b] = np.count_nonzero(indexes_for_bins == int(b))

        return hist
