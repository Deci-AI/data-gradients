import numpy as np

from src.utils import SegBatchData
from src.feature_extractors.segmentation.segmentation_abstract import SegmentationFeatureExtractorAbstract
from src.logger.logger_utils import create_bar_plot


class CountNumObjects(SegmentationFeatureExtractorAbstract):
    """
    Semantic Segmentation task feature extractor -
    Count number of objects in each image, over all classes.
    For better display, show bins of 0-10 objects and then show bins with size of 5.
    Full histogram will be with X-axis of [0, 1, ... 10, 11-15, 16-20, ... 81-85, 85+]
    """
    def __init__(self):
        super().__init__()
        self._number_of_objects_per_image = dict()
        self._total_objects: int = 0
        self._bin_size: int = 5

    def execute(self, data: SegBatchData):
        for image_contours in data.contours:
            num_objects_in_image = sum([len(cls_contours) for cls_contours in image_contours])
            self._total_objects += num_objects_in_image
            if num_objects_in_image in self._number_of_objects_per_image:
                self._number_of_objects_per_image[num_objects_in_image] += 1
            else:

                self._number_of_objects_per_image.update({num_objects_in_image: 1})

    def process(self, ax, train):

        hist = self._into_buckets()

        values = self.normalize(hist.values(), sum(list(hist.values())))

        create_bar_plot(ax, values, hist.keys(), x_label="# Objects in image", y_label="% Of Images",
                        title="# Objects per image", train=train, color=self.colors[int(train)], yticks=True)

        ax.grid(visible=True, axis='y')
        return dict(zip(hist.keys(), hist.values()))

    def _into_buckets(self):
        if len(self._number_of_objects_per_image) < 10:
            return self._number_of_objects_per_image

        bins = [*range(10), *range(10, max(list(self._number_of_objects_per_image.keys())), self._bin_size)]

        indexes = np.digitize(list(self._number_of_objects_per_image.keys()), bins)

        bins += [999]
        indexes_for_bins = np.array([bins[i] for i in indexes])

        hist = dict.fromkeys(bins, 0)

        for i, (key, value) in enumerate(self._number_of_objects_per_image.items()):
            hist[indexes_for_bins[i]] += value

        keys = list(hist.keys())
        for key in keys:
            if key == 999:
                hist[f'{bins[-2]}+'] = hist[999]
                del hist[999]
            elif key > 10:
                new_key = f'{key-5}<{key}'
                hist[new_key] = hist[key]
                del hist[key]
            else:
                continue

        return hist
