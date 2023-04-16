import numpy as np

from data_gradients.utils import SegBatchData
from data_gradients.feature_extractors.feature_extractor_abstract import (
    FeatureExtractorAbstract,
)
from data_gradients.utils.data_classes.extractor_results import Results


class CountNumComponents(FeatureExtractorAbstract):
    """
    Semantic Segmentation task feature extractor -
    Count number of objects in each image, over all classes.
    For better display, show bins of 0-10 objects and then show bins with size of num_bins.
    Full histogram will be with X-axis of [0, 1, ... 10, 11-15, 16-20, ... 81-85, 85+]
    """

    def __init__(self):
        super().__init__()
        self._hist = {"train": dict(), "val": dict()}
        self._total_objects = {"train": 0, "val": 0}

    def _execute(self, data: SegBatchData):
        for image_contours in data.contours:
            num_objects_in_image = sum(
                [len(cls_contours) for cls_contours in image_contours]
            )
            self._total_objects[data.split] += num_objects_in_image
            if num_objects_in_image in self._hist[data.split]:
                self._hist[data.split][num_objects_in_image] += 1
            else:

                self._hist[data.split].update({num_objects_in_image: 1})

    def _post_process(self, split: str):
        values, bins = self._process_data(split)
        results = Results(
            bins=bins,
            values=values,
            plot="bar-plot",
            split=split,
            color=self.colors[split],
            title="# Components per image",
            x_label="# Components in image",
            y_label="% Of Images",
            y_ticks=True,
            ax_grid=True,
        )

        return results

    def _process_data(self, split: str):
        self.merge_dict_splits(self._hist)
        hist = self._into_buckets(self._hist[split])
        values = self.normalize(hist.values(), sum(list(hist.values())))
        bins = hist.keys()
        return values, bins

    @staticmethod
    def _into_buckets(number_of_objects_per_image):
        # TODO: Refactor this code
        if len(number_of_objects_per_image) < 10:
            return number_of_objects_per_image
        min_bin = min(list(number_of_objects_per_image.keys()))
        max_bin = int(
            np.average((sorted(list(number_of_objects_per_image.keys())))[-10:])
        )

        bin_size = int(5 + 5 * (int(max_bin / 50)))

        bins = [
            *range(min_bin - 1, 10),
            *range(10, max(list(number_of_objects_per_image.keys())), bin_size),
        ]

        indexes = np.digitize(list(number_of_objects_per_image.keys()), bins)

        bins += [999]
        indexes_for_bins = np.array([bins[i] for i in indexes])

        hist = dict.fromkeys(bins, 0)

        for i, (key, value) in enumerate(number_of_objects_per_image.items()):
            hist[indexes_for_bins[i]] += value

        keys = list(hist.keys())
        for key in keys:
            if key == 999:
                hist[f"{bins[-2]}+"] = hist[999]
                del hist[999]
            elif key > 10:
                new_key = f"{key-bin_size}<{key}"
                if key - bin_size > 0:
                    hist[new_key] = hist[key]
                    del hist[key]
            else:
                continue

        return hist
