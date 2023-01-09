from data_gradients.feature_extractors.feature_extractor_abstract import FeatureExtractorAbstract
from data_gradients.utils import BatchData
from data_gradients.utils.data_classes.extractor_results import Results


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

    def _execute(self, data: BatchData):
        self._num_images[data.split] += len(data.images)
        self._num_labels[data.split] += len(data.labels)
        # TODO: Fix how to get background image if not segmentation
        for i, image_contours in enumerate(data.contours):
            if len(image_contours) < 1:
                self._num_bg_images[data.split] += 1

    def _post_process(self, split):
        values, bins = self._process_data(split)
        results = Results(bins=bins,
                          values=values,
                          plot='bar-plot',
                          split=split,
                          title='# Images & Labels',
                          color=self.colors[split],
                          y_label='Total #',
                          ticks_rotation=0,
                          y_ticks=True)
        return results

    def _process_data(self, split):
        values = [self._num_images[split], self._num_labels[split], self._num_bg_images[split]]
        bins = ["images", "labels", "background images"]
        return values, bins

