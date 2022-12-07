import torch
from feature_extractors.feature_extractor_abstract import FeatureExtractorAbstract


class NumberOfBackgroundImages(FeatureExtractorAbstract):
    def __init__(self):
        super().__init__()
        self._background_counter: int = 0

    def execute(self, data):
        for label in data.labels:
            self._background_counter += 1 if torch.sum(label).item() == 0 else 0

    def process(self, ax, train):
        pass
        # print('Number of background images is: ', self._background_counter)
