from src.feature_extractors.feature_extractor_abstract import FeatureExtractorAbstract


class ImagesResolutions(FeatureExtractorAbstract):
    def __init__(self):
        super().__init__()
        self._res_dict = dict()
        self._channels_last = False

    def execute(self, data):
        for image in data.images:
            res = tuple(image.shape[:-1] if self._channels_last else image.shape[1:])
            if res not in self._res_dict:
                self._res_dict[res] = 1
            else:
                self._res_dict[res] += 1

    def process(self, ax, train):
        pass
        # print('Resolutions dict: ', self._res_dict)
