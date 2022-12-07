from feature_extractors.feature_extractor_abstract import FeatureExtractorAbstract


class ImagesAspectRatios(FeatureExtractorAbstract):
    def __init__(self):
        super().__init__()
        self._ar_dict = dict()
        self._channels_last = False

    def execute(self, data):
        for image in data.images:
            res = tuple(image.shape[:-1] if self._channels_last else image.shape[1:])
            ar = res[0] / res[1]
            if ar not in self._ar_dict:
                self._ar_dict[ar] = 1
            else:
                self._ar_dict[ar] += 1

    def process(self, ax, train):
        pass
        # print('Aspect ratio dict: ', self._ar_dict)

