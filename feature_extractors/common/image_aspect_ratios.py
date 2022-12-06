from feature_extractors import FeatureExtractorBuilder


class ImagesAspectRatios(FeatureExtractorBuilder):
    def __init__(self, train_set):
        super().__init__(train_set)
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

    def process(self, ax):
        pass
        # print('Aspect ratio dict: ', self._ar_dict)

