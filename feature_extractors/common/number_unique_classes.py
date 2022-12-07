from feature_extractors.feature_extractor_abstract import FeatureExtractorAbstract


class NumberOfUniqueClasses(FeatureExtractorAbstract):
    def __init__(self):
        super().__init__()
        self._unique_classes = set()

    def execute(self, data):
        for label in data.labels:
            for val in label.unique():
                self._unique_classes.add(val.item())

    def process(self, ax, train):
        pass
        # print('Number of unique classes: ', len(self._unique_classes))

