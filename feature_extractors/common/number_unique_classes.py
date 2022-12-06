from feature_extractors import FeatureExtractorBuilder


class NumberOfUniqueClasses(FeatureExtractorBuilder):
    def __init__(self, train_set):
        super().__init__(train_set)
        self._unique_classes = set()

    def execute(self, data):
        for label in data.labels:
            for val in label.unique():
                self._unique_classes.add(val.item())

    def process(self, ax):
        pass
        # print('Number of unique classes: ', len(self._unique_classes))

