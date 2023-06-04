from data_gradients.common.registry.registry import register_feature_extractor
from data_gradients.feature_extractors.abstract_feature_extractor import AbstractFeatureExtractor
from data_gradients.utils.data_classes.data_samples import DetectionSample
from data_gradients.visualize.plot_options import ImagesRenderer
from data_gradients.visualize.detection import draw_bboxes
from data_gradients.feature_extractors.abstract_feature_extractor import Feature


@register_feature_extractor()
class DetectionSampleVisualization(AbstractFeatureExtractor):
    def __init__(self, n_samples_to_visualize: int = 8, n_cols: int = 2):
        """
        :param n_samples_to_visualize:  Number of samples to visualize.
        :param n_cols:                  Number of columns in the grid.
        """
        self.n_samples_to_visualize = n_samples_to_visualize
        self.n_cols = n_cols
        self.samples = []

    def update(self, sample: DetectionSample):

        if len(self.samples) < self.n_samples_to_visualize:
            image = draw_bboxes(image=sample.image, labels=sample.class_ids, bboxes_xyxy=sample.bboxes_xyxy)
            self.samples.append(image)

    def aggregate(self) -> Feature:
        plot_options = ImagesRenderer(title=self.title, n_cols=self.n_cols)
        feature = Feature(data=self.samples, plot_options=plot_options, json={})
        return feature

    @property
    def title(self) -> str:
        return "Visualization of Samples"

    @property
    def description(self) -> str:
        return (
            f"Visualization of {self.n_samples_to_visualize} random samples from the dataset. "
            f"This can be useful to see how your dataset looks like which can be valuable to understand following features."
        )
