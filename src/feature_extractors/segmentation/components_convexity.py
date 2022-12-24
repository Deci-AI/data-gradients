from src.utils import SegBatchData
from src.feature_extractors.segmentation.segmentation_abstract import SegmentationFeatureExtractorAbstract
from src.logger.logger_utils import create_bar_plot


class ComponentsConvexity(SegmentationFeatureExtractorAbstract):
    """
    Semantic Segmentation task feature extractor -
    """

    def __init__(self):
        super().__init__()

    def execute(self, data: SegBatchData):
        for image_contours in data.contours:
            pass

    def process(self, ax, train):
        pass
