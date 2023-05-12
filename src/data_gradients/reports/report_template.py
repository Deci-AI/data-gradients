from typing import List

from data_gradients.feature_extractors.result import FeaturesCollection
from data_gradients.reports import AbstractReportWidget


class ReportTemplate:
    """
    A report template is a collection of report widgets that constitute a report.
    """

    widgets: List[AbstractReportWidget]

    def __init__(self, widgets: List[AbstractReportWidget]):
        self.widgets = widgets

    @classmethod
    def get_report_template_with_valid_widgets(cls, result: FeaturesCollection) -> "ReportTemplate":
        """
        Returns a default report template that includes all widgets that are available for the given result.
        If the result contains segmentation features, segmentation widgets are included.

        :param result: The result for which the report template should be created.
        :return: An instance of ReportTemplate.
        """
        from data_gradients.reports import (
            ImageSizeDistribution,
            AverageImageBrightness,
            SegmentationClassDistribution,
            SegmentationMasksArea,
            DatasetSplitDistribution,
        )

        widgets = [
            ImageSizeDistribution(),
            DatasetSplitDistribution(),
            AverageImageBrightness(),
        ]

        if result.mask_features is not None:
            widgets += [
                SegmentationClassDistribution(),
                SegmentationMasksArea(),
            ]

        return ReportTemplate(widgets)

    @classmethod
    def from_config(cls, config) -> "ReportTemplate":
        ...
