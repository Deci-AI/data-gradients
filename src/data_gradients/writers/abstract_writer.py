from abc import abstractmethod, ABC

from data_gradients.feature_extractors.result import FeaturesCollection
from data_gradients.reports.report_template import ReportTemplate


class AbstractWriter(ABC):
    """
    A writer that writes a report to an HTML file.
    """

    @abstractmethod
    def write_report(self, results: FeaturesCollection, template: ReportTemplate) -> None:
        ...
