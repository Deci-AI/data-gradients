from matplotlib import pyplot as plt

from data_gradients.feature_extractors.result import FeaturesCollection
from data_gradients.reports.report_template import ReportTemplate
from data_gradients.visualize.seaborn_renderer import SeabornRenderer


class NotebookWriter:
    """
    A writer that plot all matplotlib figures via plt.show() call.
    This makes them appear in the notebook.
    """
    def __init__(self):
        pass

    def write_report(self, results: FeaturesCollection, template: ReportTemplate) -> None:
        sns = SeabornRenderer()
        for widget in template.widgets:
            fig = widget.to_figure(results, sns)

            plt.show()
