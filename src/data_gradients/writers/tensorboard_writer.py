import os

from torch.utils.tensorboard import SummaryWriter

from data_gradients.feature_extractors.result import FeaturesCollection
from data_gradients.reports.report_template import ReportTemplate
from data_gradients.visualize.seaborn_renderer import SeabornRenderer
from data_gradients.writers.abstract_writer import AbstractWriter


class TensorboardWriter(AbstractWriter):
    """
    A writer that writes a report to a markdown file.
    """

    def __init__(self, output_directory: str):
        """
        :param output_file: The output file to which the report should be written.
        :param images_subfolder: The subfolder in which the images should be stored. If None - images stored is the same folder as the report.
                                 If "img" - images stored in the "img" subfolder.
        """
        self.output_directory = os.path.abspath(output_directory)

    def write_report(self, results: FeaturesCollection, template: ReportTemplate) -> None:
        sns = SeabornRenderer()

        with SummaryWriter(self.output_directory) as tb_writer:
            for widget in template.widgets:
                fig = widget.to_figure(results, sns)

                widget_image_name = widget.__class__.__name__
                tb_writer.add_figure(widget_image_name, fig, close=True)
