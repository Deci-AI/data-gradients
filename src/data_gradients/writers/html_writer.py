import os
from typing import Optional

from matplotlib import pyplot as plt

from data_gradients.feature_extractors.result import FeaturesCollection
from data_gradients.reports.report_template import ReportTemplate
from data_gradients.visualize.seaborn_renderer import SeabornRenderer
from data_gradients.writers.abstract_writer import AbstractWriter


class HTMLWriter(AbstractWriter):
    """
    A writer that writes a report to an HTML file.
    """

    def __init__(self, output_file: str, images_subfolder: Optional[str] = None):
        """
        :param output_file: The output file to which the report should be written.
        :param images_subfolder: The subfolder in which the images should be stored. If None - images stored is the same folder as the report.
                                 If "img" - images stored in the "img" subfolder.
        """
        self.output_file = os.path.abspath(output_file)
        self.images_subfolder = images_subfolder

    def write_report(self, results: FeaturesCollection, template: ReportTemplate) -> None:
        sns = SeabornRenderer()

        output_dir = os.path.dirname(self.output_file)
        os.makedirs(output_dir, exist_ok=True)
        if self.images_subfolder is not None:
            os.makedirs(os.path.join(output_dir, self.images_subfolder), exist_ok=True)

        # Report generation
        with open(self.output_file, "w") as f:
            f.write("<html>\n")
            f.write("<head>\n")
            f.write("<title>Dataset result</title>\n")
            f.write("</head>\n")
            f.write("<body>\n")

            for widget in template.widgets:
                fig = widget.to_figure(results, sns)

                widget_image_name = f"{widget.__class__.__name__}.png"
                relative_image_output_path = (
                    os.path.join(self.images_subfolder, widget_image_name).replace("\\", "/") if self.images_subfolder is not None else widget_image_name
                )

                fig.savefig(os.path.join(output_dir, relative_image_output_path))
                plt.close(fig)

                f.write(f"<h1>{widget.__class__.__name__}</h1>\n")
                f.write("<br>\n")
                f.write(f'<img src="{relative_image_output_path}" >\n')
                f.write("<br>\n")

            f.write("</body>\n")
            f.write("</html>\n")
